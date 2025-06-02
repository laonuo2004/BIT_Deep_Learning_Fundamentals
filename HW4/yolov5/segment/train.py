# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv5 segment model on a segment dataset Models and datasets download automatically from the latest YOLOv5
release.

Usage - Single-GPU training:
    $ python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640  # from pretrained (recommended)
    $ python segment/train.py --data coco128-seg.yaml --weights '' --cfg yolov5s-seg.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import segment.val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_imshow, # 导入了但未使用
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html # 获取当前进程在分布式训练中的本地排名
RANK = int(os.getenv("RANK", -1)) # 获取当前进程在分布式训练中的全局排名
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1)) # 获取分布式训练中的进程总数
GIT_INFO = check_git_info() # 获取Git信息


def train(hyp, opt, device, callbacks):
    """
    Trains the YOLOv5 model on a dataset, managing hyperparameters, model optimization, logging, and validation.

    `hyp` is path/to/hyp.yaml or hyp dictionary.
    """
    (
        save_dir,
        epochs,
        batch_size,
        weights,
        single_cls,
        evolve,
        data,
        cfg,
        resume,
        noval,
        nosave,
        workers,
        freeze,
        mask_ratio,
    ) = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
        opt.mask_ratio,
    )
    # callbacks.run('on_pretrain_routine_start') # 回调函数：训练例程开始前

    # Directories
    w = save_dir / "weights"  # weights dir # 权重保存目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir # 创建目录
    last, best = w / "last.pt", w / "best.pt" # 最新和最佳模型权重文件路径

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict # 加载超参数字典
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items())) # 打印超参数
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints # 将超参数保存到opt中，用于检查点保存

    # Save run settings
    if not evolve: # 如果不是超参数演化模式
        yaml_save(save_dir / "hyp.yaml", hyp) # 保存超参数到yaml文件
        yaml_save(save_dir / "opt.yaml", vars(opt)) # 保存训练选项到yaml文件

    # Loggers
    data_dict = None
    if RANK in {-1, 0}: # 如果是主进程（单GPU或DDP的第一个进程）
        logger = GenericLogger(opt=opt, console_logger=LOGGER) # 初始化通用日志记录器

    # Config
    plots = not evolve and not opt.noplots  # create plots # 是否创建绘图
    overlap = not opt.no_overlap # 是否重叠掩码
    cuda = device.type != "cpu" # 是否使用CUDA
    init_seeds(opt.seed + 1 + RANK, deterministic=True) # 初始化随机种子
    with torch_distributed_zero_first(LOCAL_RANK): # 确保在分布式训练中，只有主进程执行此块
        data_dict = data_dict or check_dataset(data)  # check if None # 检查数据集配置
    train_path, val_path = data_dict["train"], data_dict["val"] # 训练集和验证集路径
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes # 类别数量
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset # 是否为COCO数据集

    # Model
    check_suffix(weights, ".pt")  # check weights # 检查权重文件后缀
    pretrained = weights.endswith(".pt") # 是否使用预训练权重
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally # 下载权重文件
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak # 加载检查点到CPU
        model = SegmentationModel(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device) # 创建分割模型
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys # 排除的键
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32 # 检查点状态字典转换为FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect # 交叉字典，只保留模型中存在的键
        model.load_state_dict(csd, strict=False)  # load # 加载状态字典
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report # 报告加载情况
    else:
        model = SegmentationModel(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create # 从头创建分割模型
    amp = check_amp(model)  # check AMP # 检查是否支持自动混合精度训练

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze # 需要冻结的层
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers # 默认所有层都可训练
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze): # 如果当前层在冻结列表中
            LOGGER.info(f"freezing {k}") # 打印冻结信息
            v.requires_grad = False # 冻结该层

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride) # 网格大小（最大步长）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple # 验证图像尺寸是网格大小的倍数

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size # 单GPU模式下，估计最佳批处理大小
        batch_size = check_train_batch_size(model, imgsz, amp) # 检查训练批处理大小
        logger.update_params({"batch_size": batch_size}) # 更新日志参数
        # loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size # 标称批处理大小
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing # 梯度累积步数
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay # 缩放权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"]) # 初始化优化器

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf'] # 余弦退火学习率调度器
    else:

        def lf(x):
            """Linear learning rate scheduler decreasing from 1 to hyp['lrf'] over 'epochs'."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear # 线性学习率调度器

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs) # 创建学习率调度器

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None # 初始化模型EMA（指数移动平均）

    # Resume
    best_fitness, start_epoch = 0.0, 0 # 初始化最佳适应度和起始epoch
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume) # 智能恢复训练
        del ckpt, csd # 删除检查点和状态字典，释放内存

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1: # 如果是单机多卡DP模式
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model) # 使用DataParallel

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1: # 如果使用SyncBatchNorm且是DDP模式
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) # 转换SyncBatchNorm
        LOGGER.info("Using SyncBatchNorm()") # 打印信息

    # Trainloader
    train_loader, dataset = create_dataloader( # 创建训练数据加载器
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        mask_downsample_ratio=mask_ratio,
        overlap_mask=overlap,
    )
    labels = np.concatenate(dataset.labels, 0) # 连接所有标签
    mlc = int(labels[:, 0].max())  # max label class # 最大标签类别
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}" # 检查标签类别是否超出范围

    # Process 0
    if RANK in {-1, 0}: # 如果是主进程
        val_loader = create_dataloader( # 创建验证数据加载器
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            mask_downsample_ratio=mask_ratio,
            overlap_mask=overlap,
            prefix=colorstr("val: "),
        )[0]

        if not resume: # 如果不是恢复训练
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor # 运行自动锚框检查
            model.half().float()  # pre-reduce anchor precision # 预先降低锚框精度

            if plots:
                plot_labels(labels, names, save_dir) # 绘制标签分布图
        # callbacks.run('on_pretrain_routine_end', labels, names) # 回调函数：训练例程结束

    # DDP mode
    if cuda and RANK != -1: # 如果是DDP模式
        model = smart_DDP(model) # 使用智能DDP封装模型

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) # 检测层数量
    hyp["box"] *= 3 / nl  # scale to layers # 缩放box损失权重
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers # 缩放cls损失权重
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers # 缩放obj损失权重
    hyp["label_smoothing"] = opt.label_smoothing # 标签平滑
    model.nc = nc  # attach number of classes to model # 将类别数量附加到模型
    model.hyp = hyp  # attach hyperparameters to model # 将超参数附加到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights # 附加类别权重
    model.names = names # 附加类别名称

    # Start training
    t0 = time.time() # 记录训练开始时间
    nb = len(train_loader)  # number of batches # 批次数量
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations) # 热身迭代次数
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1 # 上一次优化步骤
    maps = np.zeros(nc)  # mAP per class # 每类mAP
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls) # 结果指标
    scheduler.last_epoch = start_epoch - 1  # do not move # 设置学习率调度器的last_epoch
    scaler = torch.cuda.amp.GradScaler(enabled=amp) # 初始化梯度缩放器，用于混合精度训练
    stopper, stop = EarlyStopping(patience=opt.patience), False # 初始化早停机制
    compute_loss = ComputeLoss(model, overlap=overlap)  # init loss class # 初始化损失计算类
    # callbacks.run('on_train_start') # 回调函数：训练开始
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    ) # 打印训练信息
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # callbacks.run('on_train_epoch_start') # 回调函数：训练epoch开始
        model.train() # 设置模型为训练模式

        # Update image weights (optional, single-GPU only)
        if opt.image_weights: # 如果使用图像权重
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights # 类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights # 图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx # 随机加权选择索引

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses # 平均损失
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch) # 在DDP模式下设置采样器epoch
        pbar = enumerate(train_loader) # 遍历训练数据加载器
        LOGGER.info(
            ("\n" + "%11s" * 8)
            % ("Epoch", "GPU_mem", "box_loss", "seg_loss", "obj_loss", "cls_loss", "Instances", "Size")
        ) # 打印训练进度条头部信息
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar # 创建进度条
        optimizer.zero_grad() # 梯度清零
        for i, (imgs, targets, paths, _, masks) in pbar:  # batch ------------------------------------------------------
            # callbacks.run('on_train_batch_start') # 回调函数：训练批次开始
            ni = i + nb * epoch  # number integrated batches (since train start) # 累计批次数量
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0 # 图像预处理

            # Warmup
            if ni <= nw: # 热身阶段
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round()) # 梯度累积步数
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)]) # 学习率调整
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]) # 动量调整

            # Multi-scale
            if opt.multi_scale: # 多尺度训练
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size # 随机选择图像尺寸
                sf = sz / max(imgs.shape[2:])  # scale factor # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple) # 新的图像尺寸
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False) # 图像插值

            # Forward
            with torch.cuda.amp.autocast(amp): # 自动混合精度
                pred = model(imgs)  # forward # 前向传播
                loss, loss_items = compute_loss(pred, targets.to(device), masks=masks.to(device).float()) # 计算损失
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode # DDP模式下损失平均
                if opt.quad:
                    loss *= 4.0 # 四倍损失

            # Backward
            scaler.scale(loss).backward() # 损失反向传播

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate: # 达到梯度累积步数
                scaler.unscale_(optimizer)  # unscale gradients # 解除梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients # 梯度裁剪
                scaler.step(optimizer)  # optimizer.step # 优化器步进
                scaler.update() # 更新梯度缩放器
                optimizer.zero_grad() # 梯度清零
                if ema:
                    ema.update(model) # 更新EMA模型
                last_opt_step = ni # 更新上一次优化步骤

            # Log
            if RANK in {-1, 0}: # 如果是主进程
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses # 更新平均损失
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB) # GPU内存使用
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 6)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                ) # 更新进度条描述
                # callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths)
                # if callbacks.stop_training:
                #    return

                # Mosaic plots
                if plots: # 如果需要绘图
                    if ni < 3:
                        plot_images_and_masks(imgs, targets, masks, paths, save_dir / f"train_batch{ni}.jpg") # 绘制图像和掩码
                    if ni == 10:
                        files = sorted(save_dir.glob("train*.jpg"))
                        logger.log_images(files, "Mosaics", epoch) # 记录马赛克图像
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers # 获取学习率
        scheduler.step() # 学习率调度器步进

        if RANK in {-1, 0}: # 如果是主进程
            # mAP
            # callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"]) # 更新EMA模型属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop # 是否为最后一轮或可能早停
            if not noval or final_epoch:  # Calculate mAP # 计算mAP
                results, maps, _ = validate.run( # 运行验证
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                    mask_downsample_ratio=mask_ratio,
                    overlap=overlap,
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95] # 计算适应度
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check # 早停检查
            if fi > best_fitness:
                best_fitness = fi # 更新最佳适应度
            log_vals = list(mloss) + list(results) + lr # 记录值
            # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            # Log val metrics and media
            metrics_dict = dict(zip(KEYS, log_vals)) # 创建指标字典
            logger.log_metrics(metrics_dict, epoch) # 记录指标

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save # 如果需要保存模型
                ckpt = { # 创建检查点字典
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last) # 保存最新检查点
                if best_fitness == fi:
                    torch.save(ckpt, best) # 保存最佳检查点
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt") # 按周期保存检查点
                    logger.log_model(w / f"epoch{epoch}.pt") # 记录模型
                del ckpt # 删除检查点，释放内存
                # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks # 在DDP模式下广播早停信号
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks # 如果早停，则中断所有DDP进程

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}: # 如果是主进程
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.") # 打印训练时长
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers # 剥离优化器信息
                if f is best:
                    LOGGER.info(f"\nValidating {f}...") # 验证最佳模型
                    results, _, _ = validate.run( # 运行验证
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        mask_downsample_ratio=mask_ratio,
                        overlap=overlap,
                    )  # val best model with plots
                    if is_coco:
                        # callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
                        metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr)) # 创建指标字典
                        logger.log_metrics(metrics_dict, epoch) # 记录指标

        # callbacks.run('on_train_end', last, best, epoch, results)
        # on train end callback using genericLogger
        logger.log_metrics(dict(zip(KEYS[4:16], results)), epochs) # 记录最终指标
        if not opt.evolve:
            logger.log_model(best, epoch) # 记录最佳模型
        if plots:
            plot_results_with_masks(file=save_dir / "results.csv")  # save results.png # 绘制结果图
            files = ["results.png", "confusion_matrix.png", *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R"))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}") # 打印结果保存路径
            logger.log_images(files, "Results", epoch + 1) # 记录结果图像
            logger.log_images(sorted(save_dir.glob("val*.jpg")), "Validation", epoch + 1) # 记录验证图像
    torch.cuda.empty_cache() # 清空CUDA缓存
    return results


def parse_opt(known=False):
    """
    Parses command line arguments for training configurations, returning parsed arguments.

    Supports both known and unknown args.
    """
    parser = argparse.ArgumentParser() # 创建命令行参数解析器
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s-seg.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-seg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Instance Segmentation Args
    parser.add_argument("--mask-ratio", type=int, default=4, help="Downsample the truth masks to saving memory") # 掩码下采样比例
    parser.add_argument("--no-overlap", action="store_true", help="Overlap masks train faster at slightly less mAP") # 是否不重叠掩码

    return parser.parse_known_args()[0] if known else parser.parse_args() # 解析参数


def main(opt, callbacks=Callbacks()):
    """Initializes training or evolution of YOLOv5 models based on provided configuration and options."""
    if RANK in {-1, 0}: # 如果是主进程
        print_args(vars(opt)) # 打印参数
        check_git_status() # 检查Git状态
        check_requirements(ROOT / "requirements.txt") # 检查依赖项

    # Resume
    if opt.resume and not opt.evolve:  # resume from specified or most recent last.pt # 恢复训练
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run()) # 获取最新运行的检查点
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml # 训练选项yaml文件
        opt_data = opt.data  # original dataset # 原始数据集
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f) # 加载yaml文件
        else:
            d = torch.load(last, map_location="cpu")["opt"] # 从检查点加载选项
        opt = argparse.Namespace(**d)  # replace # 替换选项
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate # 重新设置cfg, weights, resume
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout # 避免HUB恢复认证超时
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks # 检查文件和yaml
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified" # 必须指定cfg或weights
        if opt.evolve: # 如果是超参数演化模式
            if opt.project == str(ROOT / "runs/train-seg"):  # if default project name, rename to runs/evolve-seg
                opt.project = str(ROOT / "runs/evolve-seg") # 更改项目名称
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume # 传递resume到exist_ok并禁用resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name # 使用model.yaml作为名称
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # 设置保存目录

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size) # 选择设备
    if LOCAL_RANK != -1: # 如果是DDP模式
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}" # 检查image_weights兼容性
        assert not opt.evolve, f"--evolve {msg}" # 检查evolve兼容性
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size" # 检查batch_size
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE" # 检查batch_size是否为WORLD_SIZE的倍数
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command" # 检查CUDA设备数量
        torch.cuda.set_device(LOCAL_RANK) # 设置CUDA设备
        device = torch.device("cuda", LOCAL_RANK) # 设置设备
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo") # 初始化分布式进程组

    # Train
    if not opt.evolve: # 如果不是超参数演化模式
        train(opt.hyp, opt, device, callbacks) # 调用train函数进行训练

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = { # 超参数演化元数据
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (1, 0.0, 1.0),
        }  # segment copy-paste (probability)

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict # 加载超参数字典
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"] # 如果禁用自动锚框，则删除相关参数
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch # 只在最终epoch验证/保存
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv" # 演化结果文件路径
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            ) # 从GCS下载演化结果

        for _ in range(opt.evolve):  # generations to evolve # 演化代数
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate # 如果演化结果文件存在，则选择最佳超参数并变异
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted' # 父代选择方法
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1) # 加载演化结果
                n = min(5, len(x))  # number of previous results to consider # 考虑的先前结果数量
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations # 选择前n个最佳变异
                w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0) # 计算权重
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection # 加权选择父代
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination # 加权组合父代

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma # 变异概率和标准差
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1 # 增益
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0) # 变异
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 12] * v[i])  # mutate # 应用变异

            # Constrain to limits
            for k, v in meta.items(): # 限制超参数在指定范围内
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks) # 训练变异后的超参数
            callbacks = Callbacks() # 重置回调
            # Write mutation results
            print_mutation(KEYS[4:16], results, hyp.copy(), save_dir, opt.bucket) # 打印变异结果

        # Plot results
        plot_evolve(evolve_csv) # 绘制演化结果
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python train.py --hyp {evolve_yaml}"
        ) # 打印演化完成信息


def run(**kwargs):
    """
    Executes YOLOv5 training with given parameters, altering options programmatically; returns updated options.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True) # 解析参数
    for k, v in kwargs.items():
        setattr(opt, k, v) # 设置选项
    main(opt) # 调用main函数
    return opt


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 执行主函数
