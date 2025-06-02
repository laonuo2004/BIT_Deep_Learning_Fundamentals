# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

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
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
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
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
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

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Train a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        None

    Models and datasets download automatically from the latest YOLOv5 release.

    Example:
        Single-GPU training:
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    # 从opt中解析训练参数
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
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
    )
    callbacks.run("on_pretrain_routine_start")

    # 目录设置
    w = save_dir / "weights"  # 权重保存目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 创建目录
    last, best = w / "last.pt", w / "best.pt" # 最新和最佳权重文件路径

    # 超参数加载
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # 加载超参数字典
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # 用于将超参数保存到检查点

    # 保存运行设置
    if not evolve: # 如果不是超参数进化模式
        yaml_save(save_dir / "hyp.yaml", hyp) # 保存超参数
        yaml_save(save_dir / "opt.yaml", vars(opt)) # 保存训练选项

    # 日志记录器初始化
    data_dict = None
    if RANK in {-1, 0}: # 主进程或单GPU模式
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # 注册回调动作
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # 处理自定义数据集artifact链接
        data_dict = loggers.remote_dataset
        if resume:  # 如果从远程artifact恢复训练
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # 配置训练环境
    plots = not evolve and not opt.noplots  # 是否生成训练图表
    cuda = device.type != "cpu" # 是否使用CUDA
    init_seeds(opt.seed + 1 + RANK, deterministic=True) # 初始化随机种子
    with torch_distributed_zero_first(LOCAL_RANK): # 确保在分布式训练中，只有主进程执行数据检查
        data_dict = data_dict or check_dataset(data)  # 检查数据集配置
    train_path, val_path = data_dict["train"], data_dict["val"] # 训练集和验证集路径
    nc = 1 if single_cls else int(data_dict["nc"])  # 类别数量
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # 判断是否为COCO数据集

    # 模型构建与加载
    check_suffix(weights, ".pt")  # 检查权重文件后缀
    pretrained = weights.endswith(".pt") # 是否使用预训练权重
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # 下载权重文件
        ckpt = torch.load(weights, map_location="cpu")  # 加载检查点到CPU
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # 创建模型
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # 排除的键
        csd = ckpt["model"].float().state_dict()  # 检查点状态字典 (FP32)
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 交集操作，匹配模型和检查点中的键
        model.load_state_dict(csd, strict=False)  # 加载状态字典
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # 报告加载情况
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # 从头创建模型
    amp = check_amp(model)  # 检查是否支持自动混合精度训练 (AMP)

    # 冻结层
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # 需要冻结的层
    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认所有层都可训练
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN转0 (注释掉，因为可能导致训练结果不稳定)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False # 冻结指定层

    # 图像尺寸
    gs = max(int(model.stride.max()), 32)  # 网格大小 (最大步长)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 验证图像尺寸是gs的倍数

    # 批处理大小
    if RANK == -1 and batch_size == -1:  # 仅单GPU模式，估计最佳批处理大小
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # 优化器
    nbs = 64  # 标称批处理大小
    accumulate = max(round(nbs / batch_size), 1)  # 梯度累积步数
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # 缩放权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # 学习率调度器
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # 余弦退火学习率调度器
    else:
        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # 线性学习率调度器

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器

    # EMA (指数移动平均)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # 恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd # 删除检查点和状态字典以释放内存

    # DP模式 (数据并行)
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model) # 使用DataParallel

    # SyncBatchNorm (同步批归一化)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # 训练数据加载器
    train_loader, dataset = create_dataloader(
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
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # 最大标签类别
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # 主进程操作 (RANK -1 或 0)
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
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
            prefix=colorstr("val: "),
        )[0]

        if not resume: # 如果不是恢复训练
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # 运行AutoAnchor
            model.half().float()  # 预先降低锚点精度

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP模式 (分布式数据并行)
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # 模型属性设置
    nl = de_parallel(model).model[-1].nl  # 检测层数量 (用于缩放超参数)
    hyp["box"] *= 3 / nl  # 缩放box损失增益
    hyp["cls"] *= nc / 80 * 3 / nl  # 缩放cls损失增益
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # 缩放obj损失增益
    hyp["label_smoothing"] = opt.label_smoothing # 标签平滑
    model.nc = nc  # 附加类别数量到模型
    model.hyp = hyp  # 附加超参数到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 附加类别权重
    model.names = names # 附加类别名称

    # 开始训练
    t0 = time.time() # 记录开始时间
    nb = len(train_loader)  # 批次数量
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # 热身迭代次数
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制热身阶段不超过训练周期的一半
    last_opt_step = -1
    maps = np.zeros(nc)  # 每类mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 学习率调度器从正确纪元开始
    scaler = torch.cuda.amp.GradScaler(enabled=amp) # AMP梯度缩放器
    stopper, stop = EarlyStopping(patience=opt.patience), False # 早停机制
    compute_loss = ComputeLoss(model)  # 初始化损失计算类
    callbacks.run("on_train_start") # 触发训练开始回调
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    for epoch in range(start_epoch, epochs):  # 训练循环 (epoch) ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start") # 触发训练纪元开始回调
        model.train() # 设置模型为训练模式

        # 更新图像权重 (可选，仅单GPU)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机加权索引

        # 更新mosaic边界 (可选)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # 平均损失 (box, obj, cls)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch) # 分布式训练中设置采样器纪元
        pbar = enumerate(train_loader) # 遍历训练数据加载器
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # 进度条
        optimizer.zero_grad() # 梯度清零
        for i, (imgs, targets, paths, _) in pbar:  # 批次循环 -------------------------------------------------------------
            callbacks.run("on_train_batch_start") # 触发训练批次开始回调
            ni = i + nb * epoch  # 已处理的批次总数
            imgs = imgs.to(device, non_blocking=True).float() / 255  # 图像数据类型转换和归一化

            # 热身阶段
            if ni <= nw:
                xi = [0, nw]  # x插值范围
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou损失比率
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round()) # 梯度累积步数
                for j, x in enumerate(optimizer.param_groups):
                    # 偏置学习率从0.1下降到lr0，其他学习率从0.0上升到lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # 随机图像尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新形状 (拉伸到gs的倍数)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False) # 插值缩放图像

            # 前向传播
            with torch.cuda.amp.autocast(amp): # 自动混合精度
                pred = model(imgs)  # 模型前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 计算损失
                if RANK != -1:
                    loss *= WORLD_SIZE  # DDP模式下梯度在设备间平均
                if opt.quad:
                    loss *= 4.0 # 四边形数据加载器损失缩放

            # 反向传播
            scaler.scale(loss).backward() # 损失反向传播并缩放梯度

            # 优化器步进
            if ni - last_opt_step >= accumulate: # 达到梯度累积步数
                scaler.unscale_(optimizer)  # 取消梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪
                scaler.step(optimizer)  # 优化器步进
                scaler.update() # 更新缩放器
                optimizer.zero_grad() # 梯度清零
                if ema:
                    ema.update(model) # 更新EMA模型
                last_opt_step = ni # 更新上次优化步进的批次索引

            # 日志记录
            if RANK in {-1, 0}: # 主进程或单GPU模式
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # GPU内存使用 (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss)) # 触发训练批次结束回调
                if callbacks.stop_training:
                    return
            # 批次循环结束 ------------------------------------------------------------------------------------------------

        # 学习率调度器步进
        lr = [x["lr"] for x in optimizer.param_groups]  # 获取当前学习率
        scheduler.step() # 学习率调度器步进

        if RANK in {-1, 0}: # 主进程或单GPU模式
            # mAP评估
            callbacks.run("on_train_epoch_end", epoch=epoch) # 触发训练纪元结束回调
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"]) # 更新EMA模型属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop # 是否为最终纪元或早停
            if not noval or final_epoch:  # 如果需要验证或已是最终纪元
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
                )

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 综合评估指标
            stop = stopper(epoch=epoch, fitness=fi)  # 早停检查
            if fi > best_fitness:
                best_fitness = fi # 更新最佳fitness
            log_vals = list(mloss) + list(results) + lr # 记录日志值
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi) # 触发拟合纪元结束回调

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):  # 如果需要保存
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(), # 保存模型状态
                    "ema": deepcopy(ema.ema).half(), # 保存EMA模型状态
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(), # 保存优化器状态
                    "opt": vars(opt), # 保存训练选项
                    "git": GIT_INFO,  # Git信息
                    "date": datetime.now().isoformat(), # 日期
                }

                # 保存last.pt, best.pt并删除检查点
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt # 删除检查点以释放内存
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi) # 触发模型保存回调

        # 早停机制 (DDP模式下广播停止信号)
        if RANK != -1:  # 如果是DDP训练
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # 广播'stop'到所有进程
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # 所有DDP进程必须中断

        # 纪元循环结束 ----------------------------------------------------------------------------------------------------
    # 训练结束 -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}: # 主进程或单GPU模式
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 剥离优化器状态，减小模型大小
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run( # 验证最佳模型并生成图表
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # COCO数据集使用0.65，其他使用0.60
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results) # 触发训练结束回调

    torch.cuda.empty_cache() # 清空CUDA缓存
    return results


def parse_opt(known=False):
    """
    Parse command-line arguments for YOLOv5 training, validation, and testing.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.

    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        print(opt)
        ```

    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    # 定义各种命令行参数
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
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
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
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

    # 日志记录器参数
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON日志记录
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Runs the main entry point for training or hyperparameter evolution with specified options and optional callbacks.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None

    Note:
        For detailed usage, refer to:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    if RANK in {-1, 0}: # 主进程或单GPU模式
        print_args(vars(opt)) # 打印所有参数
        check_git_status() # 检查Git状态
        check_requirements(ROOT / "requirements.txt") # 检查依赖

    # 恢复训练 (从指定或最近的last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run()) # 获取最近的运行或指定检查点
        opt_yaml = last.parent.parent / "opt.yaml"  # 训练选项yaml文件
        opt_data = opt.data  # 原始数据集
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # 替换当前opt
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # 重新设置配置、权重和恢复标志
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # 避免HUB恢复认证超时
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # 检查文件和yaml配置
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified" # 必须指定模型配置或权重
        if opt.evolve: # 如果是超参数进化模式
            if opt.project == str(ROOT / "runs/train"):  # 如果是默认项目名，则重命名为runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # 将resume传递给exist_ok并禁用resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # 使用model.yaml的文件名作为项目名
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # 设置保存目录

    # DDP模式 (分布式数据并行)
    device = select_device(opt.device, batch_size=opt.batch_size) # 选择设备
    if LOCAL_RANK != -1: # 如果是DDP模式
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK) # 设置当前GPU设备
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group( # 初始化分布式进程组
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # 训练或超参数进化
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks) # 调用训练函数

    # 超参数进化 (可选)
    else:
        # 超参数进化元数据 (包括是否进化、下限、上限)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # 最终OneCycleLR学习率 (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD动量/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # 优化器权重衰减
            "warmup_epochs": (False, 0.0, 5.0),  # 热身纪元 (可为小数)
            "warmup_momentum": (False, 0.0, 0.95),  # 热身初始动量
            "warmup_bias_lr": (False, 0.0, 0.2),  # 热身初始偏置学习率
            "box": (False, 0.02, 0.2),  # box损失增益
            "cls": (False, 0.2, 4.0),  # cls损失增益
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss正样本权重
            "obj": (False, 0.2, 4.0),  # obj损失增益 (随像素缩放)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss正样本权重
            "iou_t": (False, 0.1, 0.7),  # IoU训练阈值
            "anchor_t": (False, 2.0, 8.0),  # 锚点倍数阈值
            "anchors": (False, 2.0, 10.0),  # 每个输出网格的锚点数量 (0表示忽略)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet默认gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # 图像HSV-色调增强 (比例)
            "hsv_s": (True, 0.0, 0.9),  # 图像HSV-饱和度增强 (比例)
            "hsv_v": (True, 0.0, 0.9),  # 图像HSV-亮度增强 (比例)
            "degrees": (True, 0.0, 45.0),  # 图像旋转 (+/- 度)
            "translate": (True, 0.0, 0.9),  # 图像平移 (+/- 比例)
            "scale": (True, 0.0, 0.9),  # 图像缩放 (+/- 增益)
            "shear": (True, 0.0, 10.0),  # 图像剪切 (+/- 度)
            "perspective": (True, 0.0, 0.001),  # 图像透视 (+/- 比例), 范围0-0.001
            "flipud": (True, 0.0, 1.0),  # 图像上下翻转 (概率)
            "fliplr": (True, 0.0, 1.0),  # 图像左右翻转 (概率)
            "mosaic": (True, 0.0, 1.0),  # 图像mosaic (概率)
            "mixup": (True, 0.0, 1.0),  # 图像mixup (概率)
            "copy_paste": (True, 0.0, 1.0),  # 分割copy-paste (概率)
        }

        # 遗传算法配置
        pop_size = 50 # 种群大小
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # 加载超参数字典
            if "anchors" not in hyp:  # 如果anchors在hyp.yaml中被注释
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"] # 禁用AutoAnchor时删除相关参数
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 仅在最终纪元验证/保存
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 可进化的索引
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv" # 进化结果保存路径
        if opt.bucket:
            # 如果存在，下载evolve.csv
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # 删除meta字典中第一个值为False的项
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # 复制超参数字典
        for item in del_:
            del meta[item]  # 从meta字典中移除
            del hyp_GA[item]  # 从hyp_GA字典中移除

        # 设置lower_limit和upper_limit数组，用于定义搜索空间边界
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # 创建gene_ranges列表，用于存储种群中每个基因的值范围
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # 初始化种群
        initial_values = []

        # 如果从之前的检查点恢复进化
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # 如果不从检查点恢复，则从opt.evolve_population中的.yaml文件生成初始值
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # 为种群的其余部分生成搜索空间内的随机值
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # 运行遗传算法，进行固定数量的世代
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # 自适应精英大小
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # 评估种群中每个个体的适应度
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks) # 训练模型并获取结果
                callbacks = Callbacks() # 重置回调
                # 写入变异结果
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2]) # 记录mAP@.5作为适应度分数

            # 使用自适应锦标赛选择选择最适合的个体进行繁殖
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # 自适应锦标赛大小
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # 执行锦标赛选择以选择最佳个体
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # 将精英个体添加到选定索引
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # 通过交叉和变异创建下一代
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # 自适应交叉率
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # 自适应变异率
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1) # 随机扰动
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1]) # 限制在范围内
                next_generation.append(child)
            # 用新一代替换旧种群
            population = next_generation
        # 打印找到的最佳解决方案
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # 绘制结果
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python train.py --hyp {evolve_yaml}"
        )


def generate_individual(input_ranges, individual_length):
    """
    Generate an individual with random hyperparameters within specified ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene (hyperparameter).
        individual_length (int): The number of genes (hyperparameters) in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Example:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```

    Note:
        The individual returned will have a length equal to `individual_length`, with each gene value being a floating-point
        number within its specified range in `input_ranges`.
    """
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound)) # 在指定范围内生成随机值
    return individual


def run(**kwargs):
    """
    Execute YOLOv5 training with specified options, allowing optional overrides through keyword arguments.

    Args:
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-low.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a
            value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/ hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/train'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        ```python
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    opt = parse_opt(True) # 解析命令行参数
    for k, v in kwargs.items():
        setattr(opt, k, v) # 设置参数
    main(opt) # 调用主函数
    return opt


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 调用主函数
