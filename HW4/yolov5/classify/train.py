# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv5 classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    check_git_info,
    check_git_status,
    check_requirements,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (
    ModelEMA,
    de_parallel,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_DDP,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(opt, device):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True) # 初始化随机种子，确保结果可复现
    save_dir, data, bs, epochs, nw, imgsz, pretrained = (
        opt.save_dir,
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    cuda = device.type != "cpu" # 判断是否使用CUDA

    # Directories
    wdir = save_dir / "weights" # 权重保存目录
    wdir.mkdir(parents=True, exist_ok=True)  # make dir # 创建目录
    last, best = wdir / "last.pt", wdir / "best.pt" # 最新和最佳模型权重文件路径

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt)) # 保存训练参数到YAML文件

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None # 初始化日志记录器

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT): # 确保在分布式训练中只有一个进程下载数据
        data_dir = data if data.is_dir() else (DATASETS_DIR / data) # 数据集目录
        if not data_dir.is_dir():
            LOGGER.info(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
            t = time.time()
            if str(data) == "imagenet":
                subprocess.run(["bash", str(ROOT / "data/scripts/get_imagenet.sh")], shell=True, check=True) # 下载ImageNet数据集
            else:
                url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{data}.zip"
                download(url, dir=data_dir.parent) # 下载其他数据集
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes # 获取类别数量
    trainloader = create_classification_dataloader(
        path=data_dir / "train",
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
    ) # 创建训练数据加载器

    test_dir = data_dir / "test" if (data_dir / "test").exists() else data_dir / "val"  # data/test or data/val # 测试/验证集目录
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(
            path=test_dir,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE * 2,
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=nw,
        ) # 创建测试/验证数据加载器

    # Model
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith(".pt"):
            model = attempt_load(opt.model, device="cpu", fuse=False) # 加载预训练模型或自定义模型
        elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            model = torchvision.models.__dict__[opt.model](weights="IMAGENET1K_V1" if pretrained else None) # 加载TorchVision模型
        else:
            m = hub.list("ultralytics/yolov5")  # + hub.list('pytorch/vision')  # models
            raise ModuleNotFoundError(f"--model {opt.model} not found. Available models are: \n" + "\n".join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'")
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # convert to classification model # 将检测模型转换为分类模型
        reshape_classifier_output(model, nc)  # update class count # 调整分类器输出层以匹配类别数量
    for m in model.modules():
        if not pretrained and hasattr(m, "reset_parameters"):
            m.reset_parameters() # 如果不是预训练模型，则重置参数
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout # 设置Dropout比率
    for p in model.parameters():
        p.requires_grad = True  # for training # 启用参数梯度计算

    model = model.to(device) # 将模型移动到指定设备

    # Info
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes  # attach class names # 绑定类别名称
        model.transforms = testloader.dataset.torch_transforms  # attach inference transforms # 绑定推理变换
        model_info(model) # 打印模型信息
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / "train_images.jpg") # 显示训练图像示例
        logger.log_images(file, name="Train Examples") # 记录训练图像示例
        logger.log_graph(model, imgsz)  # log model # 记录模型计算图

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay) # 初始化优化器

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0) # 最终学习率比例

    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    def lf(x):
        """Linear learning rate scheduler function, scaling learning rate from initial value to `lrf` over `epochs`."""
        return (1 - x / epochs) * (1 - lrf) + lrf  # linear # 线性学习率衰减函数

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # 初始化学习率调度器
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None # 初始化模型EMA（指数移动平均）

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model) # 启用分布式数据并行训练

    # Train
    t0 = time.time() # 记录训练开始时间
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function # 初始化损失函数
    best_fitness = 0.0 # 最佳性能指标
    scaler = amp.GradScaler(enabled=cuda) # 初始化混合精度训练的梯度缩放器
    val = test_dir.stem  # 'val' or 'test'
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} test\n"
        f"Using {nw * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n"
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"
    ) # 打印训练信息
    for epoch in range(epochs):  # loop over the dataset multiple times # 遍历每个训练周期
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness # 初始化损失和性能指标
        model.train() # 设置模型为训练模式
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch) # 在分布式训练中设置采样器epoch
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT) # 初始化进度条
        for i, (images, labels) in pbar:  # progress bar # 遍历训练数据批次
            images, labels = images.to(device, non_blocking=True), labels.to(device) # 将数据移动到指定设备

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled # 启用混合精度自动转换
                loss = criterion(model(images), labels) # 计算损失

            # Backward
            scaler.scale(loss).backward() # 损失反向传播并进行梯度缩放

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients # 取消梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients # 梯度裁剪
            scaler.step(optimizer) # 更新模型参数
            scaler.update() # 更新梯度缩放器
            optimizer.zero_grad() # 清零梯度
            if ema:
                ema.update(model) # 更新EMA模型

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses # 更新平均训练损失
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB) # 获取GPU内存使用情况
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36 # 更新进度条描述

                # Test
                if i == len(pbar) - 1:  # last batch # 在每个epoch的最后一个批次进行验证
                    top1, top5, vloss = validate.run(
                        model=ema.ema, dataloader=testloader, criterion=criterion, pbar=pbar
                    )  # test accuracy, loss # 运行验证并获取Top-1、Top-5准确率和验证损失
                    fitness = top1  # define fitness as top1 accuracy # 将Top-1准确率作为性能指标

        # Scheduler
        scheduler.step() # 更新学习率

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness # 更新最佳性能指标

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate # 记录各项指标
            logger.log_metrics(metrics, epoch) # 记录日志

            # Save model
            final_epoch = epoch + 1 == epochs # 判断是否为最后一个epoch
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(), # 保存模型权重
                    "ema": None,  # deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": None,  # optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last) # 保存最新模型
                if best_fitness == fitness:
                    torch.save(ckpt, best) # 如果是最佳模型则保存
                del ckpt # 删除检查点，释放内存

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(
            f"\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)"
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f"\nPredict:         python classify/predict.py --weights {best} --source im.jpg"
            f"\nValidate:        python classify/val.py --weights {best} --data {data_dir}"
            f"\nExport:          python export.py --weights {best} --include onnx"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
            f"\nVisualize:       https://netron.app\n"
        ) # 打印训练完成信息和后续操作提示

        # Plot examples
        images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels # 获取测试集前25张图像和标签
        pred = torch.max(ema.ema(images.to(device)), 1)[1] # 获取模型预测结果
        file = imshow_cls(images, labels, pred, de_parallel(model).names, verbose=False, f=save_dir / "test_images.jpg") # 显示测试图像示例

        # Log results
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_images(file, name="Test Examples (true-predicted)", epoch=epoch) # 记录测试图像示例
        logger.log_model(best, epochs, metadata=meta) # 记录最佳模型


def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s-cls.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="imagenette160", help="cifar10, cifar100, mnist, imagenet, ...")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    return parser.parse_known_args()[0] if known else parser.parse_args() # 解析命令行参数


def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt)) # 打印参数
        check_git_status() # 检查Git状态
        check_requirements(ROOT / "requirements.txt") # 检查依赖

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size) # 选择设备
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, "AutoBatch is coming soon for classification, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK) # 设置当前进程的CUDA设备
        device = torch.device("cuda", LOCAL_RANK) # 创建设备对象
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo") # 初始化分布式进程组

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run # 设置保存目录

    # Train
    train(opt, device) # 调用训练函数


def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True) # 解析参数
    for k, v in kwargs.items():
        setattr(opt, k, v) # 设置参数
    main(opt) # 执行主函数
    return opt


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 执行主函数
