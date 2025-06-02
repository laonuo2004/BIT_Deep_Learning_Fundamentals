# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 segment model on a segment dataset.

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_native, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """Saves detection results in txt format; includes class, xywh (normalized), optionally confidence if `save_conf` is
    True.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh # 归一化增益
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh # 归一化xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format # 标签格式
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n") # 写入文件


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """
    Saves a JSON file with detection results including bounding boxes, category IDs, scores, and segmentation masks.

    Example JSON result: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.
    """
    from pycocotools.mask import encode

    def single_encode(x):
        """Encodes binary mask arrays into RLE (Run-Length Encoding) format for JSON serialization."""
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0] # 将二进制掩码数组编码为RLE格式
        rle["counts"] = rle["counts"].decode("utf-8") # 解码counts
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem # 图像ID
    box = xyxy2xywh(predn[:, :4])  # xywh # 边界框转换为xywh格式
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner # 中心点转换为左上角
    pred_masks = np.transpose(pred_masks, (2, 0, 1)) # 调整预测掩码维度
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks) # 多线程编码掩码
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        ) # 添加到JSON字典


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels.
    """
    if masks:
        if overlap:
            nl = len(labels) # 标签数量
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640) # 重复GT掩码
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0) # 根据索引设置GT掩码
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0] # 插值GT掩码
            gt_masks = gt_masks.gt_(0.5) # 二值化GT掩码
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1)) # 计算掩码IoU
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4]) # 计算边界框IoU

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool) # 初始化正确预测矩阵
    correct_class = labels[:, 0:1] == detections[:, 5] # 类别是否匹配
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match # IoU和类别都匹配
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou] # 匹配结果
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # 按IoU降序排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # 移除重复的检测
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # 移除重复的标签
            correct[matches[:, 1].astype(int), i] = True # 标记正确预测
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode() # 禁用梯度计算，加速推理
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
):
    """Validates a YOLOv5 segmentation model on specified dataset, producing metrics, plots, and optional JSON
    output.
    """
    if save_json:
        check_requirements("pycocotools>=2.0.6") # 检查pycocotools依赖
        process = process_mask_native  # more accurate # 使用更精确的掩码处理方法
    else:
        process = process_mask  # faster # 使用更快的掩码处理方法

    # Initialize/load model and set device
    training = model is not None # 判断是否在训练模式下调用
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model # 获取模型设备
        half &= device.type != "cpu"  # half precision only supported on CUDA # 半精度只支持CUDA
        model.half() if half else model.float() # 设置模型精度
        nm = de_parallel(model).model[-1].nm  # number of masks # 掩码数量
    else:  # called directly
        device = select_device(device, batch_size=batch_size) # 选择设备

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run # 递增运行目录
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir # 创建保存目录

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 加载模型
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine # 获取模型步长、是否为PyTorch模型、是否为JIT、是否为引擎
        imgsz = check_img_size(imgsz, s=stride)  # check image size # 检查图像尺寸
        half = model.fp16  # FP16 supported on limited backends with CUDA # 半精度支持
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks # 掩码数量
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models") # 非PyTorch模型强制批处理大小为1

        # Data
        data = check_dataset(data)  # check # 检查数据集

    # Configure
    model.eval() # 设置模型为评估模式
    cuda = device.type != "cpu" # 是否使用CUDA
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset # 是否为COCO数据集
    nc = 1 if single_cls else int(data["nc"])  # number of classes # 类别数量
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95 # IoU阈值向量
    niou = iouv.numel() # IoU阈值数量

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            ) # 检查权重和数据集的类别数量是否匹配
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup # 模型预热
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks # 填充和矩形推理
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images # 任务类型
        dataloader = create_dataloader( # 创建数据加载器
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio,
        )[0]

    seen = 0 # 已处理图像数量
    confusion_matrix = ConfusionMatrix(nc=nc) # 混淆矩阵
    names = model.names if hasattr(model, "names") else model.module.names  # get class names # 类别名称
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names)) # 转换为字典格式
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000)) # 类别映射
    s = ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP50-95)",
    ) # 打印格式
    dt = Profile(device=device), Profile(device=device), Profile(device=device) # 时间分析器
    metrics = Metrics() # 指标计算器
    loss = torch.zeros(4, device=device) # 损失
    jdict, stats = [], [] # JSON字典和统计信息
    # callbacks.run('on_val_start') # 回调函数：验证开始
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar # 进度条
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # callbacks.run('on_val_batch_start') # 回调函数：验证批次开始
        with dt[0]: # 预处理时间
            if cuda:
                im = im.to(device, non_blocking=True) # 图像移动到设备
                targets = targets.to(device) # 目标移动到设备
                masks = masks.to(device) # 掩码移动到设备
            masks = masks.float() # 掩码转换为浮点型
            im = im.half() if half else im.float()  # uint8 to fp16/32 # 图像精度转换
            im /= 255  # 0 - 255 to 0.0 - 1.0 # 图像归一化
            nb, _, height, width = im.shape  # batch size, channels, height, width # 获取图像尺寸信息

        # Inference
        with dt[1]: # 推理时间
            preds, protos, train_out = model(im) if compute_loss else (*model(im, augment=augment)[:2], None) # 模型推理

        # Loss
        if compute_loss:
            loss += compute_loss((train_out, protos), targets, masks)[1]  # box, obj, cls # 计算损失

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels # 目标坐标转换为像素
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling # 自动标注
        with dt[2]: # NMS时间
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det, nm=nm
            ) # 非极大值抑制

        # Metrics
        plot_masks = []  # masks for plotting # 用于绘图的掩码
        for si, (pred, proto) in enumerate(zip(preds, protos)): # 遍历每个图像的预测结果
            labels = targets[targets[:, 0] == si, 1:] # 标签
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions # 标签数量，预测数量
            path, shape = Path(paths[si]), shapes[si][0] # 路径和形状
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init # 初始化正确掩码
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init # 初始化正确边界框
            seen += 1 # 已处理图像数量加1

            if npr == 0: # 如果没有预测结果
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0])) # 添加统计信息
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0]) # 处理混淆矩阵
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si # 掩码索引
            gt_masks = masks[midx] # GT掩码
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:]) # 处理预测掩码

            # Predictions
            if single_cls:
                pred[:, 5] = 0 # 单类别设置
            predn = pred.clone() # 复制预测结果
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred # 缩放预测边界框

            # Evaluate
            if nl: # 如果有标签
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes # 目标边界框
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels # 缩放标签边界框
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels # 拼接标签
                correct_bboxes = process_batch(predn, labelsn, iouv) # 处理边界框
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True) # 处理掩码
                if plots:
                    confusion_matrix.process_batch(predn, labelsn) # 处理混淆矩阵
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls) # 添加统计信息

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8) # 预测掩码转换为uint8
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15])  # filter top 15 to plot # 过滤前15个掩码用于绘图

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt") # 保存txt结果
            if save_json:
                pred_masks = scale_image(
                    im[si].shape[1:], pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1]
                ) # 缩放掩码图像
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary # 保存json结果
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si]) # 回调函数：验证图像结束

        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0) # 拼接掩码
            plot_images_and_masks(im, targets, masks, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names) # 绘制标签图像和掩码
            plot_images_and_masks(
                im,
                output_to_target(preds, max_det=15),
                plot_masks,
                paths,
                save_dir / f"val_batch{batch_i}_pred.jpg",
                names,
            )  # pred # 绘制预测图像和掩码

        # callbacks.run('on_val_batch_end') # 回调函数：验证批次结束

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy # 将统计信息转换为numpy数组
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names) # 计算AP
        metrics.update(results) # 更新指标
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class # 每类目标数量

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # print format # 打印格式
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results())) # 打印平均结果
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels") # 警告：没有标签

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i))) # 打印每类结果

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image # 每张图像的速度
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t) # 打印速度

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values())) # 绘制混淆矩阵
    # callbacks.run('on_val_end') # 回调函数：验证结束

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results() # 获取平均结果

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights # 权重文件名
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations # 标注文件路径
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions # 预测文件路径
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...") # 评估pycocotools mAP
        with open(pred_json, "w") as f:
            json.dump(jdict, f) # 保存JSON文件

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api # 初始化标注API
            pred = anno.loadRes(pred_json)  # init predictions api # 初始化预测API
            results = []
            for eval in COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm"): # 遍历bbox和segm评估
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate # 图像ID
                eval.evaluate() # 评估
                eval.accumulate() # 累积
                eval.summarize() # 总结
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5) # 更新结果
            map_bbox, map50_bbox, map_mask, map50_mask = results # 获取mAP结果
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}") # 打印错误信息

    # Return results
    model.float()  # for training # 模型转换为浮点型
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}") # 打印结果保存路径
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask # 最终指标
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t # 返回结果


def parse_opt():
    """Parses command line arguments for configuring YOLOv5 options like dataset path, weights, batch size, and
    inference settings.
    """
    parser = argparse.ArgumentParser() # 创建命令行参数解析器
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args() # 解析命令行参数
    opt.data = check_yaml(opt.data)  # check YAML # 检查YAML文件
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid # 如果保存混合结果，则保存txt
    print_args(vars(opt)) # 打印参数
    return opt


def main(opt):
    """Executes YOLOv5 tasks including training, validation, testing, speed, and study with configurable options."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop")) # 检查依赖项

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results") # 警告：置信度阈值过高可能导致无效结果
        if opt.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone") # 警告：保存混合结果可能导致mAP虚高
        run(**vars(opt)) # 运行验证

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights] # 权重列表
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results # 设置半精度
        if opt.task == "speed":  # speed benchmarks # 速度基准测试
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False # 设置阈值
            for opt.weights in weights:
                run(**vars(opt), plots=False) # 运行验证，不绘图

        elif opt.task == "study":  # speed vs mAP benchmarks # 速度与mAP基准测试
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to # 保存文件名
            x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis # 图像尺寸范围
            for opt.imgsz in x:  # img-size
                LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...") # 打印信息
                r, _, t = run(**vars(opt), plots=False) # 运行验证
                y.append(r + t)  # results and times # 添加结果和时间
            np.savetxt(f, y, fmt="%10.4g")  # save # 保存结果
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"]) # 压缩文件
            plot_val_study(x=x)  # plot # 绘制研究图
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")') # 任务未实现


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 执行主函数
