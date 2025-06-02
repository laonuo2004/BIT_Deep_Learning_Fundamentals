# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode() # 装饰器，禁用梯度计算，减少内存消耗，加速推理
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False, # 是否以原始分辨率绘制掩码
):
    """Run YOLOv5 segmentation inference on diverse sources including images, videos, directories, and streams."""
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # 判断是否为图片或视频文件
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")) # 判断是否为URL
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file) # 判断是否为摄像头或流
    screenshot = source.lower().startswith("screen") # 判断是否为屏幕截图
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) # 选择设备，CPU或CUDA
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 加载模型
    stride, names, pt = model.stride, model.names, model.pt # 获取模型步长、类别名称、是否为PyTorch模型
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸，确保是步长的整数倍

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载视频流
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt) # 加载屏幕截图
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载图片或视频文件
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 模型预热，确保首次推理速度
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device)) # 初始化计数器、窗口列表和时间分析器
    for path, im, im0s, vid_cap, s in dataset: # 遍历数据集
        with dt[0]: # 预处理时间
            im = torch.from_numpy(im).to(model.device) # 将图像从numpy数组转换为torch张量并移动到指定设备
            im = im.half() if model.fp16 else im.float()  # 将图像数据类型转换为FP16或FP32
            im /= 255  # 归一化图像像素值到0-1范围
            if len(im.shape) == 3:
                im = im[None]  # 扩展维度以适应批处理，增加一个batch维度

        # Inference
        with dt[1]: # 推理时间
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False # 如果需要可视化，创建可视化保存路径
            pred, proto = model(im, augment=augment, visualize=visualize)[:2] # 模型前向传播，获取预测结果和原型掩码

        # NMS
        with dt[2]: # NMS时间
            # 对预测结果进行非极大值抑制（NMS），过滤掉低置信度或重叠的边界框
            # pred: 预测结果，包含边界框、置信度、类别和掩码系数
            # conf_thres: 置信度阈值
            # iou_thres: IoU阈值
            # classes: 过滤的类别
            # agnostic_nms: 是否进行类别无关的NMS
            # max_det: 每张图像的最大检测数量
            # nm: 掩码数量
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # 遍历每张图像的预测结果
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # 转换为Path对象
            save_path = str(save_dir / p.name)  # 图像保存路径
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # 文本保存路径
            s += "{:g}x{:g} ".format(*im.shape[2:])  # 打印图像尺寸信息
            imc = im0.copy() if save_crop else im0  # 用于保存裁剪图像的副本
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # 初始化标注器
            if len(det): # 如果存在检测结果
                if retina_masks: # 如果使用RetinaMasks
                    # scale bbox first the crop masks
                    # 将边界框坐标从推理尺寸缩放到原始图像尺寸
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    # 处理掩码，生成原始分辨率的掩码
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    # 处理掩码，生成上采样后的掩码
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    # 将边界框坐标从推理尺寸缩放到原始图像尺寸
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt: # 如果需要保存文本结果
                    # 将掩码转换为多边形分割，并缩放到原始图像尺寸，然后归一化
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))
                    ]

                # Print results
                for c in det[:, 5].unique(): # 遍历每个检测到的类别
                    n = (det[:, 5] == c).sum()  # 计算该类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到打印字符串

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]], # 根据类别获取颜色
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                    / 255
                    if retina_masks
                    else im[i],
                )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])): # 遍历每个检测结果
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # 将分割点展平
                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # 标签格式
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # 整数类别
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}") # 标签文本
                        annotator.box_label(xyxy, label, color=colors(c, True)) # 绘制边界框和标签
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop: # 如果需要保存裁剪图像
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True) # 保存裁剪后的边界框图像

            # Stream results
            im0 = annotator.result() # 获取标注后的图像
            if view_img: # 如果需要显示图像
                if platform.system() == "Linux" and p not in windows: # Linux系统下创建窗口
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 显示图像
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond # 按'q'退出
                    exit()

            # Save results (image with detections)
            if save_img: # 如果需要保存图像
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0) # 保存图像
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # 初始化视频写入器
                    vid_writer[i].write(im0) # 写入视频帧

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms") # 打印推理时间

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image # 计算每张图像的平均速度
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t) # 打印总速度
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}") # 打印结果保存路径
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    parser = argparse.ArgumentParser() # 创建命令行参数解析器
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    opt = parser.parse_args() # 解析命令行参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand # 如果只提供一个尺寸，则将其复制为(h, w)
    print_args(vars(opt)) # 打印所有参数
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop")) # 检查依赖项
    run(**vars(opt)) # 调用run函数执行推理


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 执行主函数
