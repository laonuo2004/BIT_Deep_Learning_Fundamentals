# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
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
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
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
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # 模型路径或triton URL
    source=ROOT / "data/images",  # 文件/目录/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml路径
    imgsz=(640, 640),  # 推理尺寸 (高, 宽)
    conf_thres=0.25,  # 置信度阈值
    iou_thres=0.45,  # NMS IoU阈值
    max_det=1000,  # 每张图像最大检测数量
    device="",  # cuda设备，例如 0 或 0,1,2,3 或 cpu
    view_img=False,  # 显示结果
    save_txt=False,  # 将结果保存到*.txt
    save_format=0,  # 保存框坐标的格式，YOLO格式或Pascal-VOC格式 (0为YOLO，1为Pascal-VOC)
    save_csv=False,  # 将结果保存为CSV格式
    save_conf=False,  # 在--save-txt标签中保存置信度
    save_crop=False,  # 保存裁剪的预测框
    nosave=False,  # 不保存图像/视频
    classes=None,  # 按类别过滤: --class 0, 或 --class 0 2 3
    agnostic_nms=False,  # 类别无关NMS
    augment=False,  # 增强推理
    visualize=False,  # 可视化特征
    update=False,  # 更新所有模型
    project=ROOT / "runs/detect",  # 将结果保存到project/name
    name="exp",  # 将结果保存到project/name
    exist_ok=False,  # 现有project/name可用，不递增
    line_thickness=3,  # 边界框线粗 (像素)
    hide_labels=False,  # 隐藏标签
    hide_conf=False,  # 隐藏置信度
    half=False,  # 使用FP16半精度推理
    dnn=False,  # 对ONNX推理使用OpenCV DNN
    vid_stride=1,  # 视频帧率步长
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # 是否保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # 判断是否为文件
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")) # 判断是否为URL
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file) # 判断是否为摄像头或流
    screenshot = source.lower().startswith("screen") # 判断是否为屏幕截图
    if is_url and is_file:
        source = check_file(source)  # 下载文件

    # 目录设置
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行目录
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 加载模型
    device = select_device(device) # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 加载多后端检测模型
    stride, names, pt = model.stride, model.names, model.pt # 获取模型步长、类别名称、是否为PyTorch模型
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    # 数据加载器
    bs = 1  # 批次大小
    if webcam:
        view_img = check_imshow(warn=True) # 检查是否可以显示图像
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载视频流
        bs = len(dataset) # 批次大小为流的数量
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt) # 加载屏幕截图
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # 加载图像
    vid_path, vid_writer = [None] * bs, [None] * bs # 视频路径和写入器

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 模型预热
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device)) # 已处理图像数量，窗口列表，计时器
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]: # 预处理时间
            im = torch.from_numpy(im).to(model.device) # 将图像转换为Tensor并移动到设备
            im = im.half() if model.fp16 else im.float()  # uint8转fp16/32
            im /= 255  # 0 - 255转0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 扩展批次维度
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0) # 如果是XML模型且批次大小大于1，则分块

        # 推理
        with dt[1]: # 推理时间
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False # 可视化路径
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize) # 模型前向传播
        # NMS (非极大值抑制)
        with dt[2]: # NMS时间
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # 执行NMS

        # 第二阶段分类器 (可选)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 定义CSV文件路径
        csv_path = save_dir / "predictions.csv"

        # 创建或追加到CSV文件
        def write_to_csv(image_name, prediction, confidence):
            """将图像的预测数据写入CSV文件，如果文件存在则追加。"""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader() # 写入CSV头部
                writer.writerow(data) # 写入数据行

        # 处理预测结果
        for i, det in enumerate(pred):  # 每张图像
            seen += 1 # 已处理图像数量加1
            if webcam:  # 批次大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count # 获取路径、原始图像、帧数
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0) # 获取路径、原始图像、帧数

            p = Path(p)  # 转换为Path对象
            save_path = str(save_dir / p.name)  # 保存图像路径
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # 保存txt路径
            s += "{:g}x{:g} ".format(*im.shape[2:])  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
            imc = im0.copy() if save_crop else im0  # 用于保存裁剪图像
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # 初始化标注器
            if len(det): # 如果有检测结果
                # 将框从img_size缩放到im0尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每类检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 整数类别
                    label = names[c] if hide_conf else f"{names[c]}" # 标签
                    confidence = float(conf) # 置信度
                    confidence_str = f"{confidence:.2f}" # 置信度字符串

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str) # 写入CSV

                    if save_txt:  # 写入文件
                        if save_format == 0: # YOLO格式
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # 归一化xywh
                        else: # Pascal-VOC格式
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # 标签格式
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # 在图像上添加边界框
                        c = int(cls)  # 整数类别
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}") # 标签文本
                        annotator.box_label(xyxy, label, color=colors(c, True)) # 绘制边界框和标签
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True) # 保存裁剪框

            # 流式结果
            im0 = annotator.result() # 获取标注后的图像
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小 (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 显示图像
                cv2.waitKey(1)  # 1毫秒等待

            # 保存结果 (带检测的图像)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0) # 保存图像
                else:  # 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 强制结果视频后缀为*.mp4
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # 创建视频写入器
                    vid_writer[i].write(im0) # 写入帧

        # 打印时间 (仅推理)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)  # 每张图像的速度
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型 (修复SourceChangeWarning)


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    # 定义命令行参数
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展图像尺寸
    print_args(vars(opt)) # 打印参数
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop")) # 检查依赖
    run(**vars(opt)) # 调用run函数


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 调用主函数
