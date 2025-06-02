# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
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
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
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
    print_args,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(224, 224),  # inference size (height, width)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    nosave=False,  # do not save images/videos
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-cls",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """Conducts YOLOv5 classification inference on diverse input sources and saves results."""
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) # 选择运行设备，如CPU或CUDA
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 加载模型，支持多种后端
    stride, names, pt = model.stride, model.names, model.pt # 获取模型步长、类别名称和是否为PyTorch模型
    imgsz = check_img_size(imgsz, s=stride)  # 检查并调整图像尺寸以适应模型步长

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride) # 从视频流加载数据
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt) # 从屏幕截图加载数据
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride) # 从图片或视频文件加载数据
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 模型预热，优化首次推理速度
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device)) # 初始化计数器、窗口列表和性能分析器
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]: # 预处理时间
            im = torch.Tensor(im).to(model.device) # 将图像转换为Tensor并移动到指定设备
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 # 根据模型精度要求转换为半精度或浮点精度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim # 增加批次维度

        # Inference
        with dt[1]: # 推理时间
            results = model(im) # 执行模型推理

        # Post-process
        with dt[2]: # 后处理时间
            pred = F.softmax(results, dim=1)  # probabilities # 对推理结果进行softmax，得到类别概率

        # Process predictions
        for i, prob in enumerate(pred):  # per image # 遍历每个图像的预测结果
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg # 结果保存路径
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt # 文本结果保存路径

            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string # 记录图像尺寸
            annotator = Annotator(im0, example=str(names), pil=True) # 初始化图像标注器

            # Print results
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices # 获取Top-5预测类别的索引
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, " # 格式化Top-5预测结果字符串

            # Write results
            text = "\n".join(f"{prob[j]:.2f} {names[j]}" for j in top5i) # 格式化文本结果
            if save_img or view_img:  # Add bbox to image
                annotator.text([32, 32], text, txt_color=(255, 255, 255)) # 在图像上添加文本标注
            if save_txt:  # Write to file
                with open(f"{txt_path}.txt", "a") as f:
                    f.write(text + "\n") # 将文本结果写入文件

            # Stream results
            im0 = annotator.result() # 获取标注后的图像
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 显示图像
                cv2.waitKey(1)  # 1 millisecond # 等待按键，用于刷新显示

            # Save results (image with detections)
            if save_img:
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
        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms") # 打印推理时间

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image # 计算每张图像的平均处理速度
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning) # 剥离优化器信息，更新模型


def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[224], help="inference size h,w")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-cls", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand # 如果只指定一个尺寸，则扩展为高宽相同
    print_args(vars(opt)) # 打印解析后的参数
    return opt


def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop")) # 检查并安装必要的依赖
    run(**vars(opt)) # 调用run函数执行推理


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 执行主函数
