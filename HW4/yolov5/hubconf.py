# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5.

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Creates or loads a YOLOv5 model, with options for pretrained weights and model customization.

    Args:
        name (str): Model name (e.g., 'yolov5s') or path to the model checkpoint (e.g., 'path/to/best.pt').
        pretrained (bool, optional): If True, loads pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels the model expects. Defaults to 3.
        classes (int, optional): Number of classes the model is expected to detect. Defaults to 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper for various input formats. Defaults to True.
        verbose (bool, optional): If True, prints detailed information during the model creation/loading process. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters (e.g., 'cpu', 'cuda'). If None, selects
            the best available device. Defaults to None.

    Returns:
        (DetectMultiBackend | AutoShape): The loaded YOLOv5 model, potentially wrapped with AutoShape if specified.

    Examples:
        ```python
        import torch
        from ultralytics import _create

        # Load an official YOLOv5s model with pretrained weights
        model = _create('yolov5s')

        # Load a custom model from a local checkpoint
        model = _create('path/to/custom_model.pt', pretrained=False)

        # Load a model with specific input channels and classes
        model = _create('yolov5s', channels=1, classes=10)
        ```

    Notes:
        For more information on model loading and customization, visit the
        [YOLOv5 PyTorch Hub Documentation](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/).
    """
    from pathlib import Path

    # 导入模型构建和工具函数
    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING) # 如果不显示详细信息，则将日志级别设置为警告
    # 检查requirements.txt中的依赖项，排除opencv-python, tensorboard, thop
    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    name = Path(name)
    # 确定模型路径：如果name没有后缀且不是目录，则添加.pt后缀；否则直接使用name
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device) # 选择合适的设备（CPU或CUDA）
        # 如果是预训练模型且通道数为3、类别数为80（默认COCO数据集配置）
        if pretrained and channels == 3 and classes == 80:
            try:
                # 尝试加载检测模型，并根据autoshape参数决定是否融合
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    # 如果是分类模型，发出警告，因为它不支持AutoShape
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                            "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                        )
                    # 如果是分割模型，发出警告，因为它不支持AutoShape
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                            "You will not be able to run inference with this model."
                        )
                    else:
                        # 对于其他模型（如检测模型），应用AutoShape包装器，以便处理多种输入格式和NMS
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                # 如果DetectMultiBackend加载失败，尝试加载任意模型
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            # 如果不是预训练模型或参数不匹配，则从yaml配置创建模型
            # 查找模型对应的yaml配置文件
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # 创建检测模型
            if pretrained:
                # 加载预训练权重
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt["model"].float().state_dict()  # 获取模型状态字典并转换为FP32
                # 交叉字典，排除anchors，以便加载部分预训练权重
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
                model.load_state_dict(csd, strict=False)  # 加载状态字典，strict=False允许不完全匹配
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names  # 设置类别名称属性
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # 重置日志级别为信息
        return model.to(device) # 将模型移动到指定设备并返回

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help."
        raise Exception(s) from e


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    """
    Loads a custom or local YOLOv5 model from a given path with optional autoshaping and device specification.

    Args:
        path (str): Path to the custom model file (e.g., 'path/to/model.pt').
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to model if True, enabling compatibility with various input
            types (default is True).
        _verbose (bool): If True, prints all informational messages to the screen; otherwise, operates silently
            (default is True).
        device (str | torch.device | None): Device to load the model on, e.g., 'cpu', 'cuda', torch.device('cuda:0'), etc.
            (default is None, which automatically selects the best available device).

    Returns:
        torch.nn.Module: A YOLOv5 model loaded with the specified parameters.

    Notes:
        For more details on loading models from PyTorch Hub:
        https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Examples:
        ```python
        # Load model from a given path with autoshape enabled on the best available device
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')

        # Load model from a local path without autoshape on the CPU device
        model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local', autoshape=False, device='cpu')
        ```
    """
    # 调用_create函数加载自定义模型
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-nano model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of classes for object detection. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model for various formats (file/URI/PIL/
            cv2/np) and non-maximum suppression (NMS) during inference. Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Specifies the device to use for model computation. If None, uses the best device
            available (i.e., GPU if available, otherwise CPU). Defaults to None.

    Returns:
        DetectionModel | ClassificationModel | SegmentationModel: The instantiated YOLOv5-nano model, potentially with
            pretrained weights and autoshaping applied.

    Notes:
        For further details on loading models from PyTorch Hub, refer to [PyTorch Hub models](https://pytorch.org/hub/
        ultralytics_yolov5).

    Examples:
        ```python
        import torch
        from ultralytics import yolov5n

        # Load the YOLOv5-nano model with defaults
        model = yolov5n()

        # Load the YOLOv5-nano model with a specific device
        model = yolov5n(device='cuda')
        ```
    """
    # 调用_create函数加载yolov5n模型
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Create a YOLOv5-small (yolov5s) model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device configuration.

    Args:
        pretrained (bool, optional): Flag to load pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): Whether to wrap the model with YOLOv5's .autoshape() for handling various input formats.
            Defaults to True.
        _verbose (bool, optional): Flag to print detailed information regarding model loading. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model computation, can be 'cpu', 'cuda', or
            torch.device instances. If None, automatically selects the best available device. Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-small model configured and loaded according to the specified parameters.

    Example:
        ```python
        import torch

        # Load the official YOLOv5-small model with pretrained weights
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Load the YOLOv5-small model from a specific branch
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')

        # Load a custom YOLOv5-small model from a local checkpoint
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')

        # Load a local YOLOv5-small model specifying source as local repository
        model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')
        ```

    Notes:
        For more details on model loading and customization, visit
        the [YOLOv5 PyTorch Hub Documentation](https://pytorch.org/hub/ultralytics_yolov5/).
    """
    # 调用_create函数加载yolov5s模型
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-medium model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): Apply YOLOv5 .autoshape() wrapper to the model for handling various input formats.
            Default is True.
        _verbose (bool, optional): Whether to print detailed information to the screen. Default is True.
        device (str | torch.device | None, optional): Device specification to use for model parameters (e.g., 'cpu', 'cuda').
            Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5-medium model.

    Usage Example:
        ```python
        import torch

        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load YOLOv5-medium from Ultralytics repository
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5m')  # Load from the master branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m.pt')  # Load a custom/local YOLOv5-medium model
        model = torch.hub.load('.', 'custom', 'yolov5m.pt', source='local')  # Load from a local repository
        ```

    For more information, visit https://pytorch.org/hub/ultralytics_yolov5.
    """
    # 调用_create函数加载yolov5m模型
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-large model with options for pretraining, channels, classes, autoshaping, verbosity, and device
    selection.

    Args:
        pretrained (bool): Load pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to model. Default is True.
        _verbose (bool): Print all information to screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters, e.g., 'cpu', 'cuda', or a torch.device instance.
            Default is None.

    Returns:
        YOLOv5 model (torch.nn.Module): The YOLOv5-large model instantiated with specified configurations and possibly
        pretrained weights.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        ```

    Notes:
        For additional details, refer to the PyTorch Hub models documentation:
        https://pytorch.org/hub/ultralytics_yolov5
    """
    # 调用_create函数加载yolov5l模型
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Perform object detection using the YOLOv5-xlarge model with options for pretraining, input channels, class count,
    autoshaping, verbosity, and device specification.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of model classes for object detection. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper for handling different input formats. Defaults to
            True.
        _verbose (bool): If True, prints detailed information during model loading. Defaults to True.
        device (str | torch.device | None): Device specification for computing the model, e.g., 'cpu', 'cuda:0', torch.device('cuda').
            Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-xlarge model loaded with the specified parameters, optionally with pretrained weights and
        autoshaping applied.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
        ```

    For additional details, refer to the official YOLOv5 PyTorch Hub models documentation:
    https://docs.ultralytics.com/yolov5
    """
    # 调用_create函数加载yolov5x模型
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-nano-P6 model with options for pretraining, channels, classes, autoshaping, verbosity, and device.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper to the model. Default is True.
        _verbose (bool, optional): If True, prints all information to screen. Default is True.
        device (str | torch.device | None, optional): Device to use for model parameters. Can be 'cpu', 'cuda', or None.
            Default is None.

    Returns:
        torch.nn.Module: YOLOv5-nano-P6 model loaded with the specified configurations.

    Example:
        ```python
        import torch
        model = yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device='cuda')
        ```

    Notes:
        For more information on PyTorch Hub models, visit: https://pytorch.org/hub/ultralytics_yolov5
    """
    # 调用_create函数加载yolov5n6模型
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiate the YOLOv5-small-P6 model with options for pretraining, input channels, number of classes, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of object detection classes. Default is 80.
        autoshape (bool): If True, applies YOLOv5 .autoshape() wrapper to the model, allowing for varied input formats.
            Default is True.
        _verbose (bool): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None): Device specification for model parameters (e.g., 'cpu', 'cuda', or torch.device).
            Default is None, which selects an available device automatically.

    Returns:
        torch.nn.Module: The YOLOv5-small-P6 model instance.

    Usage:
        ```python
        import torch

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s6')  # load from a specific branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/yolov5s6.pt')  # custom/local model
        model = torch.hub.load('.', 'custom', 'path/to/yolov5s6.pt', source='local')  # local repo model
        ```

    Notes:
        - For more information, refer to the PyTorch Hub models documentation at https://pytorch.org/hub/ultralytics_yolov5

    Raises:
        Exception: If there is an error during model creation or loading, with a suggestion to visit the YOLOv5
            tutorials for help.
    """
    # 调用_create函数加载yolov5s6模型
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Create YOLOv5-medium-P6 model with options for pretraining, channel count, class count, autoshaping, verbosity, and
    device.

    Args:
        pretrained (bool): If True, loads pretrained weights. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to the model for file/URI/PIL/cv2/np inputs and NMS.
            Default is True.
        _verbose (bool): If True, prints detailed information to the screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters. Default is None, which uses the
            best available device.

    Returns:
        torch.nn.Module: The YOLOv5-medium-P6 model.

    Refer to the PyTorch Hub models documentation: https://pytorch.org/hub/ultralytics_yolov5 for additional details.

    Example:
        ```python
        import torch

        # Load YOLOv5-medium-P6 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
        ```

    Notes:
        - The model can be loaded with pre-trained weights for better performance on specific tasks.
        - The autoshape feature simplifies input handling by allowing various popular data formats.
    """
    # 调用_create函数加载yolov5m6模型
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiate the YOLOv5-large-P6 model with options for pretraining, channel and class counts, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool, optional): If True, load pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, apply YOLOv5 .autoshape() wrapper to the model for input flexibility. Default is True.
        _verbose (bool, optional): If True, print all information to the screen. Default is True.
        device (str | torch.device | None, optional): Device to use for model parameters, e.g., 'cpu', 'cuda', or torch.device.
            If None, automatically selects the best available device. Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5-large-P6 model.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')  # official model
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5l6')  # from specific branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/yolov5l6.pt')  # custom/local model
        model = torch.hub.load('.', 'custom', 'path/to/yolov5l6.pt', source='local')  # local repository
        ```

    Note:
        Refer to [PyTorch Hub Documentation](https://pytorch.org/hub/ultralytics_yolov5/) for additional usage instructions.
    """
    # 调用_create函数加载yolov5l6模型
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates the YOLOv5-xlarge-P6 model with options for pretraining, number of input channels, class count, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): If True, applies YOLOv5 .autoshape() wrapper to the model. Default is True.
        _verbose (bool): If True, prints all information to the screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters, can be a string, torch.device object, or
            None for default device selection. Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5-xlarge-P6 model.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # load the YOLOv5-xlarge-P6 model
        ```

    Note:
        For more information on YOLOv5 models, visit the official documentation:
        https://docs.ultralytics.com/yolov5
    """
    # 调用_create函数加载yolov5x6模型
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s", help="model name")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    # 通过命令行参数选择模型，并创建模型实例
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    # 定义多种输入图像格式，包括文件名、Path对象、URI、OpenCV图像、PIL图像和Numpy数组
    imgs = [
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV (BGR to RGB)
        Image.open("data/images/bus.jpg"),  # PIL
        np.zeros((320, 640, 3)),
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # 对图像进行批量推理，指定输入尺寸

    # Results
    results.print() # 打印推理结果
    results.save() # 保存推理结果