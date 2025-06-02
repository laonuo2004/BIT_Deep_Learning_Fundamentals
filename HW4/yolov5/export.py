# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
    yaml_save,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS环境


class iOSModel(torch.nn.Module):
    """一个iOS兼容的YOLOv5模型包装器，根据图像尺寸对输入图像进行归一化。"""

    def __init__(self, model, im):
        """
        使用基于图像尺寸的归一化初始化iOS兼容模型。

        Args:
            model (torch.nn.Module): 要适配iOS兼容性的PyTorch模型。
            im (torch.Tensor): 表示批处理图像的输入张量，形状为 (B, C, H, W)。

        Returns:
            None: 此方法不返回任何值。

        Notes:
            此初始化器根据输入图像尺寸配置归一化，这对于确保模型在iOS设备上的兼容性和正常功能至关重要。
            归一化步骤涉及如果图像是正方形则除以图像宽度；否则，可能适用其他条件。
        """
        super().__init__()
        b, c, h, w = im.shape  # 批次，通道，高度，宽度
        self.model = model
        self.nc = model.nc  # 类别数量
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # 广播 (较慢，较小)
            # np = model(im)[0].shape[1]  # 点的数量
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # 显式 (较快，较大)

    def forward(self, x):
        """
        对输入张量进行前向传播，返回类别置信度和归一化坐标。

        Args:
            x (torch.Tensor): 包含图像数据的输入张量，形状为 (批次, 通道, 高度, 宽度)。

        Returns:
            torch.Tensor: 包含归一化坐标 (xywh)、置信度分数 (conf) 和类别概率 (cls) 的拼接张量，
            形状为 (N, 4 + 1 + C)，其中 N 是预测数量，C 是类别数量。

        Examples:
            ```python
            model = iOSModel(pretrained_model, input_image)
            output = model.forward(torch_input_tensor)
            ```
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # 置信度 (3780, 80), 坐标 (3780, 4)


def export_formats():
    r"""
    返回支持的YOLOv5模型导出格式及其属性的DataFrame。

    Returns:
        pandas.DataFrame: 包含支持的导出格式及其属性的DataFrame。DataFrame包括格式名称、CLI参数后缀、
        文件扩展名或目录名称以及指示导出格式是否支持训练和检测的布尔标志。

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        DataFrame包含以下列：
        - Format: 模型格式的名称 (例如，PyTorch, TorchScript, ONNX等)。
        - Include Argument: 用于导出脚本以包含此格式的参数。
        - File Suffix: 与格式关联的文件扩展名或目录名称。
        - Supports Training: 格式是否支持训练。
        - Supports Detection: 格式是否支持检测。
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    """
    记录成功或失败、执行时间以及文件大小，用于使用@try_export包装的YOLOv5模型导出函数。

    Args:
        inner_func (Callable): 要由装饰器包装的模型导出函数。

    Returns:
        Callable: 记录执行细节的包装函数。执行时，此包装函数返回：
            - 元组 (str | torch.nn.Module): 成功时 — 导出模型的文件路径和模型实例。
            - 元组 (None, None): 失败时 — None值表示导出失败。

    Examples:
        ```python
        @try_export
        def export_onnx(model, filepath):
            # implementation here
            pass

        exported_file, exported_model = export_onnx(yolo_model, 'path/to/save/model.onnx')
        ```

    Notes:
        有关其他要求和模型导出格式，请参阅
        [Ultralytics YOLOv5 GitHub仓库](https://github.com/ultralytics/ultralytics)。
    """
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """记录使用@try_export装饰器包装的模型导出函数的成功/失败和执行细节。"""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):
    """
    将YOLOv5模型导出为TorchScript格式。

    Args:
        model (torch.nn.Module): 要导出的YOLOv5模型。
        im (torch.Tensor): 用于跟踪TorchScript模型的示例输入张量。
        file (Path): 导出TorchScript模型的保存路径。
        optimize (bool): 如果为True，则应用移动部署优化。
        prefix (str): 日志消息的可选前缀。默认为'TorchScript:'。

    Returns:
        (str | None, torch.jit.ScriptModule | None): 一个元组，包含导出模型的文件路径 (字符串)
            和TorchScript模型 (torch.jit.ScriptModule)。如果导出失败，元组的两个元素都将为None。

    Notes:
        - 此函数使用跟踪来创建TorchScript模型。
        - 元数据，包括输入图像形状、模型步长和类别名称，保存在TorchScript模型包内的额外文件 (`config.txt`) 中。
        - 有关移动优化的信息，请参阅PyTorch教程: https://pytorch.org/tutorials/recipes/mobile_interpreter.html

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        # Load model
        weights = 'yolov5s.pt'
        device = select_device('')
        model = attempt_load(weights, device=device)

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640).to(device)

        # Export model
        file = Path('yolov5s.torchscript')
        export_torchscript(model, im, file, optimize=False)
        ```
    """
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")

    ts = torch.jit.trace(model, im, strict=False) # 跟踪模型以创建TorchScript
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files) # 优化并保存用于移动端
    else:
        ts.save(str(f), _extra_files=extra_files) # 保存TorchScript模型
    return f, None


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    """
    将YOLOv5模型导出为ONNX格式，支持动态轴和可选的模型简化。

    Args:
        model (torch.nn.Module): 要导出的YOLOv5模型。
        im (torch.Tensor): 用于模型跟踪的示例输入张量，通常形状为 (1, 3, 高度, 宽度)。
        file (pathlib.Path | str): ONNX模型的输出文件路径。
        opset (int): 用于导出的ONNX opset版本。
        dynamic (bool): 如果为True，则为批次、高度和宽度维度启用动态轴。
        simplify (bool): 如果为True，则应用ONNX模型简化以进行优化。
        prefix (str): 日志消息的前缀字符串，默认为'ONNX:'。

    Returns:
        tuple[pathlib.Path | str, None]: 保存的ONNX模型文件路径和None (与装饰器一致)。

    Raises:
        ImportError: 如果未安装导出所需的库 (例如，'onnx', 'onnx-simplifier')。
        AssertionError: 如果简化检查失败。

    Notes:
        此函数所需的包可以通过以下方式安装：
        ```
        pip install onnx onnx-simplifier onnxruntime onnxruntime-gpu
        ```

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        # Load model
        weights = 'yolov5s.pt'
        device = select_device('')
        model = attempt_load(weights, map_location=device)

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640).to(device)

        # Export model
        file_path = Path('yolov5s.onnx')
        export_onnx(model, im, file_path, opset=12, dynamic=True, simplify=True)
        ```
    """
    check_requirements("onnx>=1.12.0")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))

    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"] # 根据模型类型设置输出名称
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # 形状(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # 形状(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # 形状(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # 形状(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic仅与cpu兼容
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: torch>=1.12的DNN推理可能需要do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    # 检查
    model_onnx = onnx.load(f)  # 加载onnx模型
    onnx.checker.check_model(model_onnx)  # 检查onnx模型

    # 元数据
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # 简化
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnxslim"))
            import onnxslim

            LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")
    return f, model_onnx


@try_export
def export_openvino(file, metadata, half, int8, data, prefix=colorstr("OpenVINO:")):
    """
    将YOLOv5模型导出为OpenVINO格式，可选FP16和INT8量化。

    Args:
        file (Path): OpenVINO模型的输出文件路径。
        metadata (dict): 包含模型元数据 (如名称和步长) 的字典。
        half (bool): 如果为True，则以FP16精度导出模型。
        int8 (bool): 如果为True，则以INT8量化导出模型。
        data (str): INT8量化所需的数据集YAML文件路径。
        prefix (str): 用于日志记录的前缀字符串 (默认为"OpenVINO:")。

    Returns:
        (str, openvino.runtime.Model | None): OpenVINO模型文件路径和openvino.runtime.Model对象 (如果导出成功)；
        否则为None。

    Notes:
        - 需要`openvino-dev`包版本2023.0或更高。安装命令:
          `$ pip install openvino-dev>=2023.0`
        - 对于INT8量化，还需要`nncf`库版本2.5.0或更高。安装命令:
          `$ pip install nncf>=2.5.0`

    Examples:
        ```python
        from pathlib import Path
        from ultralytics import YOLOv5

        model = YOLOv5('yolov5s.pt')
        export_openvino(Path('yolov5s.onnx'), metadata={'names': model.names, 'stride': model.stride}, half=True,
                        int8=False, data='data.yaml')
        ```

        这将YOLOv5模型导出为OpenVINO，使用FP16精度但没有INT8量化，并保存到指定文件路径。
    """
    check_requirements("openvino-dev>=2023.0")  # 需要openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    f = str(file).replace(file.suffix, f"_{'int8_' if int8 else ''}openvino_model{os.sep}")
    f_onnx = file.with_suffix(".onnx")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)

    ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework="onnx", compress_to_fp16=half)  # 导出

    if int8:
        check_requirements("nncf>=2.5.0")  # 需要至少2.5.0版本才能使用后训练量化
        import nncf
        import numpy as np

        from utils.dataloaders import create_dataloader

        def gen_dataloader(yaml_path, task="train", imgsz=640, workers=4):
            """根据给定的YAML数据集配置生成用于模型训练或验证的DataLoader。"""
            data_yaml = check_yaml(yaml_path)
            data = check_dataset(data_yaml)
            dataloader = create_dataloader(
                data[task], imgsz=imgsz, batch_size=1, stride=32, pad=0.5, single_cls=False, rect=False, workers=workers
            )[0]
            return dataloader

        # noqa: F811

        def transform_fn(data_item):
            """
            量化转换函数。

            从dataloader项中提取和预处理输入数据以进行量化。

            Args:
               data_item: DataLoader在迭代期间生成的数据项元组

            Returns:
                input_tensor: 用于量化的输入数据
            """
            assert data_item[0].dtype == torch.uint8, "input image must be uint8 for the quantization preprocessing"

            img = data_item[0].numpy().astype(np.float32)  # uint8转fp16/32
            img /= 255.0  # 0 - 255转0.0 - 1.0
            return np.expand_dims(img, 0) if img.ndim == 3 else img

        ds = gen_dataloader(data)
        quantization_dataset = nncf.Dataset(ds, transform_fn)
        ov_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

    ov.serialize(ov_model, f_ov)  # 保存
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # 添加metadata.yaml
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr("PaddlePaddle:")):
    """
    使用X2Paddle将YOLOv5 PyTorch模型导出为PaddlePaddle格式，保存转换后的模型和元数据。

    Args:
        model (torch.nn.Module): 要导出的YOLOv5模型。
        im (torch.Tensor): 用于模型跟踪的输入张量。
        file (pathlib.Path): 要转换的源文件路径。
        metadata (dict): 要与模型一起保存的额外元数据。
        prefix (str): 日志信息的前缀。

    Returns:
        tuple (str, None): 一个元组，第一个元素是保存的PaddlePaddle模型的路径，第二个元素是None。

    Examples:
        ```python
        from pathlib import Path
        import torch

        # Assume 'model' is a pre-trained YOLOv5 model and 'im' is an example input tensor
        model = ...  # Load your model here
        im = torch.randn((1, 3, 640, 640))  # Dummy input tensor for tracing
        file = Path("yolov5s.pt")
        metadata = {"stride": 32, "names": ["person", "bicycle", "car", "motorbike"]}

        export_paddle(model=model, im=im, file=file, metadata=metadata)
        ```

    Notes:
        确保已安装`paddlepaddle`和`x2paddle`，因为这些是导出函数所需的。可以通过pip安装：
        ```
        $ pip install paddlepaddle x2paddle
        ```
    """
    check_requirements(("paddlepaddle>=3.0.0", "x2paddle"))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    f = str(file).replace(".pt", f"_paddle_model{os.sep}")

    pytorch2paddle(module=model, save_dir=f, jit_type="trace", input_examples=[im])  # 导出
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # 添加metadata.yaml
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, nms, mlmodel, prefix=colorstr("CoreML:")):
    """
    将YOLOv5模型导出为CoreML格式，可选NMS、INT8和FP16支持。

    Args:
        model (torch.nn.Module): 要导出的YOLOv5模型。
        im (torch.Tensor): 用于跟踪模型的示例输入张量。
        file (pathlib.Path): 保存CoreML模型的路径对象。
        int8 (bool): 指示是否使用INT8量化的标志 (默认为False)。
        half (bool): 指示是否使用FP16量化的标志 (默认为False)。
        nms (bool): 指示是否包含非极大值抑制的标志 (默认为False)。
        mlmodel (bool): 指示是否导出为旧的*.mlmodel格式的标志 (默认为False)。
        prefix (str): 用于日志记录的前缀字符串 (默认为'CoreML:')。

    Returns:
        tuple[pathlib.Path | None, None]: 保存的CoreML模型文件路径，如果出现错误则为 (None, None)。

    Notes:
        导出的CoreML模型将以.mlmodel扩展名保存。
        量化仅在macOS上支持。

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.yolo import Model
        model = Model(cfg, ch=3, nc=80)
        im = torch.randn(1, 3, 640, 640)
        file = Path("yolov5s_coreml")
        export_coreml(model, im, file, int8=False, half=False, nms=True, mlmodel=False)
        ```
    """
    check_requirements("coremltools")
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    if mlmodel:
        f = file.with_suffix(".mlmodel")
        convert_to = "neuralnetwork"
        precision = None
    else:
        f = file.with_suffix(".mlpackage")
        convert_to = "mlprogram"
        precision = ct.precision.FLOAT16 if half else ct.precision.FLOAT32
    if nms:
        model = iOSModel(model, im) # 如果启用NMS，则使用iOSModel包装器
    ts = torch.jit.trace(model, im, strict=False)  # TorchScript模型
    ct_model = ct.convert(
        ts,
        inputs=[ct.ImageType("image", shape=im.shape, scale=1 / 255, bias=[0, 0, 0])],
        convert_to=convert_to,
        compute_precision=precision,
    )
    bits, mode = (8, "kmeans") if int8 else (16, "linear") if half else (32, None)
    if bits < 32:
        if mlmodel:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning
                )  # 抑制numpy==1.20浮点警告，在coremltools==7.0中已修复
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        elif bits == 8:
            op_config = ct.optimize.coreml.OpPalettizerConfig(mode=mode, nbits=bits, weight_threshold=512)
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            ct_model = ct.optimize.coreml.palettize_weights(ct_model, config)
    ct_model.save(f) # 保存CoreML模型
    return f, ct_model


@try_export
def export_engine(
    model, im, file, half, dynamic, simplify, workspace=4, verbose=False, cache="", prefix=colorstr("TensorRT:")
):
    """
    将YOLOv5模型导出为TensorRT引擎格式，需要GPU和TensorRT>=7.0.0。

    Args:
        model (torch.nn.Module): 要导出的YOLOv5模型。
        im (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        file (pathlib.Path): 保存导出模型的路径。
        half (bool): 设置为True以FP16精度导出。
        dynamic (bool): 设置为True以启用动态输入形状。
        simplify (bool): 设置为True以在导出期间简化模型。
        workspace (int): 工作空间大小 (GB) (默认为4)。
        verbose (bool): 设置为True以启用详细日志输出。
        cache (str): TensorRT计时缓存路径。
        prefix (str): 日志消息前缀。

    Returns:
        (pathlib.Path, None): 包含导出模型路径和None的元组。

    Raises:
        AssertionError: 如果在CPU而不是GPU上执行。
        RuntimeError: 如果解析ONNX文件失败。

    Example:
        ```python
        from ultralytics import YOLOv5
        import torch
        from pathlib import Path

        model = YOLOv5('yolov5s.pt')  # 加载预训练的YOLOv5模型
        input_tensor = torch.randn(1, 3, 640, 640).cuda()  # GPU上的示例输入张量
        export_path = Path('yolov5s.engine')  # 导出目标

        export_engine(model.model, input_tensor, export_path, half=True, dynamic=True, simplify=True, workspace=8, verbose=True)
        ```
    """
    assert im.device.type != "cpu", "导出在CPU上运行，但必须在GPU上，例如 `python export.py --device 0`"
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    if trt.__version__[0] == "7":  # TensorRT 7处理 https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, "8.0.0", hard=True)  # 需要tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix(".onnx")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # 是否为TensorRT >= 10
    assert onnx.exists(), f"未能导出ONNX文件: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT引擎文件
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    else:  # TensorRT版本7, 8
        config.max_workspace_size = workspace * 1 << 30
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"未能加载ONNX文件: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic模型需要最大--batch-size参数")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine, open(f, "wb") as t:
        t.write(engine if is_trt10 else engine.serialize())
    if cache:  # 保存计时缓存
        with open(cache, "wb") as c:
            c.write(config.get_timing_cache().serialize())
    return f, None


@try_export
def export_saved_model(
    model,
    im,
    file,
    dynamic,
    tf_nms=False,
    agnostic_nms=False,
    topk_per_class=100,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    keras=False,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    """
    将YOLOv5模型导出为TensorFlow SavedModel格式，支持动态轴和非极大值抑制 (NMS)。

    Args:
        model (torch.nn.Module): 要转换的PyTorch模型。
        im (torch.Tensor): 用于跟踪的示例输入张量，形状为 (B, C, H, W)。
        file (pathlib.Path): 保存导出模型的路径。
        dynamic (bool): 指示是否应使用动态轴的标志。
        tf_nms (bool, optional): 启用TensorFlow非极大值抑制 (NMS)。默认为False。
        agnostic_nms (bool, optional): 启用类别无关NMS。默认为False。
        topk_per_class (int, optional): 在应用NMS之前每类保留的Top K检测数量。默认为100。
        topk_all (int, optional): 在应用NMS之前所有类别保留的Top K检测数量。默认为100。
        iou_thres (float, optional): NMS的IoU阈值。默认为0.45。
        conf_thres (float, optional): 检测的置信度阈值。默认为0.25。
        keras (bool, optional): 如果为True，则以Keras格式保存模型。默认为False。
        prefix (str, optional): 日志消息的前缀。默认为"TensorFlow SavedModel:"。

    Returns:
        tuple[str, tf.keras.Model | None]: 包含保存的模型文件夹路径和Keras模型实例的元组，
        如果TensorFlow导出失败则为None。

    Notes:
        - 该方法支持TensorFlow版本高达2.15.1。
        - 较旧的TensorFlow版本可能不支持TensorFlow NMS。
        - 如果TensorFlow版本超过2.13.1，导出到TFLite时可能会导致问题。
          参考: https://github.com/ultralytics/yolov5/issues/12489

    Example:
        ```python
        model, im = ...  # 初始化PyTorch模型和输入张量
        export_saved_model(model, im, Path("yolov5_saved_model"), dynamic=True)
        ```
    """
    # YOLOv5 TensorFlow SavedModel导出
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}<=2.15.1")

        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    if tf.__version__ > "2.13.1":
        helper_url = "https://github.com/ultralytics/yolov5/issues/12489"
        LOGGER.info(
            f"WARNING ⚠️ using Tensorflow {tf.__version__} > 2.13.1 might cause issue when exporting the model to tflite {helper_url}"
        )  # 处理问题 https://github.com/ultralytics/yolov5/issues/12489
    f = str(file).replace(".pt", "_saved_model")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # TensorFlow的BHWC顺序
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format="tf") # 保存为Keras格式
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # 完整模型
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(
            tfm,
            f,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
            if check_version(tf.__version__, "2.6")
            else tf.saved_model.SaveOptions(),
        )
    return f, keras_model


@try_export
def export_pb(keras_model, file, prefix=colorstr("TensorFlow GraphDef:")):
    """
    将YOLOv5模型导出为TensorFlow GraphDef (*.pb) 格式。

    Args:
        keras_model (tf.keras.Model): 要转换的Keras模型。
        file (Path): 保存GraphDef的输出文件路径。
        prefix (str): 可选前缀字符串；默认为指示TensorFlow GraphDef导出状态的彩色字符串。

    Returns:
        Tuple[Path, None]: 保存GraphDef模型的文件路径和None占位符。

    Notes:
        有关更多详细信息，请参阅冻结图指南: https://github.com/leimao/Frozen_Graph_TensorFlow

    Example:
        ```python
        from pathlib import Path
        keras_model = ...  # 假设存在一个Keras模型
        file = Path("model.pb")
        export_pb(keras_model, file)
        ```
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = file.with_suffix(".pb")

    m = tf.function(lambda x: keras_model(x))  # 完整模型
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False) # 写入GraphDef
    return f, None


@try_export
def export_tflite(
    keras_model, im, file, int8, per_tensor, data, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")
):
    # YOLOv5 TensorFlow Lite导出
    """
    将YOLOv5模型导出为TensorFlow Lite格式，可选INT8量化和NMS支持。

    Args:
        keras_model (tf.keras.Model): 要导出的Keras模型。
        im (torch.Tensor): 用于归一化和模型跟踪的输入图像张量。
        file (Path): 保存TensorFlow Lite模型的路径。
        int8 (bool): 如果为True，则启用INT8量化。
        per_tensor (bool): 如果为True，则禁用逐通道量化。
        data (str): 用于INT8量化中代表性数据集生成的路径。
        nms (bool): 如果为True，则启用非极大值抑制 (NMS)。
        agnostic_nms (bool): 如果为True，则启用类别无关NMS。
        prefix (str): 日志消息的前缀。

    Returns:
        (str | None, tflite.Model | None): 导出TFLite模型的文件路径和TFLite模型实例，如果导出失败则为None。

    Example:
        ```python
        from pathlib import Path
        import torch
        import tensorflow as tf

        # Load a Keras model wrapping a YOLOv5 model
        keras_model = tf.keras.models.load_model('path/to/keras_model.h5')

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640)

        # Export the model
        export_tflite(keras_model, im, Path('model.tflite'), int8=True, per_tensor=False, data='data/coco.yaml',
                      nms=True, agnostic_nms=False)
        ```

    Notes:
        - 确保已安装TensorFlow和TensorFlow Lite依赖项。
        - INT8量化需要代表性数据集以实现最佳精度。
        - TensorFlow Lite模型适用于在移动和边缘设备上进行高效推理。
    """
    import tensorflow as tf

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace(".pt", "-fp16.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen

        dataset = LoadImages(check_dataset(check_yaml(data))["train"], img_size=imgsz, auto=False)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        if per_tensor:
            converter._experimental_disable_per_channel = True
        f = str(file).replace(".pt", "-int8.tflite")
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert() # 转换模型
    open(f, "wb").write(tflite_model) # 写入TFLite模型
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr("Edge TPU:")):
    """
    将YOLOv5模型导出为Edge TPU兼容的TFLite格式；需要Linux和Edge TPU编译器。

    Args:
        file (Path): 要导出的YOLOv5模型文件路径 (.pt格式)。
        prefix (str, optional): 日志消息的前缀。默认为colorstr("Edge TPU:")。

    Returns:
        tuple[Path, None]: 导出到Edge TPU兼容的TFLite模型的路径，None。

    Raises:
        AssertionError: 如果系统不是Linux。
        subprocess.CalledProcessError: 如果任何调用Edge TPU编译器的子进程失败。

    Notes:
        要使用此函数，请确保您的Linux系统上安装了Edge TPU编译器。您可以在此处找到安装说明：
        https://coral.ai/docs/edgetpu/compiler/。

    Example:
        ```python
        from pathlib import Path
        file = Path('yolov5s.pt')
        export_edgetpu(file)
        ```
    """
    cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert platform.system() == "Linux", f"导出仅在Linux上支持。请参阅 {help_url}"
    if subprocess.run(f"{cmd} > /dev/null 2>&1", shell=True).returncode != 0:
        LOGGER.info(f"\n{prefix} 导出需要Edge TPU编译器。尝试从 {help_url} 安装")
        sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # 系统是否安装sudo
        for c in (
            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
            'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
            "sudo apt-get update",
            "sudo apt-get install edgetpu-compiler",
        ):
            subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
    f = str(file).replace(".pt", "-int8_edgetpu.tflite")  # Edge TPU模型
    f_tfl = str(file).replace(".pt", "-int8.tflite")  # TFLite模型

    subprocess.run(
        [
            "edgetpu_compiler",
            "-s",
            "-d",
            "-k",
            "10",
            "--out_dir",
            str(file.parent),
            f_tfl,
        ],
        check=True,
    ) # 运行Edge TPU编译器
    return f, None


@try_export
def export_tfjs(file, int8, prefix=colorstr("TensorFlow.js:")):
    """
    将YOLOv5模型转换为TensorFlow.js格式，可选uint8量化。

    Args:
        file (Path): 要转换的YOLOv5模型文件路径，通常具有".pt"或".onnx"扩展名。
        int8 (bool): 如果为True，则在转换过程中应用uint8量化。
        prefix (str): 日志消息的可选前缀，默认为带颜色格式的'TensorFlow.js:'。

    Returns:
        (str, None): 包含输出目录路径的字符串和None。

    Notes:
        - 此函数需要`tensorflowjs`包。使用以下命令安装：
          ```shell
          pip install tensorflowjs
          ```
        - 转换后的TensorFlow.js模型将保存在一个目录中，该目录的名称在原始文件名后附加"_web_model"后缀。
        - 转换涉及运行调用TensorFlow.js转换器工具的shell命令。

    Example:
        ```python
        from pathlib import Path
        file = Path('yolov5.onnx')
        export_tfjs(file, int8=False)
        ```
    """
    check_requirements("tensorflowjs")
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
    f = str(file).replace(".pt", "_web_model")  # js目录
    f_pb = file.with_suffix(".pb")  # *.pb路径
    f_json = f"{f}/model.json"  # *.json路径

    args = [
        "tensorflowjs_converter",
        "--input_format=tf_frozen_model",
        "--quantize_uint8" if int8 else "",
        "--output_node_names=Identity,Identity_1,Identity_2,Identity_3",
        str(f_pb),
        f,
    ]
    subprocess.run([arg for arg in args if arg], check=True) # 运行tensorflowjs转换器

    json = Path(f_json).read_text()
    with open(f_json, "w") as j:  # 按升序排序JSON Identity_*
        subst = re.sub(
            r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}}}',
            r'{"outputs": {"Identity": {"name": "Identity"}, '
            r'"Identity_1": {"name": "Identity_1"}, '
            r'"Identity_2": {"name": "Identity_2"}, '
            r'"Identity_3": {"name": "Identity_3"}}}',
            json,
        )
        j.write(subst)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    """
    向TensorFlow Lite (TFLite) 模型文件添加元数据，根据TensorFlow指南支持多个输出。

    Args:
        file (str): 要添加元数据的TFLite模型文件路径。
        metadata (dict): 要添加到模型的元数据信息，按照TFLite元数据模式的要求进行结构化。
            常见键包括"name"、"description"、"version"、"author"和"license"。
        num_outputs (int): 模型具有的输出张量数量，用于正确配置元数据。

    Returns:
        None

    Example:
        ```python
        metadata = {
            "name": "yolov5",
            "description": "YOLOv5 object detection model",
            "version": "1.0",
            "author": "Ultralytics",
            "license": "Apache License 2.0"
        }
        add_tflite_metadata("model.tflite", metadata, num_outputs=4)
        ```

    Note:
        TFLite元数据可以包含模型名称、版本、作者和其他相关详细信息。
        有关元数据结构的更多详细信息，请参阅TensorFlow Lite
        [元数据指南](https://ai.google.dev/edge/litert/models/metadata)。
    """
    with contextlib.suppress(ImportError):
        # check_requirements('tflite_support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path("/tmp/meta.txt")
        with open(tmp_file, "w") as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()


def pipeline_coreml(model, im, file, names, y, mlmodel, prefix=colorstr("CoreML Pipeline:")):
    """
    将PyTorch YOLOv5模型转换为CoreML格式，带非极大值抑制 (NMS)，处理不同的输入/输出形状，并保存模型。

    Args:
        model (torch.nn.Module): 要转换的YOLOv5 PyTorch模型。
        im (torch.Tensor): 示例输入张量，形状为 (N, C, H, W)，其中N是批次大小，C是通道数，
            H是高度，W是宽度。
        file (Path): 保存转换后的CoreML模型的路径。
        names (dict[int, str]): 类别索引到类别名称的字典映射。
        y (torch.Tensor): PyTorch模型前向传播的输出张量。
        mlmodel (bool): 指示是否导出为旧的*.mlmodel格式的标志 (默认为False)。
        prefix (str): 日志消息的自定义前缀。

    Returns:
        (Path): 保存的CoreML模型路径 (.mlmodel)。

    Raises:
        AssertionError: 如果类别名称的数量与模型中的类别数量不匹配。

    Notes:
        - 此函数需要安装`coremltools`。
        - 在非macOS环境下运行此函数可能不支持某些功能。
        - 灵活的输入形状和额外的NMS选项可以在函数内部自定义。

    Examples:
        ```python
        from pathlib import Path
        import torch

        model = torch.load('yolov5s.pt')  # 加载YOLOv5模型
        im = torch.zeros((1, 3, 640, 640))  # 示例输入张量

        names = {0: "person", 1: "bicycle", 2: "car", ...}  # 定义类别名称

        y = model(im)  # 执行前向传播以获取模型输出

        output_file = Path('yolov5s.mlmodel')  # 转换为CoreML
        pipeline_coreml(model, im, output_file, names, y)
        ```
    """
    import coremltools as ct
    from PIL import Image

    f = file.with_suffix(".mlmodel") if mlmodel else file.with_suffix(".mlpackage")
    print(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
    batch_size, ch, h, w = list(im.shape)  # BCHW
    t = time.time()

    # YOLOv5输出形状
    spec = model.get_spec()
    out0, out1 = iter(spec.description.output)
    if platform.system() == "Darwin":
        img = Image.new("RGB", (w, h))  # img(192 width, 320 height)
        # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
        out = model.predict({"image": img})
        out0_shape, out1_shape = out[out0.name].shape, out[out1.name].shape
    else:  # linux和windows无法运行model.predict()，从pytorch输出y获取尺寸
        s = tuple(y[0].shape)
        out0_shape, out1_shape = (s[1], s[2] - 5), (s[1], 4)  # (3780, 80), (3780, 4)

    # 检查
    nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
    na, nc = out0_shape
    # na, nc = out0.type.multiArrayType.shape  # 锚点数量，类别数量
    assert len(names) == nc, f"{len(names)}个名称与nc={nc}不匹配"  # 检查

    # 定义输出形状 (缺失)
    out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
    out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
    # spec.neuralNetwork.preprocessing[0].featureName = '0'

    # 灵活输入形状
    # from coremltools.models.neural_network import flexible_shape_utils
    # s = [] # 形状
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (高度, 宽度)
    # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
    # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # 形状范围
    # r.add_height_range((192, 640))
    # r.add_width_range((192, 640))
    # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

    # 打印
    print(spec.description)

    # 从spec创建模型
    weights_dir = None
    weights_dir = None if mlmodel else str(f / "Data/com.apple.CoreML/weights")
    model = ct.models.MLModel(spec, weights_dir=weights_dir)

    # 3. 创建NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = "confidence"
    nms_spec.description.output[1].name = "coordinates"

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name  # 1x507x80
    nms.coordinatesInputFeatureName = out1.name  # 1x507x4
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    # 4. 将模型组合成管道
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=["confidence", "coordinates"],
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # 修正数据类型
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # 更新元数据
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.versionString = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.shortDescription = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.author = "glenn.jocher@ultralytics.com"
    pipeline.spec.description.metadata.license = "https://github.com/ultralytics/yolov5/blob/master/LICENSE"
    pipeline.spec.description.metadata.userDefined.update(
        {
            "classes": ",".join(names.values()),
            "iou_threshold": str(nms.iouThreshold),
            "confidence_threshold": str(nms.confidenceThreshold),
        }
    )

    # 保存模型
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
    model.input_description["image"] = "输入图像"
    model.input_description["iouThreshold"] = f"(可选) IoU阈值覆盖 (默认: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(可选) 置信度阈值覆盖 (默认: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = '框 × 类别置信度 (参见用户定义元数据 "classes")'
    model.output_description["coordinates"] = "框 × [x, y, 宽度, 高度] (相对于图像尺寸)"
    model.save(f)  # 管道化模型
    print(f"{prefix} 管道成功 ({time.time() - t:.2f}s), 保存为 {f} ({file_size(f):.1f} MB)")


@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml路径'
    weights=ROOT / "yolov5s.pt",  # 权重路径
    imgsz=(640, 640),  # 图像 (高, 宽)
    batch_size=1,  # 批次大小
    device="cpu",  # cuda设备，例如 0 或 0,1,2,3 或 cpu
    include=("torchscript", "onnx"),  # 包含格式
    half=False,  # FP16半精度导出
    inplace=False,  # 设置YOLOv5 Detect() inplace=True
    keras=False,  # 使用Keras
    optimize=False,  # TorchScript: 优化用于移动端
    int8=False,  # CoreML/TF INT8量化
    per_tensor=False,  # TF逐张量量化
    dynamic=False,  # ONNX/TF/TensorRT: 动态轴
    cache="",  # TensorRT: 计时缓存路径
    simplify=False,  # ONNX: 简化模型
    mlmodel=False,  # CoreML: 导出为*.mlmodel格式
    opset=12,  # ONNX: opset版本
    verbose=False,  # TensorRT: 详细日志
    workspace=4,  # TensorRT: 工作空间大小 (GB)
    nms=False,  # TF: 添加NMS到模型
    agnostic_nms=False,  # TF: 添加类别无关NMS到模型
    topk_per_class=100,  # TF.js NMS: 每类保留的topk
    topk_all=100,  # TF.js NMS: 所有类别保留的topk
    iou_thres=0.45,  # TF.js NMS: IoU阈值
    conf_thres=0.25,  # TF.js NMS: 置信度阈值
):
    """
    将YOLOv5模型导出为指定格式，包括ONNX、TensorRT、CoreML和TensorFlow。

    Args:
        data (str | Path): 数据集YAML配置文件的路径。默认为'data/coco128.yaml'。
        weights (str | Path): 预训练模型权重文件的路径。默认为'yolov5s.pt'。
        imgsz (tuple): 图像尺寸，格式为 (高度, 宽度)。默认为 (640, 640)。
        batch_size (int): 导出模型的批次大小。默认为1。
        device (str): 运行导出的设备，例如'0'表示GPU，'cpu'表示CPU。默认为'cpu'。
        include (tuple): 导出中包含的格式。默认为 ('torchscript', 'onnx')。
        half (bool): 标志，用于以FP16半精度导出模型。默认为False。
        inplace (bool): 设置YOLOv5 Detect()模块inplace=True。默认为False。
        keras (bool): 标志，用于TensorFlow SavedModel导出时使用Keras。默认为False。
        optimize (bool): 优化TorchScript模型以用于移动部署。默认为False。
        int8 (bool): 对CoreML或TensorFlow模型应用INT8量化。默认为False。
        per_tensor (bool): 对TensorFlow模型应用逐张量量化。默认为False。
        dynamic (bool): 为ONNX、TensorFlow或TensorRT导出启用动态轴。默认为False。
        cache (str): TensorRT计时缓存路径。默认为空字符串。
        simplify (bool): 在导出期间简化ONNX模型。默认为False。
        opset (int): ONNX opset版本。默认为12。
        verbose (bool): 为TensorRT导出启用详细日志记录。默认为False。
        workspace (int): TensorRT工作空间大小 (GB)。默认为4。
        nms (bool): 向TensorFlow模型添加非极大值抑制 (NMS)。默认为False。
        agnostic_nms (bool): 向TensorFlow模型添加类别无关NMS。默认为False。
        topk_per_class (int): TensorFlow.js NMS中每类保留的Top-K框。默认为100。
        topk_all (int): TensorFlow.js NMS中所有类别保留的Top-K框。默认为100。
        iou_thres (float): NMS的IoU阈值。默认为0.45。
        conf_thres (float): NMS的置信度阈值。默认为0.25。
        mlmodel (bool): 标志，用于CoreML导出时使用*.mlmodel。默认为False。

    Returns:
        None

    Notes:
        - 模型导出基于'include'参数中指定的格式。
        - 请注意某些标志相互排斥的组合，例如`--half`和`--dynamic`。

    Example:
        ```python
        run(
            data="data/coco128.yaml",
            weights="yolov5s.pt",
            imgsz=(640, 640),
            batch_size=1,
            device="cpu",
            include=("torchscript", "onnx"),
            half=False,
            inplace=False,
            keras=False,
            optimize=False,
            int8=False,
            per_tensor=False,
            dynamic=False,
            cache="",
            simplify=False,
            opset=12,
            verbose=False,
            mlmodel=False,
            workspace=4,
            nms=False,
            agnostic_nms=False,
            topk_per_class=100,
            topk_all=100,
            iou_thres=0.45,
            conf_thres=0.25,
        )
        ```
    """
    t = time.time()
    include = [x.lower() for x in include]  # 转换为小写
    fmts = tuple(export_formats()["Argument"][1:])  # --include参数
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"错误: 无效的--include {include}, 有效的--include参数是 {fmts}"
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # 导出布尔值
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)  # PyTorch权重

    # 加载PyTorch模型
    device = select_device(device)
    if half:
        assert device.type != "cpu" or coreml, "--half仅与GPU导出兼容，例如使用--device 0"
        assert not dynamic, "--half与--dynamic不兼容，例如使用--half或--dynamic但不能同时使用"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # 加载FP32模型

    # 检查
    imgsz *= 2 if len(imgsz) == 1 else 1  # 扩展
    if optimize:
        assert device.type == "cpu", "--optimize与cuda设备不兼容，例如使用--device cpu"

    # 输入
    gs = int(max(model.stride))  # 网格大小 (最大步长)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # 验证img_size是gs的倍数
    ch = next(model.parameters()).size(1)  # 需要输入图像通道
    im = torch.zeros(batch_size, ch, *imgsz).to(device)  # 图像尺寸(1,3,320,192) BCHW iDetection

    # 更新模型
    model.eval() # 设置模型为评估模式
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # 空运行
    if half and not coreml:
        im, model = im.half(), model.half()  # 转换为FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # 模型输出形状
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # 模型元数据
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # 导出
    f = [""] * len(fmts)  # 导出文件名
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)  # 抑制TracerWarning
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT在ONNX之前需要
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose, cache)
    if onnx or xml:  # OpenVINO需要ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half, int8, data)
    if coreml:  # CoreML
        f[4], ct_model = export_coreml(model, im, file, int8, half, nms, mlmodel)
        if nms:
            pipeline_coreml(ct_model, im, file, model.names, y, mlmodel)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow格式
        assert not tflite or not tfjs, "TFLite和TF.js模型必须单独导出，请仅传递一种类型。"
        assert not isinstance(model, ClassificationModel), "分类模型导出到TF格式尚不支持。"
        f[5], s_model = export_saved_model(
            model.cpu(),
            im,
            file,
            dynamic,
            tf_nms=nms or agnostic_nms or tfjs,
            agnostic_nms=agnostic_nms or tfjs,
            topk_per_class=topk_per_class,
            topk_all=topk_all,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            keras=keras,
        )
        if pb or tfjs:  # pb是tfjs的先决条件
            f[6], _ = export_pb(s_model, file)
        if tflite or edgetpu:
            f[7], _ = export_tflite(
                s_model, im, file, int8 or edgetpu, per_tensor, data=data, nms=nms, agnostic_nms=agnostic_nms
            )
            if edgetpu:
                f[8], _ = export_edgetpu(file)
            add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(s_model.outputs))
        if tfjs:
            f[9], _ = export_tfjs(file, int8)
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # 完成
    f = [str(x) for x in f if x]  # 过滤掉''和None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # 类型
        det &= not seg  # 分割模型继承自SegmentationModel(DetectionModel)
        dir = Path("segment" if seg else "classify" if cls else "")
        h = "--half" if half else ""  # --half FP16推理参数
        s = (
            "# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference"
            if cls
            else "# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference"
            if seg
            else ""
        )
        LOGGER.info(
            f"\n导出完成 ({time.time() - t:.1f}s)"
            f"\n结果保存到 {colorstr('bold', file.parent.resolve())}"
            f"\n检测:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\n验证:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f"\n可视化:       https://netron.app"
        )
    return f  # 返回导出文件/目录列表


def parse_opt(known=False):
    """
    解析YOLOv5模型导出配置的命令行选项。

    Args:
        known (bool): 如果为True，则使用`argparse.ArgumentParser.parse_known_args`；否则，使用`argparse.ArgumentParser.parse_args`。
                      默认为False。

    Returns:
        argparse.Namespace: 包含解析后的命令行参数的对象。

    Example:
        ```python
        opts = parse_opt()
        print(opts.data)
        print(opts.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--cache", type=str, default="", help="TensorRT: timing cache file path")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--mlmodel", action="store_true", help="CoreML: Export in *.mlmodel format")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt)) # 打印参数
    return opt


def main(opt):
    """Run(**vars(opt))  # 使用解析的选项执行run函数。"""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt() # 解析命令行参数
    main(opt) # 调用主函数
