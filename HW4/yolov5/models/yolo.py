# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个锚点输出的数量 (x, y, w, h, conf, class_probs)
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 锚点数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化锚点网格
        # 注册锚点为 buffer，不作为模型参数，但随模型保存
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 输出卷积层，将特征图通道数转换为适合检测头的输出通道数
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # 是否使用原地操作

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # 推理输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 重塑输出张量为 (batch_size, anchors, grid_y, grid_x, outputs_per_anchor)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理模式
                # 如果是动态模式或网格大小与当前特征图大小不匹配，则重新生成网格
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    # 分割输出为 xy、wh、conf 和 mask
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    # 计算预测框的中心坐标 (xy) 和宽高 (wh)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    # 拼接 xywh、置信度、mask
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    # 分割输出为 xy、wh 和 conf
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # 计算预测框的中心坐标 (xy) 和宽高 (wh)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    # 拼接 xywh、置信度
                    y = torch.cat((xy, wh, conf), 4)
                # 重塑为 (batch_size, total_predictions, outputs_per_anchor)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        # 如果是训练模式，返回原始输出；否则返回拼接后的预测结果 (用于推理) 或 (拼接后的预测结果, 原始输出) (用于导出)
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # 网格形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # 生成网格坐标
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # 添加网格偏移，并扩展到指定形状
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # 计算锚点网格
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.no = 5 + nc + self.nm  # 每个锚点输出的数量 (x, y, w, h, conf, class_probs, mask_coeffs)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型层
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        # 通过原型层生成原型掩码
        p = self.proto(x[0])
        # 调用父类的检测方法
        x = self.detect(self, x)
        # 如果是训练模式，返回检测结果和原型掩码；如果是导出模式，返回检测结果和原型掩码；否则返回检测结果、原型掩码和原始输出
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                # 从之前的层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行模块
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                delattr(m, "bn")  # 移除批归一化层
                m.forward = m.forward_fuse  # 更新前向传播方法
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 模型字典
        else:  # 是 *.yaml 文件
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # 模型字典

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # 覆盖 yaml 值
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # 覆盖 yaml 值
        # 解析模型配置，构建模型层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # 默认类别名称
        self.inplace = self.yaml.get("inplace", True)

        # 构建步长，锚点
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 计算步长
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # 检查锚点顺序
            m.anchors /= m.stride.view(-1, 1, 1)  # 锚点归一化
            self.stride = m.stride
            self._initialize_biases()  # 只运行一次

        # 初始化权重，偏置
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # 增强推理
        return self._forward_once(x, profile, visualize)  # 单尺度推理，训练

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # 高度，宽度
        s = [1, 0.83, 0.67]  # 尺度
        f = [None, 3, None]  # 翻转 (2-上下翻转, 3-左右翻转)
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # 前向传播
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)  # 反归一化预测结果
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强尾部
        return torch.cat(y, 1), None  # 增强推理，训练

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # 反归一化
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # 上下翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # 左右翻转
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # 反归一化
            if flips == 2:
                y = img_size[0] - y  # 上下翻转
            elif flips == 3:
                x = img_size[1] - x  # 左右翻转
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # 检测层数量 (P3-P5)
        g = sum(4**x for x in range(nl))  # 网格点数
        e = 1  # 排除层数
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][:, :-i]  # 大尺度
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][:, i:]  # 小尺度
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # 初始化目标置信度偏置
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            # 初始化类别置信度偏置
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # 保留 YOLOv5 'Model' 类以实现向后兼容


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # 解包 DetectMultiBackend
        model.model = model.model[:cutoff]  # 骨干网络
        m = model.model[-1]  # 最后一层
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # 模块输入通道
        c = Classify(ch, nc)  # 分类层
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # 索引，来自，类型
        model.model[-1] = c  # 替换
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数，例如 Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚点数量
    no = na * (nc + 5)  # 输出数量 = 锚点数量 * (类别数 + 5)

    layers, save, c2 = [], [], ch[-1]  # 层，保存列表，输出通道
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 评估字符串
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # 评估字符串

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 深度增益
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是输出层
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # 重复次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚点数量
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 附加索引，'来自'索引，类型，参数数量
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # 检查 YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # 选项
    if opt.line_profile:  # 逐层分析模型速度
        model(im, profile=True)

    elif opt.profile:  # 分析前向-后向传播
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # 测试所有 yolo*.yaml 模型
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # 报告融合模型摘要
        model.fuse()
