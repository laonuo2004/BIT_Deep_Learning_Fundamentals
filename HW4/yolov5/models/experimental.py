# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            # 如果启用权重，则初始化可学习的权重参数
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            # 如果启用权重，则对权重进行 sigmoid 激活并乘以 2
            w = torch.sigmoid(self.w) * 2
            # 对除第一个输入外的其他输入应用权重并累加
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            # 如果不启用权重，则直接累加所有输入
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            # 如果通道数均等分配，则计算每个组的中间通道数
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            # 如果不均等分配，则通过线性方程组求解中间通道数
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        # 创建多个深度可分离卷积层，每个层使用不同的核大小
        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        # 将所有卷积层的输出拼接，然后进行批归一化和激活
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        # 对每个模型进行前向传播，并获取其输出
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        # 将所有模型的输出在通道维度上拼接
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    # 遍历所有权重文件，加载模型
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        # 检查并设置模型步长
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        # 检查并设置类别名称
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        # 将模型添加到集成中，并根据 fuse 参数决定是否融合和设置为评估模式
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    # 对模型中的特定模块进行 inplace 操作和兼容性更新
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    # 如果只有一个模型，则直接返回该模型
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    # 为集成模型设置名称、类别数和 yaml 配置
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    # 设置集成模型的最大步长
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    # 确保所有模型的类别数相同
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
