from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    """
    ResNet中的Bottleneck模块。
    - 包含三个卷积层：1x1, 3x3, 1x1。
    - 3x3卷积层后可能接平均池化层用于下采样。
    - 包含残差连接。
    """
    expansion = 4 # Bottleneck模块的输出通道是输入通道的4倍

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 第一个1x1卷积，用于降维
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个3x3卷积，进行特征提取
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 如果步长大于1，则在第二个卷积后进行平均池化以实现下采样
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个1x1卷积，用于升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # 如果需要下采样或通道数不匹配，则创建残差连接的下采样分支
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)), # 先进行平均池化
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), # 再进行1x1卷积调整通道数
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x # 保存输入作为残差连接的identity

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out) # 如果stride > 1，在此处进行下采样
        out = self.bn3(self.conv3(out))

        # 如果存在下采样分支，则对identity进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # 残差连接
        out = self.relu3(out) # 最终激活
        return out


class AttentionPool2d(nn.Module):
    """
    用于图像特征提取的注意力池化层。
    - 将2D特征图展平并添加位置编码。
    - 使用多头注意力机制对特征进行池化，提取全局表示。
    - 类似于ViT中的Class Token机制，但这里是针对ResNet特征图。
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 位置编码，包含一个额外的token用于全局特征（类似于Class Token）
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # QKV投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出投影层
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # NCHW -> (HW)NC: 将特征图展平，并调整维度顺序
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        # (HW+1)NC: 在序列开头添加一个平均池化后的全局特征，作为查询Q
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # (HW+1)NC: 添加位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        # 执行多头注意力机制
        # query: x[:1] (全局特征), key: x (所有特征), value: x (所有特征)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0) # 返回全局特征


class ModifiedResNet(nn.Module):
    """
    一个修改过的ResNet模型，用于CLIP的视觉编码器。
    - 包含3层“stem”卷积，而不是传统的1层，并使用平均池化代替最大池化。
    - 在步长大于1的卷积前添加平均池化，实现抗锯齿下采样。
    - 最终的池化层是QKV注意力池化（AttentionPool2d），而不是全局平均池化。
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem (初始的3层卷积，用于提取浅层特征)
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2) # stem后的平均池化

        # residual layers (残差层)
        self._inplanes = width  # 记录当前输入通道数，用于构建后续层
        self.layer1 = self._make_layer(width, layers[0]) # 构建第一个残差块组
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2) # 构建第二个残差块组，并进行下采样
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2) # 构建第三个残差块组，并进行下采样
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2) # 构建第四个残差块组，并进行下采样

        embed_dim = width * 32  # ResNet特征维度，即最后一个残差块组的输出通道数
        # 最终的注意力池化层，用于将特征图转换为固定维度的向量
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        构建ResNet的残差层。
        - planes: 当前层的输出通道数（Bottleneck内部的planes）。
        - blocks: 当前层包含的Bottleneck块的数量。
        - stride: 第一个Bottleneck块的步长，用于控制下采样。
        """
        # 第一个Bottleneck块可能进行下采样
        layers = [Bottleneck(self._inplanes, planes, stride)]

        # 更新_inplanes为当前Bottleneck块的输出通道数，用于后续块的输入
        self._inplanes = planes * Bottleneck.expansion
        # 构建剩余的Bottleneck块
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            # 3层stem卷积和激活
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x) # stem后的平均池化
            return x

        x = x.type(self.conv1.weight.dtype) # 确保输入数据类型与模型权重一致
        x = stem(x) # 运行stem
        x = self.layer1(x) # 运行layer1
        x = self.layer2(x) # 运行layer2
        x = self.layer3(x) # 运行layer3
        x = self.layer4(x) # 运行layer4
        x = self.attnpool(x) # 运行注意力池化

        return x


class LayerNorm(nn.LayerNorm):
    """
    继承自torch.nn.LayerNorm，用于处理fp16（半精度浮点数）数据类型。
    - 在计算时将输入转换为float32，计算完成后再转回原始类型，以避免精度问题。
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype # 记录原始数据类型
        ret = super().forward(x.type(torch.float32)) # 转换为float32进行计算
        return ret.type(orig_type) # 转换回原始数据类型


class QuickGELU(nn.Module):
    """
    QuickGELU激活函数，是GELU的近似版本。
    - 计算公式为 $x \cdot \sigma(1.702 \cdot x)$，其中 $\sigma$ 是Sigmoid函数。
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer编码器中的一个残差注意力块。
    - 包含一个多头自注意力层和一个MLP（多层感知机）块。
    - 每个子层都应用残差连接和层归一化。
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) # 多头自注意力层
        self.ln_1 = LayerNorm(d_model) # 第一个层归一化
        self.mlp = nn.Sequential(OrderedDict([ # MLP块
            ("c_fc", nn.Linear(d_model, d_model * 4)), # 线性层，维度扩展
            ("gelu", QuickGELU()), # QuickGELU激活函数
            ("c_proj", nn.Linear(d_model * 4, d_model)) # 线性层，维度还原
        ]))
        self.ln_2 = LayerNorm(d_model) # 第二个层归一化
        self.attn_mask = attn_mask # 注意力掩码，用于文本编码器中的因果掩码

    def attention(self, x: torch.Tensor):
        # 将注意力掩码移动到正确的设备和数据类型
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 执行多头自注意力
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x)) # 自注意力层，残差连接和层归一化
        x = x + self.mlp(self.ln_2(x)) # MLP层，残差连接和层归一化
        return x


class Transformer(nn.Module):
    """
    Transformer编码器，由多个ResidualAttentionBlock组成。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width # 模型维度
        self.layers = layers # 块的数量
        # 堆叠ResidualAttentionBlock
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer模型，用于CLIP的视觉编码器。
    - 将图像分割成patch，进行线性投影。
    - 添加可学习的Class Embedding和位置编码。
    - 通过Transformer编码器处理序列。
    - 最终通过线性投影得到图像特征。
    """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution # 输入图像分辨率
        self.output_dim = output_dim # 输出特征维度
        # 图像patch嵌入层：将图像分割成patch并进行线性投影
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # 可学习的Class Token
        # 位置编码：包含Class Token和所有patch的位置编码
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width) # Transformer前的层归一化

        self.transformer = Transformer(width, layers, heads) # Transformer编码器

        self.ln_post = LayerNorm(width) # Transformer后的层归一化
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) # 最终的线性投影层

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [batch_size, width, grid, grid] (将图像转换为patch嵌入)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch_size, width, grid ** 2] (展平grid)
        x = x.permute(0, 2, 1)  # shape = [batch_size, grid ** 2, width] (调整维度顺序)
        # 拼接Class Token和patch嵌入，并添加位置编码
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [batch_size, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) # Transformer前的层归一化

        x = x.permute(1, 0, 2)  # NLD -> LND (调整维度顺序以适应Transformer输入)
        x = self.transformer(x) # 运行Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD (调整维度顺序回NLD)

        x = self.ln_post(x[:, 0, :]) # 取出Class Token的输出，并进行层归一化

        if self.proj is not None:
            x = x @ self.proj # 线性投影到最终输出维度

        return x


class CLIP(nn.Module):
    """
    CLIP模型的主体结构。
    - 包含一个视觉编码器（ModifiedResNet或VisionTransformer）和一个文本编码器（Transformer）。
    - 负责图像和文本的特征提取，并计算它们之间的相似度。
    """
    def __init__(self,
                 embed_dim: int, # 嵌入维度，图像和文本特征的共同维度
                 # vision
                 image_resolution: int, # 图像输入分辨率
                 vision_layers: Union[Tuple[int, int, int, int], int], # 视觉编码器层数配置
                 vision_width: int, # 视觉编码器宽度
                 vision_patch_size: int, # Vision Transformer的patch大小
                 # text
                 context_length: int, # 文本序列的最大长度
                 vocab_size: int, # 词汇表大小
                 transformer_width: int, # 文本Transformer的宽度
                 transformer_heads: int, # 文本Transformer的头数
                 transformer_layers: int # 文本Transformer的层数
                 ):
        super().__init__()

        self.context_length = context_length

        # 根据vision_layers的类型选择视觉编码器：ResNet或VisionTransformer
        if isinstance(vision_layers, (tuple, list)): # 如果是元组或列表，表示使用ResNet
            vision_heads = vision_width * 32 // 64 # ResNet的注意力池化头数
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else: # 否则使用VisionTransformer
            vision_heads = vision_width // 64 # Vision Transformer的头数
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 文本Transformer编码器
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask() # 构建因果注意力掩码
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # 文本token嵌入层
        # 文本位置编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width) # 文本Transformer后的层归一化

        # 文本特征投影层，将文本Transformer的输出投影到与图像特征相同的嵌入维度
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # 对数尺度参数，用于控制图像和文本特征相似度计算的尺度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters() # 初始化模型参数

    def initialize_parameters(self):
        """
        初始化模型参数。
        - 使用正态分布初始化token嵌入和位置编码。
        - 对ResNet和Transformer中的特定层进行初始化。
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"): # 将ResNet中Bottleneck的最后一个BatchNorm层的权重初始化为0
                        nn.init.zeros_(param)

        # Transformer层的初始化
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构建因果注意力掩码。
        - 掩码是一个上三角矩阵，对角线及以下为0，对角线以上为-inf。
        - 确保文本编码器在生成当前token的表示时，只能关注到当前token及之前的token。
        """
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf")) # 填充负无穷
        mask.triu_(1)  # zero out the lower diagonal (将上三角部分（不含对角线）设置为0)
        return mask

    @property
    def dtype(self):
        """
        返回模型权重的数据类型。
        """
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        图像编码器前向传播。
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        文本编码器前向传播。
        - 将token ID转换为嵌入向量。
        - 添加位置编码。
        - 通过Transformer处理。
        - 提取[EOS] token的特征作为文本表示。
        - 线性投影到嵌入维度。
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model] (token嵌入)

        x = x + self.positional_embedding.type(self.dtype) # 添加位置编码
        x = x.permute(1, 0, 2)  # NLD -> LND (调整维度顺序以适应Transformer输入)
        x = self.transformer(x) # 运行Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD (调整维度顺序回NLD)
        x = self.ln_final(x).type(self.dtype) # 最终层归一化

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 提取每个序列中[EOS] token（即argmax(dim=-1)对应的token）的特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # 线性投影

        return x

    def forward(self, image, text):
        """
        CLIP模型的前向传播。
        - 分别编码图像和文本。
        - 对特征进行L2归一化。
        - 计算图像和文本特征之间的余弦相似度作为logits。
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features (L2归一化)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits (计算余弦相似度作为logits)
        logit_scale = self.logit_scale.exp() # 将对数尺度参数转换为实际尺度
        logits_per_image = logit_scale * image_features @ text_features.t() # 图像对文本的logits
        logits_per_text = logits_per_image.t() # 文本对图像的logits

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    将模型中适用参数转换为fp16（半精度浮点数）。
    - 提高计算效率和减少内存占用。
    """

    def _convert_weights_to_fp16(l):
        # 卷积层和线性层
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # 多头注意力层
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # 文本投影和视觉投影层
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16) # 递归应用转换函数


def build_model(state_dict: dict):
    """
    根据预训练模型的state_dict构建CLIP模型。
    - 从state_dict中解析模型配置参数。
    - 实例化CLIP模型。
    - 将模型参数转换为fp16。
    - 加载state_dict并设置为评估模式。
    """
    # 判断视觉编码器是Vision Transformer还是ResNet
    vit = "visual.proj" in state_dict

    if vit: # Vision Transformer
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # 计算Vision Transformer的层数
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        # 计算图像分辨率
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: # Modified ResNet
        # 计算ResNet的层数（每个layer组的Bottleneck块数量）
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        # 计算图像分辨率
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None # ResNet没有patch_size概念
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # 从state_dict中提取其他模型参数
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 实例化CLIP模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 删除state_dict中不再需要的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model) # 将模型参数转换为fp16
    model.load_state_dict(state_dict) # 加载预训练权重
    return model.eval() # 设置为评估模式
