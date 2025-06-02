import hashlib # 用于校验文件哈希值，确保下载文件的完整性
import os # 用于路径操作和文件系统交互
import urllib # 用于从 URL 下载文件
import warnings # 用于发出警告信息
from packaging import version # 用于比较 PyTorch 版本
from typing import Union, List # 类型提示

import torch # PyTorch 核心库
from PIL import Image # Pillow 库，用于图像处理
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize # TorchVision 中的图像预处理方法
from tqdm import tqdm # 用于显示下载进度条

from .model import build_model # 从同级目录的 model.py 导入模型构建函数
from .simple_tokenizer import SimpleTokenizer as _Tokenizer # 从同级目录的 simple_tokenizer.py 导入分词器

try:
    from torchvision.transforms import InterpolationMode # 尝试导入新的插值模式
    BICUBIC = InterpolationMode.BICUBIC # 使用 torchvision 新版本的双三次插值
except ImportError:
    BICUBIC = Image.BICUBIC # 如果导入失败，则使用 PIL 库中的双三次插值


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended") # 检查 PyTorch 版本，如果低于1.7.1则发出警告


__all__ = ["available_models", "load", "tokenize"] # 定义了当 `from clip import *` 时会导出的公共接口
_tokenizer = _Tokenizer() # 初始化分词器实例，供 tokenize 函数使用

# _MODELS 字典存储了预训练模型的名称及其对应的下载链接
# 每个模型的 URL 结构中包含了 SHA256 哈希值，用于下载后的文件校验
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt", # 注意这个模型指定了不同的输入分辨率
}


def _download(url: str, root: str): # 定义私有下载函数
    """
    从给定的URL下载文件到指定根目录，并进行SHA256校验。
    """
    os.makedirs(root, exist_ok=True) # 创建下载目录，如果目录已存在则不报错
    filename = os.path.basename(url) # 从 URL 中获取文件名

    expected_sha256 = url.split("/")[-2] # 从 URL 中提取预期的 SHA256 哈希值
    download_target = os.path.join(root, filename) # 拼接完整的下载路径

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file") # 如果目标路径存在但不是文件，则抛出异常

    if os.path.isfile(download_target):
        # 如果文件已存在，则校验其 SHA256 哈希值
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target # 哈希值匹配，直接返回文件路径
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file") # 哈希值不匹配，警告并重新下载

    # 下载文件并显示进度条
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192) # 每次读取 8KB
                if not buffer:
                    break # 读取完毕

                output.write(buffer)
                loop.update(len(buffer)) # 更新进度条

    # 再次校验下载文件的 SHA256 哈希值
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match") # 哈希值不匹配，抛出异常

    return download_target


def _convert_image_to_rgb(image): # 定义私有函数，将图像转换为 RGB 格式
    """
    将PIL图像转换为RGB模式。
    """
    return image.convert("RGB")


def _transform(n_px: int): # 定义私有的图像预处理函数
    """
    创建图像预处理管道。
    - n_px: 图像的目标分辨率。
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC), # 将图像短边缩放到 n_px，保持宽高比，使用双三次插值
        CenterCrop(n_px), # 从中心裁剪出 n_px x n_px 大小的图像
        _convert_image_to_rgb, # 转换为 RGB 格式
        ToTensor(), # 将 PIL Image 或 numpy.ndarray 转换为 FloatTensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # 使用 CLIP 论文中指定的均值和标准差进行归一化
        # 这些均值和标准差是根据 WIT 数据集计算得出的
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    # 返回可用的预训练模型名称列表
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
        # 模型名称，可以是 `available_models()` 返回的预定义名称，或者是包含 state_dict 的模型检查点文件路径
    device : Union[str, torch.device]
        The device to put the loaded model
        # 指定模型加载到的设备（如 "cpu" 或 "cuda"）
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
        # 是否加载 TorchScript JIT 优化的模型。默认为 False，加载更易于修改的非 JIT 模型。
        # JIT 模型通常运行更快，但可调试性和灵活性较差。
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
        # 模型文件的下载根目录，默认为用户缓存目录下的 "clip" 文件夹

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
        # 加载的 CLIP 模型实例
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        # 一个图像预处理函数，用于将 PIL 图像转换为模型可接受的张量格式
    """
    if name in _MODELS:
        # 如果是预定义模型名称，则下载模型文件
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 如果是文件路径，则直接使用该路径
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}") # 模型未找到，抛出异常

    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试加载 JIT 归档文件
            # JIT (Just-In-Time) 编译可以将 PyTorch 模型序列化为一个可移植的、优化的格式
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval() # 如果 jit 为 True，则直接加载到目标 device；否则先加载到 "cpu"
            state_dict = None # JIT 模型不需要单独的 state_dict
        except RuntimeError:
            # 加载失败，则尝试作为保存的 state_dict 加载
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False # 如果原本期望加载 JIT 模型但失败，则回退到非 JIT 模式
            state_dict = torch.load(opened_file, map_location="cpu") # state_dict 总是先加载到 CPU，后续再转移到目标设备

    if not jit:
        # 如果不使用 JIT 模型，则通过 build_model 函数构建模型并加载 state_dict
        # state_dict or model.state_dict() 的逻辑：如果 state_dict 非空（即从 .pt 文件加载），则使用它；否则（从 JIT 模型转换而来），使用 model.state_dict()
        model_state_dict = state_dict or model.state_dict() # 获取模型参数
        model = build_model(model_state_dict).to(device) # 构建模型并将其移动到指定设备
        if str(device) == "cpu":
            model.float() # 如果在 CPU 上运行，确保模型参数是 float32 类型
        # 返回模型实例和对应的图像预处理函数
        # model.visual.input_resolution 是从模型配置中读取的图像输入分辨率
        return model, _transform(model.visual.input_resolution)

    # --- 以下是处理 JIT 模型的特殊逻辑 ---
    # JIT 模型在加载后，可能需要对其内部的设备和数据类型进行一些调整（patching）
    # 这是因为 JIT 模型在序列化时会硬编码一些设备信息

    # patch the device names: 修正 JIT 模型中的设备名称
    # 创建一个临时的 JIT 跟踪模块，用于获取目标设备的表示
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1] # 获取代表目标设备的节点

    def _node_get(node: torch._C.Node, key: str): # 辅助函数，用于获取 JIT 节点的属性
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module): # 递归地修改 JIT 模块图中的设备常量
        """
        递归地修改JIT模块图中的设备常量，将其从硬编码的设备（如"cuda"）替换为目标设备。
        """
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError: # 有些 JIT 模块可能没有 graph 属性或者访问会出错
            graphs = []

        if hasattr(module, "forward1"): # 兼容旧版 JIT 模型可能存在的 forward1
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"): # 遍历所有常量节点
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    # 如果常量值是 "cuda" 开头的设备字符串，则替换为目标设备的节点
                    node.copyAttributes(device_node)

    model.apply(patch_device) # 对整个模型应用设备修正
    patch_device(model.encode_image) # 对图像编码器应用设备修正
    patch_device(model.encode_text) # 对文本编码器应用设备修正

    # patch dtype to float32 on CPU: 如果目标设备是 CPU，修正 JIT 模型中的数据类型为 float32
    if str(device) == "cpu":
        # 创建一个临时的 JIT 跟踪模块，用于获取 float32 数据类型的表示
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1] # aten::to 是 PyTorch 中用于类型转换的操作
        float_node = float_input.node() # 获取代表 float32 类型的节点

        def patch_float(module): # 递归地修改 JIT 模块图中的数据类型常量
            """
            递归地修改JIT模块图中的数据类型常量，将float16替换为float32（仅当设备为CPU时）。
            """
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"): # 遍历所有类型转换节点
                    inputs = list(node.inputs())
                    # aten::to 的第二个或第三个参数通常是目标 dtype
                    # value 5 通常代表 torch.float16 (half)
                    for i in [1, 2]: 
                        if _node_get(inputs[i].node(), "value") == 5: # 如果目标类型是 float16
                            inputs[i].node().copyAttributes(float_node) # 则替换为 float32 节点

        model.apply(patch_float) # 对整个模型应用类型修正
        patch_float(model.encode_image) # 对图像编码器应用类型修正
        patch_float(model.encode_text) # 对文本编码器应用类型修正

        model.float() # 最后确保整个模型在 CPU 上是 float32 类型

    # JIT 模型返回时，图像预处理函数中的分辨率需要从 model.input_resolution.item() 获取
    # 这是因为 JIT 序列化的模型可能将其属性保存为张量
    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    # 返回给定输入字符串（或字符串列表）的 tokenized 表示

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
        # 要进行分词的单个字符串或字符串列表
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
        # 使用的上下文长度，所有 CLIP 模型都使用 77 作为上下文长度
        # 这是因为文本 Transformer 的位置编码是固定长度的
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
        # 如果文本编码后的长度超过上下文长度，是否进行截断。默认为 False，即超长会报错。
        # 如果为 True，则截断到 context_length，并将最后一个 token 设置为 EOT token。

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    # 返回一个二维张量，包含生成的 token。形状为 [输入字符串数量, context_length]。
    # 当 PyTorch 版本低于 1.8.0 时返回 LongTensor，否则返回 IntTensor，以兼容旧版 PyTorch 的 `index_select` 操作。
    """
    if isinstance(texts, str):
        texts = [texts] # 如果输入是单个字符串，则转换为列表

    sot_token = _tokenizer.encoder["<|startoftext|>"] # 获取文本开始标记 (Start Of Text) 的 token ID
    eot_token = _tokenizer.encoder["<|endoftext|>"] # 获取文本结束标记 (End Of Text) 的 token ID
    
    # 对每个文本进行编码，并在首尾分别添加 SOT 和 EOT token
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 根据 PyTorch 版本决定结果张量的数据类型
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long) # 对于旧版本 PyTorch，使用 LongTensor
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int) # 对于新版本 PyTorch，使用 IntTensor

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length: # 如果编码后的 token 数量超过了上下文长度
            if truncate: # 如果允许截断
                tokens = tokens[:context_length] # 截断到 context_length
                tokens[-1] = eot_token # 确保最后一个 token 是 EOT token
            else: # 如果不允许截断，则抛出运行时错误
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens) # 将 token 序列填充到结果张量中，不足部分保持为 0 (padding token)

    return result
