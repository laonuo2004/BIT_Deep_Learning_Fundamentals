<!-- Generated by SpecStory -->

# PyTorch Accelerator AttributeError Troubleshooting (2025-05-15 00:38:51)

_**User**_

当我运行到这一行代码时：

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
```

程序报错：

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[5], line 1
----> 1 device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
      2 print(f"Using {device} device")
      4 # Define model

File c:\Users\25912\.conda\envs\GNN\Lib\site-packages\torch\__init__.py:2562, in __getattr__(name)
   2559 if name in _lazy_modules:
   2560     return importlib.import_module(f".{name}", __name__)
-> 2562 raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

AttributeError: module 'torch' has no attribute 'accelerator'
```

请问这是怎么一回事？我又该要如何解决？

---

_**Assistant**_



---




---



---

您遇到的错误 `AttributeError: module 'torch' has no attribute 'accelerator'` 表明您当前使用的 PyTorch 版本中不存在 `torch.accelerator` 这个模块。这个 API 可能是较新版本的 PyTorch 才引入的，或者在您的特定安装中不可用。

**错误原因：**

`torch.accelerator` 是一个较新的用来简化设备选择的接口。如果您的 PyTorch 版本比较旧，或者某些特定编译选项没有启用，就可能找不到这个模块。

**如何解决：**

您可以使用一种更通用且向后兼容的方式来检测和选择可用的计算设备（如 GPU 或 CPU）。下面是一个常用的方法：

我会修改 `HW2/quickstart_tutorial.ipynb` 文件中的相关代码行。

```python
import torch

# 检查是否有可用的 CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    device = "cuda"
# 检查是否有可用的 MPS (Apple Silicon GPU)
elif torch.backends.mps.is_available():
    device = "mps"
# 如果都没有，则使用 CPU
else:
    device = "cpu"

print(f"Using {device} device")
```

我将为您修改文件 `HW2/quickstart_tutorial.ipynb` 中的代码。

---



---

代码修改似乎没有成功应用。我将尝试重新应用这个修改。

---

**Apply failed**

---

很抱歉，自动修改代码失败了。

您可以手动将 `HW2/quickstart_tutorial.ipynb` 文件中报错的那一行：

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
```

替换为以下代码：

```python
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

这段代码会首先检查 CUDA (NVIDIA GPU) 是否可用，如果不可用，则检查 MPS (Apple Silicon GPU) 是否可用，最后如果两者都不可用，则回退到使用 CPU。

这样修改后，您的代码应该就能正确选择可用的设备并继续运行了。如果您在修改后仍然遇到问题，请告诉我。

---

