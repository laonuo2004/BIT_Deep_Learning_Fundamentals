from .clip import * # 从同级目录的 clip.py 模块中导入所有内容，方便外部调用
# 这种写法常用于包的初始化文件，使得用户可以直接通过 `import clip` 后使用 `clip.load()` 等，而无需 `import clip.clip`
