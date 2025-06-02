import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="clip", # 项目名称，用于在 PyPI 等地方标识
    py_modules=["clip"], # 指定包含的顶级 Python 模块。这里指定了名为 "clip" 的模块，意味着项目中有一个 clip.py 文件或者一个名为 clip 的目录且包含 __init__.py。
    version="1.0", 
    description="", 
    author="OpenAI", 
    packages=find_packages(exclude=["tests*"]), # 自动查找项目中的所有包，并排除 "tests" 相关的包。"包" 是包含 __init__.py 文件的目录。
    install_requires=[ # 项目的依赖列表
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt")) # 从 requirements.txt 文件中读取依赖项
        )
    ],
    include_package_data=True, # 指示 setuptools 包含 MANIFEST.in 文件中指定的数据文件
    extras_require={'dev': ['pytest']}, # 定义额外的依赖项组，例如用于开发的依赖 ('dev' 组，包含 pytest)
)
