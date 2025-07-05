import py2nb
import sys

# 检查是否提供了命令行参数
if len(sys.argv) != 2:
    print("使用方法: python convert_script.py <python文件路径>")
    sys.exit(1)

# 获取要转换的文件路径
python_file_path = sys.argv[1]

# Convert script to notebook
notebook_path = py2nb.convert(python_file_path)
print(f"转换完成！输出文件: {python_file_path}")