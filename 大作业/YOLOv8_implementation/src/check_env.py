"""
环境检查脚本
用于验证训练所需的核心依赖是否正确安装
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version
    print(f"  Python版本: {version}")
    
    # 检查是否为推荐的Python版本
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print("  ✅ Python版本符合要求")
        return True
    else:
        print("  ⚠️ 建议使用Python 3.8+")
        return False

def check_package(package_name, import_name=None, version_attr='__version__'):
    """检查单个包的安装情况"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"  ✅ {package_name}: v{version}")
        else:
            print(f"  ✅ {package_name}: 已安装")
        return True
    except ImportError:
        print(f"  ❌ {package_name}: 未安装")
        return False

def check_pytorch_cuda():
    """检查PyTorch和CUDA支持"""
    print("\n🔥 检查PyTorch和CUDA...")
    
    # 检查PyTorch
    pytorch_ok = check_package('torch', 'torch')
    if not pytorch_ok:
        return False
    
    # 检查CUDA支持
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"  ✅ CUDA可用: v{cuda_version}")
            print(f"  ✅ GPU数量: {device_count}")
            
            # 显示每个GPU的信息
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("  ⚠️ CUDA不可用，将使用CPU训练")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch检查失败: {e}")
        return False

def check_core_packages():
    """检查核心依赖包"""
    print("\n📦 检查核心依赖包...")
    
    packages = [
        ('ultralytics', 'ultralytics'),
        ('opencv-python', 'cv2'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('tqdm', 'tqdm'),
        ('pyyaml', 'yaml'),
        ('matplotlib', 'matplotlib'),
        ('pillow', 'PIL')
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        ok = check_package(package_name, import_name)
        all_ok = all_ok and ok
    
    return all_ok

def check_ultralytics():
    """检查Ultralytics YOLO的具体功能"""
    print("\n🚀 检查Ultralytics YOLO...")
    
    try:
        from ultralytics import YOLO
        print("  ✅ YOLO类导入成功")
        
        # 检查是否可以创建模型实例
        try:
            model = YOLO('yolov8n.pt')  # 使用最小的模型进行测试
            print("  ✅ YOLOv8模型加载成功")
            return True
        except Exception as e:
            print(f"  ⚠️ YOLOv8模型加载失败: {e}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Ultralytics导入失败: {e}")
        return False

def check_data_paths():
    """检查数据路径是否存在"""
    print("\n📁 检查数据路径...")
    
    base_dir = Path('.')
    data_dir = base_dir / '../dl_detection'
    
    paths_to_check = [
        (data_dir, "数据集根目录"),
        (data_dir / 'annotations' / 'train.json', "训练标注文件"),
        (data_dir / 'train', "训练图片目录"),
        (data_dir / 'test', "测试图片目录")
    ]
    
    all_exist = True
    for path, description in paths_to_check:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✅ {description}: 存在 ({size:.1f}MB)")
            else:
                try:
                    count = len(list(path.glob('*')))
                    print(f"  ✅ {description}: 存在 ({count}个文件)")
                except:
                    print(f"  ✅ {description}: 存在")
        else:
            print(f"  ❌ {description}: 不存在 - {path}")
            all_exist = False
    
    return all_exist

def check_disk_space():
    """检查磁盘空间"""
    print("\n💾 检查磁盘空间...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        
        total_gb = total / 1024**3
        used_gb = used / 1024**3
        free_gb = free / 1024**3
        
        print(f"  总空间: {total_gb:.1f}GB")
        print(f"  已使用: {used_gb:.1f}GB")
        print(f"  可用空间: {free_gb:.1f}GB")
        
        if free_gb > 10:  # 需要至少10GB空间
            print("  ✅ 磁盘空间充足")
            return True
        else:
            print("  ⚠️ 磁盘空间可能不足，建议至少10GB可用空间")
            return False
            
    except Exception as e:
        print(f"  ⚠️ 无法检查磁盘空间: {e}")
        return True

def main():
    """主检查函数"""
    print("=" * 60)
    print("🔍 深度学习环境检查")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("PyTorch和CUDA", check_pytorch_cuda),
        ("核心依赖包", check_core_packages),
        ("Ultralytics YOLO", check_ultralytics),
        ("数据路径", check_data_paths),
        ("磁盘空间", check_disk_space)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name}检查失败: {e}")
            results.append((check_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 检查结果总结")
    print("=" * 60)
    
    passed = 0
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过: {passed}/{len(results)} 项检查")
    
    if passed == len(results):
        print("🎉 环境检查全部通过，可以开始训练！")
        return True
    else:
        print("⚠️ 部分检查未通过，请根据上述信息修复环境")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)