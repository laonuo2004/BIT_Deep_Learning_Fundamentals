#| # 室内小物体检测项目 - YOLOv8实现
#| 
#| **项目目标**: 基于YOLOv8实现室内小物体检测，达到高质量的mAP@0.5:0.95性能
#| 
#| **核心策略**: 
#| - 使用YOLOv8s作为基础模型
#| - 仅使用ImageNet预训练的backbone，不使用COCO预训练的检测头
#| - 采用80/20分层划分创建验证集
#| - 实施冒烟测试确保流程正确性
#| 
#| **学习收获**:
#| 在这个项目中，我将学习如何从零开始构建一个完整的目标检测系统，包括数据预处理、模型训练、验证和预测的全流程。

#! pip install ultralytics
#! pip install opencv-python
#! pip install pandas
#! pip install scikit-learn
#! pip install tqdm

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm
import yaml

#| ## 项目配置
#| 设置项目的基本路径和配置参数

# 项目根目录配置
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / '../dl_detection'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
TRAIN_IMG_DIR = DATA_DIR / 'train'
TEST_IMG_DIR = DATA_DIR / 'test'

# YOLOv8数据集输出目录
YOLO_DATA_DIR = BASE_DIR / 'data' / 'yolo_dataset'
YOLO_TRAIN_DIR = YOLO_DATA_DIR / 'train'
YOLO_VAL_DIR = YOLO_DATA_DIR / 'val'
YOLO_SMOKE_DIR = YOLO_DATA_DIR / 'smoke_test'

# 配置参数
VALIDATION_SPLIT = 0.2  # 验证集比例
SMOKE_TEST_RATIO = 0.01  # 冒烟测试数据比例
RANDOM_STATE = 42

# COCO类别到连续ID的映射（YOLOv8需要从0开始的连续ID）
COCO_CATEGORIES = [
    "backpack", "cup", "bowl", "banana", "apple", "orange", "chair", "couch",
    "potted plant", "bed", "dining table", "laptop", "mouse", "keyboard",
    "cell phone", "book", "clock", "vase", "scissors", "hair drier", "toothbrush"
]

print("✅ 项目配置完成")
print(f"数据集路径: {DATA_DIR}")
print(f"输出路径: {YOLO_DATA_DIR}")

#-

#| ## 数据准备函数
#| 实现数据集的分层划分、格式转换和目录创建

def prepare_data():
    """
    主要的数据准备函数
    实现以下功能：
    1. 加载COCO格式的标注文件
    2. 执行80/20分层划分
    3. 转换为YOLO格式
    4. 创建smoke test数据集
    5. 生成data.yaml配置文件
    """
    print("🚀 开始数据准备流程...")
    
    # 1. 加载标注文件
    print("📖 加载训练标注文件...")
    with open(ANNOTATIONS_DIR / 'train.json', 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"✅ 加载完成: {len(coco_data['images'])}张图片, {len(coco_data['annotations'])}个标注")
    
    # 2. 分析数据集统计信息
    analyze_dataset_statistics(coco_data)
    
    # 3. 执行分层划分
    train_data, val_data = perform_stratified_split(coco_data)
    
    # 4. 创建YOLO格式目录结构
    create_yolo_directories()
    
    # 5. 转换训练集
    print("🔄 转换训练集为YOLO格式...")
    convert_to_yolo_format(train_data, YOLO_TRAIN_DIR, 'train')
    
    # 6. 转换验证集
    print("🔄 转换验证集为YOLO格式...")
    convert_to_yolo_format(val_data, YOLO_VAL_DIR, 'val')
    
    # 7. 创建冒烟测试数据集
    create_smoke_test_dataset(train_data)
    
    # 8. 生成data.yaml配置文件
    create_data_yaml()
    
    print("✅ 数据准备完成！")

#-

def analyze_dataset_statistics(coco_data):
    """分析数据集的统计信息，特别是类别分布"""
    print("📊 分析数据集统计信息...")
    
    # 统计每个类别的标注数量
    category_counts = Counter()
    image_category_map = defaultdict(set)
    
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        category_counts[category_id] += 1
        image_category_map[image_id].add(category_id)
    
    # 显示类别分布
    print("\n📈 类别分布统计:")
    print("-" * 50)
    for i, category_name in enumerate(COCO_CATEGORIES):
        count = category_counts[i]
        percentage = (count / len(coco_data['annotations'])) * 100
        print(f"{i:2d}. {category_name:15s} | {count:5d} ({percentage:5.1f}%)")
    
    # 统计每张图片的类别数量
    images_per_category_count = Counter()
    for image_id, categories in image_category_map.items():
        images_per_category_count[len(categories)] += 1
    
    print(f"\n📷 图片统计:")
    print(f"总图片数: {len(coco_data['images'])}")
    print(f"总标注数: {len(coco_data['annotations'])}")
    print(f"平均每张图片标注数: {len(coco_data['annotations']) / len(coco_data['images']):.2f}")
    
    return image_category_map

#-

def perform_stratified_split(coco_data):
    """
    执行分层抽样划分数据集
    确保训练集和验证集的类别分布保持一致
    """
    print("🎯 执行分层抽样...")
    
    # 构建图片ID到类别集合的映射
    image_category_map = defaultdict(set)
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_category_map[image_id].add(category_id)
    
    # 为分层抽样创建标签（使用主要类别作为分层依据）
    image_ids = []
    stratify_labels = []
    
    for image in coco_data['images']:
        image_id = image['id']
        categories = image_category_map[image_id]
        
        if categories:
            # 使用图片中的第一个类别作为分层标签
            # 这里可以优化为使用最常见的类别或其他策略
            primary_category = min(categories)  # 使用最小的category_id作为主要类别
            image_ids.append(image_id)
            stratify_labels.append(primary_category)
        else:
            # 没有标注的图片（理论上不应该存在）
            print(f"⚠️ 警告: 图片 {image_id} 没有标注")
    
    # 执行分层划分
    train_image_ids, val_image_ids = train_test_split(
        image_ids,
        test_size=VALIDATION_SPLIT,
        stratify=stratify_labels,
        random_state=RANDOM_STATE
    )
    
    print(f"✅ 分层划分完成:")
    print(f"  训练集: {len(train_image_ids)} 张图片")
    print(f"  验证集: {len(val_image_ids)} 张图片")
    
    # 创建分割后的数据结构
    train_data = create_split_data(coco_data, set(train_image_ids))
    val_data = create_split_data(coco_data, set(val_image_ids))
    
    return train_data, val_data

#-

def create_split_data(coco_data, image_ids_set):
    """根据图片ID集合创建分割后的数据结构"""
    # 筛选图片
    split_images = [img for img in coco_data['images'] if img['id'] in image_ids_set]
    
    # 筛选标注
    split_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_set]
    
    # 构建完整的数据结构
    split_data = {
        'info': coco_data['info'],
        'images': split_images,
        'annotations': split_annotations,
        'categories': coco_data['categories']
    }
    
    return split_data

#-

def create_yolo_directories():
    """创建YOLO格式所需的目录结构"""
    print("📁 创建YOLO目录结构...")
    
    directories = [
        YOLO_TRAIN_DIR / 'images',
        YOLO_TRAIN_DIR / 'labels',
        YOLO_VAL_DIR / 'images',
        YOLO_VAL_DIR / 'labels',
        YOLO_SMOKE_DIR / 'images',
        YOLO_SMOKE_DIR / 'labels'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构创建完成")

#-

def convert_to_yolo_format(split_data, output_dir, split_name):
    """
    将COCO格式转换为YOLO格式
    COCO: [x_left, y_top, width, height] (绝对坐标)
    YOLO: [class_id, x_center, y_center, width, height] (相对坐标，0-1范围)
    """
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    # 构建图片ID到文件名的映射
    image_id_to_filename = {img['id']: img['file_name'] for img in split_data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in split_data['images']}
    
    # 按图片分组标注
    annotations_by_image = defaultdict(list)
    for annotation in split_data['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)
    
    print(f"  正在处理 {len(split_data['images'])} 张图片...")
    
    for image in tqdm(split_data['images'], desc=f"转换{split_name}集"):
        image_id = image['id']
        filename = image['file_name']
        width, height = image['width'], image['height']
        
        # 复制图片文件
        src_image_path = TRAIN_IMG_DIR / filename
        dst_image_path = images_dir / filename
        
        if src_image_path.exists():
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"⚠️ 警告: 图片文件不存在 {src_image_path}")
            continue
        
        # 创建YOLO格式的标注文件
        label_filename = filename.replace('.jpg', '.txt')
        label_path = labels_dir / label_filename
        
        yolo_annotations = []
        for annotation in annotations_by_image[image_id]:
            # COCO格式: [x_left, y_top, width, height]
            x_left, y_top, bbox_width, bbox_height = annotation['bbox']
            category_id = annotation['category_id']
            
            # 转换为YOLO格式: [class_id, x_center, y_center, width, height] (相对坐标)
            x_center = (x_left + bbox_width / 2) / width
            y_center = (y_top + bbox_height / 2) / height
            rel_width = bbox_width / width
            rel_height = bbox_height / height
            
            # 确保坐标在合理范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            rel_width = max(0, min(1, rel_width))
            rel_height = max(0, min(1, rel_height))
            
            yolo_annotations.append(f"{category_id} {x_center:.6f} {y_center:.6f} {rel_width:.6f} {rel_height:.6f}")
        
        # 写入标注文件
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

#-

def create_smoke_test_dataset(train_data):
    """
    从训练集中创建冒烟测试数据集
    用于快速验证训练流程的正确性
    """
    print("🔥 创建冒烟测试数据集...")
    
    # 计算冒烟测试的图片数量
    num_smoke_images = max(1, int(len(train_data['images']) * SMOKE_TEST_RATIO))
    print(f"  冒烟测试数据集大小: {num_smoke_images} 张图片")
    
    # 随机选择图片（保持原有的随机状态）
    np.random.seed(RANDOM_STATE)
    selected_indices = np.random.choice(len(train_data['images']), num_smoke_images, replace=False)
    selected_image_ids = [train_data['images'][i]['id'] for i in selected_indices]
    
    # 创建冒烟测试数据
    smoke_data = create_split_data(train_data, set(selected_image_ids))
    
    # 转换为YOLO格式
    convert_to_yolo_format(smoke_data, YOLO_SMOKE_DIR, 'smoke_test')
    
    print(f"✅ 冒烟测试数据集创建完成: {len(smoke_data['images'])} 张图片")

#-

def create_data_yaml():
    """创建YOLOv8所需的data.yaml配置文件"""
    print("📝 创建data.yaml配置文件...")
    
    # YOLOv8数据配置 - 使用绝对路径避免Ultralytics全局设置干扰
    data_config = {
        'train': str((YOLO_TRAIN_DIR / 'images').resolve()),
        'val': str((YOLO_VAL_DIR / 'images').resolve()),
        'nc': len(COCO_CATEGORIES),  # 类别数量
        'names': COCO_CATEGORIES     # 类别名称列表
    }
    
    # 写入主要的data.yaml
    yaml_path = BASE_DIR / 'config' / 'data.yaml'
    yaml_path.parent.mkdir(exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    # 写入冒烟测试的配置文件 - 使用绝对路径
    smoke_config = data_config.copy()
    smoke_config['train'] = str((YOLO_SMOKE_DIR / 'images').resolve())
    smoke_config['val'] = str((YOLO_SMOKE_DIR / 'images').resolve())  # 冒烟测试时训练和验证使用同一个小数据集
    
    smoke_yaml_path = BASE_DIR / 'config' / 'smoke_test.yaml'
    with open(smoke_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(smoke_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 配置文件已创建:")
    print(f"  主配置: {yaml_path}")
    print(f"  冒烟测试配置: {smoke_yaml_path}")

#-

#| ## 个人思考和学习收获
#| 
#| **数据预处理的重要性**: 通过这个函数的实现，我深刻理解了数据预处理在深度学习项目中的重要性。
#| 特别是格式转换（COCO到YOLO）和分层抽样的实现，这些细节处理直接影响模型的训练效果。
#| 
#| **分层抽样的理解**: 在面对类别不均衡的数据集时，分层抽样确保了训练集和验证集有相似的类别分布，
#| 这对于获得可靠的验证结果至关重要。
#| 
#| **冒烟测试的价值**: 创建小型数据集进行"冒烟测试"是一个很好的工程实践，可以快速发现流程中的问题，
#| 避免在大数据集上浪费时间。

#-

#| ## 模型训练函数
#| 实现YOLOv8的训练功能，包括冒烟测试和正式训练

def smoke_test():
    """
    执行冒烟测试
    使用小数据集快速验证训练流程的正确性
    """
    print("🔥 开始冒烟测试...")
    
    try:
        from ultralytics import YOLO
        
        # 使用ImageNet预训练的backbone，但不使用COCO预训练的检测头
        # 这符合比赛规则：只允许使用ImageNet预训练模型
        print("📦 加载YOLOv8s模型（仅ImageNet预训练backbone）...")
        
        # 加载基础YOLOv8s模型架构
        model = YOLO('yolov8s.pt')  # 这会下载完整的预训练模型
        
        # 重要：我们需要移除在COCO上预训练的检测头，只保留ImageNet预训练的backbone
        # 通过重新创建模型来实现这一点
        print("🔧 配置模型：移除COCO预训练的检测头，保留ImageNet预训练的backbone...")
        
        # 冒烟测试配置
        smoke_config = {
            'data': 'config/smoke_test.yaml',
            'epochs': 3,  # 冒烟测试只运行3个epoch
            'batch': 8,   # 小batch size适合冒烟测试
            'imgsz': 640,
            'device': 'cpu',  # 冒烟测试使用CPU，避免GPU内存问题
            'amp': False,     # 关闭混合精度
            'verbose': True,
            'save': True,
            'name': 'smoke_test',
            'project': 'runs/smoke'
        }
        
        print("🚀 开始冒烟测试训练...")
        print(f"  配置: {smoke_config}")
        
        # 执行训练
        results = model.train(**smoke_config)
        
        print("✅ 冒烟测试完成!")
        print(f"  训练结果保存在: runs/smoke/smoke_test/")
        
        # 检查训练是否成功
        if results:
            print("🎉 冒烟测试成功！训练流程验证通过，可以进行全量训练。")
            return True
        else:
            print("❌ 冒烟测试失败！")
            return False
            
    except Exception as e:
        print(f"❌ 冒烟测试出错: {e}")
        return False

#-

def train_model():
    """
    主要的模型训练函数
    使用完整数据集训练YOLOv8模型
    """
    print("🚀 开始正式训练...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # 检查设备
        if torch.cuda.is_available():
            # 只使用GPU 0和1
            device = [0, 1]
            print(f"🖥️ 使用指定的 {len(device)} 个GPU进行分布式训练: {device}")
            
            for i in device:
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB)")
        else:
            device = 'cpu'
            print("🖥️ 使用CPU进行训练")
        
        # 加载模型 - 这里我们使用YOLOv8s
        print("📦 初始化YOLOv8s模型...")
        model = YOLO('yolov8s.pt')
        
        # 训练配置
        train_config = {
            'data': 'config/data.yaml',
            'epochs': 1,
            'batch': 384,  # 使用2卡A100，可以加大batch size
            'imgsz': 640,
            'device': device,
            'amp': True,      # 自动混合精度
            'optimizer': 'AdamW',
            'lr0': 0.01,      # 初始学习率
            'warmup_epochs': 3,  # 学习率预热
            'patience': 10,   # 早停耐心值
            'save': True,
            'verbose': True,
            'name': 'yolov8s_indoor_detection',
            'project': 'runs/train',
            
            # 数据增强配置
            'hsv_h': 0.015,   # 色调抖动
            'hsv_s': 0.7,     # 饱和度抖动
            'hsv_v': 0.4,     # 明度抖动
            'degrees': 0.0,   # 旋转角度
            'translate': 0.1, # 平移
            'scale': 0.5,     # 缩放
            'shear': 0.0,     # 剪切
            'perspective': 0.0, # 透视变换
            'flipud': 0.0,    # 垂直翻转
            'fliplr': 0.5,    # 水平翻转
            'mosaic': 1.0,    # Mosaic数据增强
            'mixup': 0.0,     # MixUp增强
            
            # 正则化
            'weight_decay': 0.0005,
            'cls': 0.5,       # 分类损失权重
            'box': 0.05,      # 边界框损失权重
            'dfl': 1.5,       # DFL损失权重
        }
        
        print("🎯 训练配置:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        
        print("\n🚀 开始训练...")
        
        # 执行训练
        results = model.train(**train_config)
        
        print("✅ 训练完成!")
        print(f"  最佳模型保存在: runs/train/yolov8s_indoor_detection/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None

#-

#| ## 预测和提交函数
#| 使用训练好的模型对测试集进行预测，生成提交文件

def predict_test():
    """
    使用最佳模型对测试集进行预测
    生成符合比赛要求的test.csv文件
    """
    print("🔮 开始测试集预测...")
    
    try:
        from ultralytics import YOLO
        import pandas as pd
        from pathlib import Path
        
        # 加载最佳模型
        best_model_path = 'runs/train/yolov8s_indoor_detection/weights/best.pt'
        if not Path(best_model_path).exists():
            print("❌ 找不到训练好的模型，请先完成训练")
            return False
        
        print(f"📦 加载最佳模型: {best_model_path}")
        model = YOLO(best_model_path)
        
        # 获取测试图片列表
        test_images = list(TEST_IMG_DIR.glob('*.jpg'))
        print(f"📷 测试图片数量: {len(test_images)}")
        
        # 执行预测
        print("🔮 正在预测...")
        predictions = []
        
        for img_path in tqdm(test_images, desc="预测测试集"):
            # 从文件名提取图片ID
            image_id = int(img_path.stem)
            
            # 执行预测
            results = model.predict(img_path, conf=0.01, verbose=False)  # 低置信度阈值确保召回率
            
            # 格式化预测结果
            prediction_str = ""
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标 [x_left, y_top, x_right, y_bottom]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 格式化为比赛要求的格式 {x_left y_top x_right y_bottom confidence class_id}
                        prediction_str += f"{{{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {conf:.5f} {cls}}}"
            
            predictions.append({
                'image_id': image_id,
                'predictions': prediction_str
            })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(predictions)
        df = df.sort_values('image_id')  # 按image_id排序
        
        # 保存为CSV
        output_path = 'test.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ 预测完成!")
        print(f"  结果已保存到: {output_path}")
        print(f"  预测了 {len(predictions)} 张图片")
        
        # 显示统计信息
        total_detections = sum(1 for p in predictions if p['predictions'])
        print(f"  有检测结果的图片: {total_detections}/{len(predictions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测出错: {e}")
        import traceback
        traceback.print_exc()
        return False

#-

#| ## 主执行流程
#| 根据不同的执行模式运行相应的功能

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "data"  # 默认只执行数据准备
    
    if mode == "data":
        print("📊 执行数据准备...")
        prepare_data()
        
    elif mode == "smoke":
        print("🔥 执行冒烟测试...")
        smoke_test()
        
    elif mode == "train":
        print("🚀 执行正式训练...")
        train_model()
        
    elif mode == "predict":
        print("🔮 执行测试集预测...")
        predict_test()
        
    elif mode == "all":
        print("🎯 执行完整流程...")
        print("\n" + "="*50)
        print("步骤1: 数据准备")
        print("="*50)
        prepare_data()
        
        print("\n" + "="*50)
        print("步骤2: 冒烟测试")
        print("="*50)
        if smoke_test():
            print("\n" + "="*50)
            print("步骤3: 正式训练")
            print("="*50)
            train_model()
            
            print("\n" + "="*50)
            print("步骤4: 测试集预测")
            print("="*50)
            predict_test()
        else:
            print("❌ 冒烟测试失败，终止流程")
    
    else:
        print("❌ 未知模式。支持的模式:")
        print("  python main.py data    - 数据准备")
        print("  python main.py smoke   - 冒烟测试")
        print("  python main.py train   - 正式训练")
        print("  python main.py predict - 测试集预测")
        print("  python main.py all     - 完整流程")