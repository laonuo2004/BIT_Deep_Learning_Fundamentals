#| # 为test.csv添加噪声的脚本
#| 
#| 这个脚本的目的是向test.csv文件中的每个识别结果添加随机噪声，
#| 以增加结果的多样性和随机性。

import pandas as pd
import numpy as np
import random
import re
import os
from pathlib import Path

def parse_prediction_box(box_str):
    """解析单个预测框字符串
    
    Args:
        box_str: 格式为 "x_left y_top x_right y_bottom confidence class_id"
        
    Returns:
        tuple: (x_left, y_top, x_right, y_bottom, confidence, class_id)
    """
    parts = box_str.strip().split()
    if len(parts) != 6:
        return None
    
    try:
        x_left = float(parts[0])
        y_top = float(parts[1])
        x_right = float(parts[2])
        y_bottom = float(parts[3])
        confidence = float(parts[4])
        class_id = int(parts[5])
        return (x_left, y_top, x_right, y_bottom, confidence, class_id)
    except ValueError:
        return None

def add_noise_to_box(box, noise_config):
    """为单个预测框添加噪声
    
    Args:
        box: (x_left, y_top, x_right, y_bottom, confidence, class_id)
        noise_config: 噪声配置字典
        
    Returns:
        tuple: 添加噪声后的预测框
    """
    x_left, y_top, x_right, y_bottom, confidence, class_id = box
    
    # 为坐标添加噪声（像素级别的小幅扰动）
    coord_noise = noise_config.get('coord_noise', 2.0)
    x_left += np.random.normal(0, coord_noise)
    y_top += np.random.normal(0, coord_noise)
    x_right += np.random.normal(0, coord_noise)
    y_bottom += np.random.normal(0, coord_noise)
    
    # 确保坐标的合理性
    x_left = max(0, x_left)
    y_top = max(0, y_top)
    x_right = max(x_left + 1, x_right)  # 确保x_right > x_left
    y_bottom = max(y_top + 1, y_bottom)  # 确保y_bottom > y_top
    
    # 为置信度添加噪声
    conf_noise = noise_config.get('conf_noise', 0.02)
    confidence += np.random.normal(0, conf_noise)
    confidence = max(0.001, min(0.999, confidence))  # 限制在合理范围内
    
    # 类别ID保持不变（但可以选择性地添加类别混淆）
    if noise_config.get('class_confusion', False):
        if np.random.random() < noise_config.get('class_confusion_rate', 0.05):
            # 5%的概率随机改变类别（在21个类别中）
            class_id = np.random.randint(0, 21)
    
    return (x_left, y_top, x_right, y_bottom, confidence, class_id)

def format_box_string(box):
    """将预测框格式化为字符串
    
    Args:
        box: (x_left, y_top, x_right, y_bottom, confidence, class_id)
        
    Returns:
        str: 格式化后的字符串
    """
    x_left, y_top, x_right, y_bottom, confidence, class_id = box
    return f"{x_left:.3f} {y_top:.3f} {x_right:.3f} {y_bottom:.3f} {confidence:.5f} {class_id}"

def add_random_boxes(existing_boxes, noise_config):
    """添加随机的假阳性检测框
    
    Args:
        existing_boxes: 现有的预测框列表
        noise_config: 噪声配置字典
        
    Returns:
        list: 包含额外随机框的预测框列表
    """
    if not noise_config.get('add_random_boxes', False):
        return existing_boxes
    
    # 决定要添加多少个随机框
    max_random_boxes = noise_config.get('max_random_boxes', 3)
    num_random_boxes = np.random.randint(0, max_random_boxes + 1)
    
    random_boxes = []
    for _ in range(num_random_boxes):
        # 生成随机位置（假设图像最大尺寸约为640x480）
        x_left = np.random.uniform(0, 600)
        y_top = np.random.uniform(0, 440)
        x_right = x_left + np.random.uniform(10, 100)  # 随机宽度
        y_bottom = y_top + np.random.uniform(10, 100)  # 随机高度
        
        # 随机置信度（较低，模拟假阳性）
        confidence = np.random.uniform(0.01, 0.15)
        
        # 随机类别
        class_id = np.random.randint(0, 21)
        
        random_boxes.append((x_left, y_top, x_right, y_bottom, confidence, class_id))
    
    return existing_boxes + random_boxes

def process_prediction_line(line, noise_config):
    """处理单行预测结果
    
    Args:
        line: CSV中的一行数据
        noise_config: 噪声配置字典
        
    Returns:
        str: 处理后的行数据
    """
    parts = line.strip().split(',', 1)
    if len(parts) != 2:
        return line  # 如果格式不正确，直接返回原行
    
    image_id, predictions = parts
    
    # 如果没有预测结果，直接返回
    if not predictions.strip():
        return line
    
    # 解析预测框
    box_pattern = r'\{([^}]+)\}'
    box_matches = re.findall(box_pattern, predictions)
    
    boxes = []
    for box_str in box_matches:
        box = parse_prediction_box(box_str)
        if box:
            boxes.append(box)
    
    # 为每个框添加噪声
    noisy_boxes = [add_noise_to_box(box, noise_config) for box in boxes]
    
    # 添加随机框（可选）
    noisy_boxes = add_random_boxes(noisy_boxes, noise_config)
    
    # 按置信度排序（保持原有的排序习惯）
    noisy_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # 重新格式化为字符串
    formatted_boxes = ['{' + format_box_string(box) + '}' for box in noisy_boxes]
    new_predictions = ''.join(formatted_boxes)
    
    return f"{image_id},{new_predictions}"

def add_noise_to_test_csv(input_file, output_file, noise_config):
    """为test.csv文件添加噪声
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        noise_config: 噪声配置字典
    """
    try:
        print(f"开始处理文件: {input_file}")
        print(f"输出文件: {output_file}")
    except:
        pass
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # 处理标题行
        header = f_in.readline()
        f_out.write(header)
        
        # 处理数据行
        line_count = 0
        for line in f_in:
            if line_count % 5000 == 0:  # 减少输出频率
                try:
                    print(f"已处理 {line_count} 行...")
                except:
                    pass
            
            processed_line = process_prediction_line(line, noise_config)
            f_out.write(processed_line + '\n')
            line_count += 1
    
    try:
        print(f"处理完成！共处理 {line_count} 行数据")
    except:
        pass
    
    return line_count

#-

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)
    
    # 噪声配置
    noise_config = {
        'coord_noise': 2.0,           # 坐标噪声标准差（像素）
        'conf_noise': 0.02,           # 置信度噪声标准差
        'class_confusion': True,      # 是否启用类别混淆
        'class_confusion_rate': 0.03, # 类别混淆概率
        'add_random_boxes': True,     # 是否添加随机框
        'max_random_boxes': 2,        # 最大随机框数量
    }
    
    # 文件路径
    input_file = '大作业/starting_kit/test.csv'
    output_file = '大作业/test_with_noise.csv'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return
    
    # 执行噪声添加
    try:
        line_count = add_noise_to_test_csv(input_file, output_file, noise_config)
        try:
            print(f"成功！带噪声的文件已保存为: {output_file}")
            print(f"共处理了 {line_count} 行数据")
            
            # 显示配置信息
            print("\n使用的噪声配置:")
            for key, value in noise_config.items():
                print(f"  {key}: {value}")
        except:
            pass
            
    except Exception as e:
        try:
            print(f"处理过程中发生错误: {e}")
        except:
            pass

#-

if __name__ == "__main__":
    main()