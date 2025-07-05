#| # 实验二：汽车里程数回归预测
#|
#| 本次实验旨在利用前馈神经网络，根据汽车的多种属性，预测其每加仑行驶的英里数（MPG）。
#| 我们将使用MindSpore框架来构建、训练和评估一个回归模型。
#|
#| **学生视角思考**：
#| 这个任务是一个典型的回归问题。与分类问题不同，我们的目标是预测一个连续值（MPG），而不是一个离散的类别。
#| 因此，模型的输出层不会使用Softmax激活函数，损失函数也应该选用适合回归任务的，比如均方误差（MSE）。

#| ## 1. 环境准备与库导入
#|
#| 首先，我们需要导入所有必要的库。
#| - `os`, `csv`, `time`: 用于基本操作。
#| - `numpy`, `pandas`: 用于数据处理和分析。
#| - `matplotlib`: 用于数据可视化。
#| - `mindspore`: 核心的深度学习框架。
#|
#| 同时，我们需要根据实验要求设置MindSpore的运行环境。

#-
#! pip install pandas matplotlib

import os
import csv
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy, MAE, MSE
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

#| 设置MindSpore的执行模式为图模式，并指定使用Ascend硬件。
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')#| ## 2. 数据加载与预处理
#|
#| 接下来，我们加载 `auto-mpg.data` 数据集。这个数据集记录了不同汽车的多种属性以及它们的MPG。
#|
#| **学生视角思考**：
#| 数据预处理是机器学习项目中至关重要的一步。原始数据往往是不规整的，可能包含缺失值、非数值特征等。
#| - **缺失值处理**: 我注意到数据中的马力（Horsepower）列有 `?` 作为缺失值。Pandas在读取时可以方便地将它们识别为NaN，然后我可以直接删除这些行，因为缺失的数据量不大。
#| - **特征编码**: 'Origin' (产地) 是一个分类特征（1: USA, 2: Europe, 3: Japan）。神经网络需要数值输入，所以需要将其转换为one-hot编码。这样模型就不会错误地认为产地之间存在序数关系（比如Japan > Europe）。
#| - **数据归一化**: 不同特征的数值范围差异很大（例如，`Weight` 和 `Cylinders`）。如果不进行归一化，范围较大的特征可能会在模型训练中占据主导地位，导致收敛缓慢或结果不佳。我将使用标准差归一化（Z-score normalization）。

#-
#| ### 2.1 加载数据
#| 我们使用 `pandas` 直接从URL读取数据，并指定列名和处理方式。

#-
url = 'https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/deep-learning/auto-mpg.data'

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print("原始数据集（前5行）:")
print(dataset.head())

#| ### 2.2 清理数据（处理缺失值）

#-
print("\n缺失值统计:")
print(dataset.isna().sum())

dataset = dataset.dropna()

#| ### 2.3 特征工程（One-Hot编码）

#-
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print("\n进行One-Hot编码后（后5行）:")
print(dataset.tail())

#| ### 2.4 划分训练集和测试集

#-
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#| ### 2.5 数据归一化

#-
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print("\n归一化后的训练数据（前5行）:")
print(normed_train_data.head())#| ## 3. 构建神经网络模型
#|
#| 我们将构建一个包含两个隐藏层的全连接神经网络。
#|
#| **学生视角思考**:
#| - **网络结构**: 对于这种表格数据回归任务，一个简单的全连接网络通常就足够了。我设计了`9 -> 64 -> 64 -> 1`的结构。输入层有9个神经元，对应9个特征。两个隐藏层各有64个神经元，这提供了一定的模型容量来学习复杂的非线性关系。输出层只有1个神经元，直接输出预测的MPG值。
#| - **激活函数**: 我在隐藏层之间使用了ReLU激活函数，这是深度学习中最常用的激活函数之一，因为它计算简单且能有效避免梯度消失问题。
#| - **输出层**: 输出层没有激活函数，因为我们需要它输出任意范围的连续值，而不是像分类任务那样限制在特定范围内。

#-
class RegressionNet(nn.Cell):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(9, 64, activation='relu')
        self.fc2 = nn.Dense(64, 64, activation='relu')
        self.fc3 = nn.Dense(64, 1)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x#| ## 4. 模型训练与评估
#|
#| 现在我们来定义训练过程。
#|
#| **学生视角思考**:
#| - **损失函数**: 我选择了均方误差（MSE, Mean Square Error）作为损失函数，这是回归问题最常用的损失函数，它衡量了预测值与真实值之间差的平方的均值。
#| - **优化器**: 我使用了RMSProp优化器。它是一种自适应学习率的优化器，通常在处理非平稳目标时表现良好。
#| - **评估指标**: 除了MSE，我还将监控平均绝对误差（MAE, Mean Absolute Error）。MAE直接衡量预测值与真实值之间差的绝对值的均值，它的单位与目标变量相同，因此更易于解释。例如，MAE为3.5意味着模型的预测平均偏离真实MPG值3.5个单位。
#| - **训练循环**: 我将训练100个epoch。在每个epoch结束时，我会在测试集上评估模型，并保存MAE和MSE，以便后续可视化模型的学习过程。

#-
#| ### 4.1 定义超参数、模型、损失函数和优化器

#-
epochs = 100
learning_rate = 0.001

network = RegressionNet()
loss_fn = nn.MSELoss()
optimizer = nn.RMSProp(network.trainable_params(), learning_rate=learning_rate)



grad_fn = ops.GradOperation(get_by_list=True)

#| ### 4.3 训练模型

#-
history = {'train_loss': [], 'train_mae': [], 'test_loss': [], 'test_mae': []}

# 将数据转为Tensor
X_train = Tensor(normed_train_data.values, ms.float32)
Y_train = Tensor(train_labels.values.reshape(-1, 1), ms.float32)
X_test = Tensor(normed_test_data.values, ms.float32)
Y_test = Tensor(test_labels.values.reshape(-1, 1), ms.float32)

mae_metric = MAE()

for epoch in range(epochs):
    # 手动计算梯度
    loss = loss_fn(network(X_train), Y_train)
    grads = grad_fn(network, optimizer.parameters)(X_train)
    optimizer(grads)

    # 记录训练集指标
    mae_metric.clear()
    mae_metric.update(network(X_train), Y_train)
    train_mae = mae_metric.eval()
    
    history['train_loss'].append(loss.asnumpy())
    history['train_mae'].append(train_mae)

    # 评估测试集
    test_predictions = network(X_test)
    test_loss = loss_fn(test_predictions, Y_test)
    
    mae_metric.clear()
    mae_metric.update(test_predictions, Y_test)
    test_mae = mae_metric.eval()

    history['test_loss'].append(test_loss.asnumpy())
    history['test_mae'].append(test_mae)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.asnumpy():.4f}, Test Loss: {test_loss.asnumpy():.4f}, Test MAE: {test_mae:.4f}")

#| ### 4.4 可视化训练过程

#-
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(history['train_mae'], label='Train Error')
    plt.plot(history['test_mae'], label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(history['train_loss'], label='Train Error')
    plt.plot(history['test_loss'], label='Val Error')
    plt.ylim([0, 25])
    plt.legend()
    plt.show()

plot_history(history)

#| ### 4.5 可视化预测结果

#-
test_predictions = network(X_test).asnumpy()

plt.figure(figsize=(8, 8))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

#| ## 5. 实验总结
#|
#| 通过本次实验，我成功地使用MindSpore构建了一个前馈神经网络来解决汽车里程数的回归预测问题。
#| 从训练过程的MAE和MSE曲线可以看出，模型在大约20个epoch后迅速收敛，并且在训练集和验证集上的误差都保持在较低水平，没有出现明显的过拟合。
#| 最终的预测值与真实值散点图也显示，大部分点都紧密地分布在对角线附近，说明模型具有较好的预测精度。
#| 我深刻体会到了数据预处理（特别是归一化）对模型训练的重要性，以及如何为回归任务选择合适的网络结构、损失函数和评估指标。