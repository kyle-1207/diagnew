# 导入数据集处理模块
from create_dataset import series_to_supervised
# 导入预处理模块
from sklearn import preprocessing
# 导入深度学习框架
import torch
import torch.nn as nn
import torch.optim as optim
# 导入数据集
#from sklearn.datasets import load_boston
# 导入评估指标
from sklearn import metrics
# 导入数据集划分
from sklearn.model_selection import train_test_split
# 导入scipy.io用于加载mat文件
import scipy.io as scio
# 导入torch.utils.data模块
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# 导入torch.nn.functional模块
import torch.nn.functional as F
# 导入torch.nn模块
from torch import nn
# 导入torchvision.transforms用于数据增强
from torchvision import transforms as tfs

INPUT_SIZE = 7
# 定义数据集类
class MyDataset(torch.utils.data.Dataset):
    # 初始化数据集
    def __init__(self, data, target):
        # 存储特征数据
        self.data = data
        # 存储目标数据
        self.target = target

    # 获取数据集中的单个样本
    def __getitem__(self, index):
        # 获取特征数据
        x = self.data[index]
        # 获取目标数据
        y = self.target[index]
        return x, y

    # 获取数据集长度
    def __len__(self):
        return len(self.data)

# 定义LSTM模型
class LSTM(nn.Module):
    # 初始化LSTM模型
    def __init__(self):
        super(LSTM, self).__init__()
        # 定义LSTM层
        self.lstm = nn.LSTM(
            # LSTM输入特征维度
            input_size=INPUT_SIZE,
            # LSTM隐藏层维度
            hidden_size=18,  # rnn hidden unit
            # LSTM层数
            num_layers=1,  # number of rnn layer
            # 输入和输出是否包含批次维度
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # 是否为双向LSTM
            bidirectional=True
        )
        # 定义输出层
        self.out = nn.Linear(36, 2)
    # 前向传播
    def forward(self, x):
        # 获取LSTM的输出和隐状态
        r_out, (hidden_state1, hidden_state2) = self.lstm(x, None) 
        # 存储每个时间步的输出
        outs = []  
        # 遍历LSTM的输出，获取每个时间步的预测结果
        for time_step in range(r_out.size(1)):  
            outs.append(self.out(r_out[:, time_step, :]))
        # 将所有时间步的预测结果堆叠成一个张量
        return torch.stack(outs, dim=1)
    
# 定义数据集类
class Dataset(Dataset): 
    # 初始化数据集
    def __init__(self, x, y, z, q, w):
        # 将输入数据转换为torch张量，并指定数据类型为double
        self.x = torch.from_numpy(x).to(torch.double)
        # 将目标数据转换为torch张量，并指定数据类型为double
        self.y = torch.from_numpy(y).to(torch.double)
        # 将特征数据转换为torch张量，并指定数据类型为double
        self.z = torch.from_numpy(z).to(torch.double)
        # 将额外特征数据转换为torch张量，并指定数据类型为double
        self.q = torch.from_numpy(q).to(torch.double)
        # 将权重数据转换为torch张量，并指定数据类型为double
        self.w = torch.from_numpy(w).to(torch.double)
    # 获取数据集长度
    def __len__(self):
        return len(self.x)
    # 获取数据集中的单个样本
    def __getitem__(self, idx):  
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx], self.w[idx]
    

# 定义组合自编码器模型
class CombinedAE(nn.Module):
    # 初始化组合自编码器
    def __init__(self, input_size, encode2_input_size, output_size, activation_fn, use_dx_in_forward):
        super(CombinedAE, self).__init__()
        # 定义第一层全连接层
        self.fc1 = nn.Linear(input_size, 1)
        # 定义第二层全连接层
        self.fc2 = nn.Linear(encode2_input_size, 1)
        # 定义解码层
        self.fc3 = nn.Linear(output_size, output_size)
        # 存储激活函数
        self.activation_fn = activation_fn
        # 是否在正向传播中使用dx
        self.use_dx_in_forward = use_dx_in_forward

    # 编码函数
    def encode(self, x):
        return self.fc1(x)

    # 第二层编码函数
    def encode2(self, x):
        return torch.sigmoid(self.fc2(x))

    # 解码函数
    def decode(self, z):
        return self.activation_fn(self.fc3(z))

    # 正向传播
    def forward(self, x, dx, q):
        # 计算重构结果
        z = self.encode(x) + self.encode2(q) + dx
        re = self.decode(z)
        return re, z

# 定义RMSE损失函数
class RMSELoss(nn.Module):
    # 初始化RMSE损失函数
    def __init__(self):
        super().__init__()
        # 定义均方误差损失
        self.mse = nn.MSELoss()

    # 前向传播
    def forward(self, yhat, y):
        # 计算RMSE
        return torch.sqrt(self.mse(yhat, y))
