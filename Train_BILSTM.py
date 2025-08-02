# 中文注释：导入常用库和自定义模块
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import math
import math
from create_dataset import series_to_supervised
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs
import scipy.stats as stats
import seaborn as sns
import pickle

# GPU设备配置
import os
# 使用指定的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # 使用GPU2和GPU3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 这里的cuda:0实际上是物理GPU2

# 打印GPU信息
if torch.cuda.is_available():
    print("\n🖥️ GPU配置信息:")
    print(f"   可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i} ({props.name}):")
        print(f"      总显存: {props.total_memory/1024**3:.1f}GB")
    print(f"\n   当前使用: GPU2和GPU3 (通过CUDA_VISIBLE_DEVICES映射为cuda:0和cuda:1)")
    print(f"   主GPU设备: cuda:0 (物理GPU2)")
else:
    print("⚠️  未检测到GPU，使用CPU训练")

# 中文注释：忽略警告信息
warnings.filterwarnings('ignore')

#----------------------------------------BiLSTM基准训练配置------------------------------
print("="*50)
print("BiLSTM基准训练模式（优化版本）")
print("直接使用原始vin_2[x[0]]和vin_3[x[0]]数据")
print("跳过Transformer训练，直接进行MC-AE训练")
print("启用双GPU数据并行和混合精度训练")
print("="*50)

#----------------------------------------数据加载------------------------------
# 从Labels.xls加载训练样本ID（0-200号）
def load_train_samples():
    """从Labels.xls加载训练样本ID"""
    try:
        import pandas as pd
        labels_path = '../QAS/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取0-200范围的样本
        all_samples = df['Num'].tolist()
        train_samples = [i for i in all_samples if 0 <= i <= 200]
        
        print(f"📋 从Labels.xls加载训练样本:")
        print(f"   训练样本范围: 0-200")
        print(f"   实际可用样本: {len(train_samples)} 个")
        print(f"   样本ID: {train_samples[:10]}..." if len(train_samples) > 10 else f"   样本ID: {train_samples}")
        
        return train_samples
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认样本范围 0-20")
        return list(range(21))

train_samples = load_train_samples()
print(f"使用QAS目录中的{len(train_samples)}个样本进行训练")

# 定义训练参数（提前定义）
EPOCH = 300
INIT_LR = 1e-5  # 进一步降低初始学习率
MAX_LR = 5e-5   # 降低最大学习率
BATCHSIZE = 4000  # 优化为4000，平衡性能和稳定性
WARMUP_EPOCHS = 10  # 增加学习率预热轮数

# 添加梯度裁剪
MAX_GRAD_NORM = 1.0  # 调整到更合理的梯度裁剪阈值
MIN_GRAD_NORM = 0.1  # 最小梯度范数阈值

# 学习率预热函数
def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return INIT_LR + (MAX_LR - INIT_LR) * epoch / WARMUP_EPOCHS
    return MAX_LR * (0.9 ** (epoch // 50))  # 每50个epoch衰减到90%

# 显示优化后的训练参数
print(f"\n⚙️  BiLSTM训练参数（优化版本）:")
print(f"   批次大小: {BATCHSIZE} (从2000增加到8000，充分利用A100显存)")
print(f"   训练轮数: {EPOCH}")
print(f"   初始学习率: {INIT_LR}")
print(f"   最大学习率: {MAX_LR}")
print(f"   数据并行: 启用")
print(f"   混合精度: 启用 (AMP)")

#----------------------------------------MC-AE训练数据准备（直接使用原始数据）------------------------
print("="*50)
print("阶段1: 准备MC-AE训练数据（使用原始BiLSTM数据）")
print("="*50)

# 数据质量检查函数
def check_data_quality(data, name, sample_id=None):
    """详细的数据质量检查"""
    prefix = f"样本 {sample_id} - " if sample_id else ""
    print(f"\n🔍 {prefix}{name} 数据质量检查:")
    
    # 基本信息
    print(f"   数据类型: {data.dtype}")
    print(f"   数据形状: {data.shape}")
    
    # 数值统计
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    
    print(f"   数值范围: [{data_np.min():.6f}, {data_np.max():.6f}]")
    print(f"   均值: {data_np.mean():.6f}")
    print(f"   标准差: {data_np.std():.6f}")
    print(f"   中位数: {np.median(data_np):.6f}")
    
    # 异常值检查
    nan_count = np.isnan(data_np).sum()
    inf_count = np.isinf(data_np).sum()
    zero_count = (data_np == 0).sum()
    negative_count = (data_np < 0).sum()
    
    print(f"   NaN数量: {nan_count}")
    print(f"   Inf数量: {inf_count}")
    print(f"   零值数量: {zero_count}")
    print(f"   负值数量: {negative_count}")
    
    # 异常值比例
    total_elements = data_np.size
    print(f"   NaN比例: {nan_count/total_elements*100:.2f}%")
    print(f"   Inf比例: {inf_count/total_elements*100:.2f}%")
    print(f"   零值比例: {zero_count/total_elements*100:.2f}%")
    print(f"   负值比例: {negative_count/total_elements*100:.2f}%")
    
    # 异常值警告
    if nan_count > 0:
        print(f"   ⚠️  检测到NaN值！")
    if inf_count > 0:
        print(f"   ⚠️  检测到无穷大值！")
    if data_np.min() < -1e6 or data_np.max() > 1e6:
        print(f"   ⚠️  检测到异常大值！范围: [{data_np.min():.2e}, {data_np.max():.2e}]")
    
    return {
        'has_nan': nan_count > 0,
        'has_inf': inf_count > 0,
        'has_extreme_values': data_np.min() < -1e6 or data_np.max() > 1e6,
        'data_type': data.dtype,
        'shape': data.shape
    }

def physics_based_data_processing(data, name, feature_type='general'):
    """基于物理约束的数据处理（参考论文方法）"""
    print(f"\n🔧 基于物理约束处理 {name}...")
    
    # 检查原始数据类型
    if isinstance(data, np.ndarray):
        print(f"   原始类型: numpy.ndarray, dtype={data.dtype}")
    elif isinstance(data, torch.Tensor):
        print(f"   原始类型: torch.Tensor, dtype={data.dtype}")
    else:
        print(f"   原始类型: {type(data)}")
    
    # 转换为numpy进行预处理
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    
    # 记录原始数据点数量
    original_data_points = data_np.shape[0]
    print(f"   原始数据点数量: {original_data_points}")
    
    print("   执行基于物理约束的数据处理...")
    
    # 1. 处理缺失数据 (Missing Data) - 用中位数替换全NaN行，保持数据点数量
    print("   步骤1: 处理缺失数据...")
    complete_nan_rows = np.isnan(data_np).all(axis=1)
    if complete_nan_rows.any():
        print(f"     检测到 {complete_nan_rows.sum()} 行完全缺失的数据")
        print(f"     用中位数替换全NaN行，保持数据点数量不变")
        
        # 对每个特征维度计算中位数
        for col in range(data_np.shape[1]):
            # 对于vin_3数据的第224列，跳过处理
            if data_np.shape[1] == 226 and col == 224:
                print(f"       特征{col}: 特殊保留列，跳过缺失数据处理")
                continue
                
            valid_values = data_np[~np.isnan(data_np[:, col]), col]
            if len(valid_values) > 0:
                median_val = np.median(valid_values)
                # 替换全NaN行中该特征的值
                data_np[complete_nan_rows, col] = median_val
                print(f"       特征{col}: 用中位数 {median_val:.4f} 替换全NaN行")
            else:
                # 如果该特征全部为NaN，用0替换
                data_np[complete_nan_rows, col] = 0.0
                print(f"       特征{col}: 全部为NaN，用0替换")
    
    # 2. 处理异常数据 (Abnormal Data) - 基于物理约束过滤
    print("   步骤2: 处理异常数据...")
    
    if feature_type == 'vin2':
        # vin_2数据处理（225列）
        print(f"     处理vin_2数据（225列）")
        
        # 索引0,1：BiLSTM和Pack电压预测值 - 限制在[0,5]V
        voltage_pred_columns = [0, 1]
        for col in voltage_pred_columns:
            col_valid_mask = (data_np[:, col] >= 0) & (data_np[:, col] <= 5)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                print(f"       电压预测列{col}: 检测到 {col_invalid_count} 个超出电压范围[0,5]V的异常值")
                data_np[data_np[:, col] < 0, col] = 0
                data_np[data_np[:, col] > 5, col] = 5
            else:
                print(f"       电压预测列{col}: 电压值在正常范围内")
        
        # 索引2-221：220个特征值 - 限制在[-5,5]范围内
        voltage_columns = list(range(2, 222))
        for col in voltage_columns:
            col_valid_mask = (data_np[:, col] >= -5) & (data_np[:, col] <= 5)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                print(f"       电压相关列{col}: 检测到 {col_invalid_count} 个超出电压范围[-5,5]的异常值")
                data_np[data_np[:, col] < -5, col] = -5
                data_np[data_np[:, col] > 5, col] = 5
            else:
                print(f"       电压相关列{col}: 电压值在正常范围内")
        
        # 索引222：电池温度 - 限制在合理温度范围[-40,80]°C
        temp_col = 222
        temp_valid_mask = (data_np[:, temp_col] >= -40) & (data_np[:, temp_col] <= 80)
        temp_invalid_count = (~temp_valid_mask).sum()
        if temp_invalid_count > 0:
            print(f"       温度列{temp_col}: 检测到 {temp_invalid_count} 个超出温度范围[-40,80]°C的异常值")
            data_np[data_np[:, temp_col] < -40, temp_col] = -40
            data_np[data_np[:, temp_col] > 80, temp_col] = 80
        
        # 索引224：电流数据 - 限制在[-1004,162]A
        current_col = 224
        current_valid_mask = (data_np[:, current_col] >= -1004) & (data_np[:, current_col] <= 162)
        current_invalid_count = (~current_valid_mask).sum()
        if current_invalid_count > 0:
            print(f"       电流列{current_col}: 检测到 {current_invalid_count} 个超出电流范围[-1004,162]A的异常值")
            data_np[data_np[:, current_col] < -1004, current_col] = -1004
            data_np[data_np[:, current_col] > 162, current_col] = 162
        
        # 其他列（索引223）：只处理极端异常值
        other_columns = [223]
        for col in other_columns:
            if col < data_np.shape[1]:
                col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
                if col_extreme_mask.any():
                    print(f"       其他列{col}: 检测到 {col_extreme_mask.sum()} 个极端异常值")
                    valid_values = data_np[~col_extreme_mask, col]
                    if len(valid_values) > 0:
                        median_val = np.median(valid_values)
                        data_np[col_extreme_mask, col] = median_val
    
    elif feature_type == 'vin3':
        # vin_3数据处理（226列）
        print(f"     处理vin_3数据（226列），第224列为特殊保留列")
        
        # 索引0,1：BiLSTM和Pack SOC预测值 - 限制在[-0.2,2.0]
        soc_pred_columns = [0, 1]
        for col in soc_pred_columns:
            col_valid_mask = (data_np[:, col] >= -0.2) & (data_np[:, col] <= 2.0)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                print(f"       SOC预测列{col}: 检测到 {col_invalid_count} 个超出SOC范围[-0.2,2.0]的异常值")
                data_np[data_np[:, col] < -0.2, col] = -0.2
                data_np[data_np[:, col] > 2.0, col] = 2.0
            else:
                print(f"       SOC预测列{col}: SOC值在正常范围内")
        
        # 索引2-111：110个单体电池真实SOC值 - 限制在[-0.2,2.0]
        cell_soc_columns = list(range(2, 112))
        for col in cell_soc_columns:
            col_valid_mask = (data_np[:, col] >= -0.2) & (data_np[:, col] <= 2.0)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                print(f"       单体SOC列{col}: 检测到 {col_invalid_count} 个超出SOC范围[-0.2,2.0]的异常值")
                data_np[data_np[:, col] < -0.2, col] = -0.2
                data_np[data_np[:, col] > 2.0, col] = 2.0
        
        # 索引112-221：110个单体电池SOC偏差值 - 不限制范围，只处理极端异常值
        soc_dev_columns = list(range(112, 222))
        for col in soc_dev_columns:
            col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
            if col_extreme_mask.any():
                print(f"       SOC偏差列{col}: 检测到 {col_extreme_mask.sum()} 个极端异常值")
                valid_values = data_np[~col_extreme_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    data_np[col_extreme_mask, col] = median_val
        
        # 索引222：电池温度 - 限制在合理温度范围[-40,80]°C
        temp_col = 222
        temp_valid_mask = (data_np[:, temp_col] >= -40) & (data_np[:, temp_col] <= 80)
        temp_invalid_count = (~temp_valid_mask).sum()
        if temp_invalid_count > 0:
            print(f"       温度列{temp_col}: 检测到 {temp_invalid_count} 个超出温度范围[-40,80]°C的异常值")
            data_np[data_np[:, temp_col] < -40, temp_col] = -40
            data_np[data_np[:, temp_col] > 80, temp_col] = 80
        
        # 索引224：特殊保留列 - 保持原值不变
        special_col = 224
        print(f"       特殊保留列{special_col}: 保持原值不变")
        
        # 索引225：电流数据 - 限制在[-1004,162]A
        current_col = 225
        current_valid_mask = (data_np[:, current_col] >= -1004) & (data_np[:, current_col] <= 162)
        current_invalid_count = (~current_valid_mask).sum()
        if current_invalid_count > 0:
            print(f"       电流列{current_col}: 检测到 {current_invalid_count} 个超出电流范围[-1004,162]A的异常值")
            data_np[data_np[:, current_col] < -1004, current_col] = -1004
            data_np[data_np[:, current_col] > 162, current_col] = 162
        
        # 其他列（索引223）：只处理极端异常值
        other_columns = [223]
        for col in other_columns:
            col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
            if col_extreme_mask.any():
                print(f"       其他列{col}: 检测到 {col_extreme_mask.sum()} 个极端异常值")
                valid_values = data_np[~col_extreme_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    data_np[col_extreme_mask, col] = median_val
            
    elif feature_type == 'current':
        # 电流物理约束：-100A到100A
        valid_mask = (data_np >= -100) & (data_np <= 100)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"     检测到 {invalid_count} 个超出电流范围[-100,100]A的异常值")
            data_np[data_np < -100] = -100
            data_np[data_np > 100] = 100
            
    elif feature_type == 'temperature':
        # 温度物理约束：-40°C到80°C
        valid_mask = (data_np >= -40) & (data_np <= 80)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"     检测到 {invalid_count} 个超出温度范围[-40,80]°C的异常值")
            data_np[data_np < -40] = -40
            data_np[data_np > 80] = 80
    
    # 3. 处理采样故障 (Sampling Faults) - 用中位数替换，保持数据点数量
    print("   步骤3: 处理采样故障...")
    
    # 检测NaN和Inf值（可能是采样故障）
    nan_mask = np.isnan(data_np)
    inf_mask = np.isinf(data_np)
    fault_mask = nan_mask | inf_mask
    
    if fault_mask.any():
        print(f"     检测到 {fault_mask.sum()} 个采样故障点")
        print(f"     用中位数替换故障点，保持数据点数量不变")
        
        # 对每个特征维度分别处理
        for col in range(data_np.shape[1]):
            # 对于vin_3数据的第224列，跳过处理
            if data_np.shape[1] == 226 and col == 224:
                print(f"       特征{col}: 特殊保留列，跳过采样故障处理")
                continue
                
            col_fault_mask = fault_mask[:, col]
            if col_fault_mask.any():
                # 计算该列的中位数（排除故障值）
                valid_values = data_np[~col_fault_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    print(f"       特征{col}: 用中位数 {median_val:.4f} 替换 {col_fault_mask.sum()} 个故障值")
                    data_np[col_fault_mask, col] = median_val
                else:
                    # 如果该列全部为故障值，用0替换
                    print(f"       特征{col}: 全部为故障值，用0替换")
                    data_np[col_fault_mask, col] = 0.0
    
    # 4. 最终检查
    print("   步骤4: 最终数据质量检查...")
    final_nan_count = np.isnan(data_np).sum()
    final_inf_count = np.isinf(data_np).sum()
    
    if final_nan_count > 0 or final_inf_count > 0:
        print(f"     ⚠️  仍有 {final_nan_count} 个NaN和 {final_inf_count} 个Inf值")
        # 最后的安全处理
        data_np[np.isnan(data_np)] = 0.0
        data_np[np.isinf(data_np)] = 0.0
    else:
        print("     ✅ 所有异常值已处理完成")
    
    # 转换为tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    
    # 检查数据点数量是否保持一致
    final_data_points = data_tensor.shape[0]
    if final_data_points == original_data_points:
        print(f"   处理完成: {data_tensor.shape}, dtype={data_tensor.dtype}")
        print(f"   ✅ 数据点数量保持一致: {original_data_points} -> {final_data_points}")
    else:
        print(f"   ⚠️  数据点数量发生变化: {original_data_points} -> {final_data_points}")
    
    return data_tensor

def physics_based_data_processing_silent(data, feature_type='general'):
    """基于物理约束的数据处理（静默模式，只返回处理后的数据）"""
    # 转换为numpy进行预处理
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    
    # 记录原始数据点数量
    original_data_points = data_np.shape[0]
    
    # 1. 处理缺失数据 (Missing Data) - 用中位数替换全NaN行，保持数据点数量
    complete_nan_rows = np.isnan(data_np).all(axis=1)
    if complete_nan_rows.any():
        # 对每个特征维度计算中位数
        for col in range(data_np.shape[1]):
            # 对于vin_3数据的第224列，跳过处理
            if data_np.shape[1] == 226 and col == 224:
                continue
                
            valid_values = data_np[~np.isnan(data_np[:, col]), col]
            if len(valid_values) > 0:
                median_val = np.median(valid_values)
                # 替换全NaN行中该特征的值
                data_np[complete_nan_rows, col] = median_val
            else:
                # 如果该特征全部为NaN，用0替换
                data_np[complete_nan_rows, col] = 0.0
    
    # 2. 处理异常数据 (Abnormal Data) - 基于物理约束过滤
    if feature_type == 'vin2':
        # vin_2数据处理（225列）
        
        # 索引0,1：BiLSTM和Pack电压预测值 - 限制在[0,5]V
        voltage_pred_columns = [0, 1]
        for col in voltage_pred_columns:
            col_valid_mask = (data_np[:, col] >= 0) & (data_np[:, col] <= 5)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                data_np[data_np[:, col] < 0, col] = 0
                data_np[data_np[:, col] > 5, col] = 5
        
        # 索引2-221：220个特征值 - 限制在[-5,5]范围内
        voltage_columns = list(range(2, 222))
        for col in voltage_columns:
            col_valid_mask = (data_np[:, col] >= -5) & (data_np[:, col] <= 5)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                data_np[data_np[:, col] < -5, col] = -5
                data_np[data_np[:, col] > 5, col] = 5
        
        # 索引222：电池温度 - 限制在合理温度范围[-40,80]°C
        temp_col = 222
        temp_valid_mask = (data_np[:, temp_col] >= -40) & (data_np[:, temp_col] <= 80)
        temp_invalid_count = (~temp_valid_mask).sum()
        if temp_invalid_count > 0:
            data_np[data_np[:, temp_col] < -40, temp_col] = -40
            data_np[data_np[:, temp_col] > 80, temp_col] = 80
        
        # 索引224：电流数据 - 限制在[-1004,162]A
        current_col = 224
        current_valid_mask = (data_np[:, current_col] >= -1004) & (data_np[:, current_col] <= 162)
        current_invalid_count = (~current_valid_mask).sum()
        if current_invalid_count > 0:
            data_np[data_np[:, current_col] < -1004, current_col] = -1004
            data_np[data_np[:, current_col] > 162, current_col] = 162
        
        # 其他列（索引223）：只处理极端异常值
        other_columns = [223]
        for col in other_columns:
            if col < data_np.shape[1]:
                col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
                if col_extreme_mask.any():
                    valid_values = data_np[~col_extreme_mask, col]
                    if len(valid_values) > 0:
                        median_val = np.median(valid_values)
                        data_np[col_extreme_mask, col] = median_val
    
    elif feature_type == 'vin3':
        # vin_3数据处理（226列）
        
        # 索引0,1：BiLSTM和Pack SOC预测值 - 限制在[-0.2,2.0]
        soc_pred_columns = [0, 1]
        for col in soc_pred_columns:
            col_valid_mask = (data_np[:, col] >= -0.2) & (data_np[:, col] <= 2.0)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                data_np[data_np[:, col] < -0.2, col] = -0.2
                data_np[data_np[:, col] > 2.0, col] = 2.0
        
        # 索引2-111：110个单体电池真实SOC值 - 限制在[-0.2,2.0]
        cell_soc_columns = list(range(2, 112))
        for col in cell_soc_columns:
            col_valid_mask = (data_np[:, col] >= -0.2) & (data_np[:, col] <= 2.0)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                data_np[data_np[:, col] < -0.2, col] = -0.2
                data_np[data_np[:, col] > 2.0, col] = 2.0
        
        # 索引112-221：110个单体电池SOC偏差值 - 不限制范围，只处理极端异常值
        soc_dev_columns = list(range(112, 222))
        for col in soc_dev_columns:
            col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
            if col_extreme_mask.any():
                valid_values = data_np[~col_extreme_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    data_np[col_extreme_mask, col] = median_val
        
        # 索引222：电池温度 - 限制在合理温度范围[-40,80]°C
        temp_col = 222
        temp_valid_mask = (data_np[:, temp_col] >= -40) & (data_np[:, temp_col] <= 80)
        temp_invalid_count = (~temp_valid_mask).sum()
        if temp_invalid_count > 0:
            data_np[data_np[:, temp_col] < -40, temp_col] = -40
            data_np[data_np[:, temp_col] > 80, temp_col] = 80
        
        # 索引224：特殊保留列 - 保持原值不变
        # 不需要处理，保持原值
        
        # 索引225：电流数据 - 限制在[-1004,162]A
        current_col = 225
        current_valid_mask = (data_np[:, current_col] >= -1004) & (data_np[:, current_col] <= 162)
        current_invalid_count = (~current_valid_mask).sum()
        if current_invalid_count > 0:
            data_np[data_np[:, current_col] < -1004, current_col] = -1004
            data_np[data_np[:, current_col] > 162, current_col] = 162
        
        # 其他列（索引223）：只处理极端异常值
        other_columns = [223]
        for col in other_columns:
            col_extreme_mask = np.isnan(data_np[:, col]) | np.isinf(data_np[:, col])
            if col_extreme_mask.any():
                valid_values = data_np[~col_extreme_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    data_np[col_extreme_mask, col] = median_val
    
    # 3. 处理采样故障 (Sampling Faults) - 用中位数替换，保持数据点数量
    # 检测NaN和Inf值（可能是采样故障）
    nan_mask = np.isnan(data_np)
    inf_mask = np.isinf(data_np)
    fault_mask = nan_mask | inf_mask
    
    if fault_mask.any():
        # 对每个特征维度分别处理
        for col in range(data_np.shape[1]):
            # 对于vin_3数据的第224列，跳过处理
            if data_np.shape[1] == 226 and col == 224:
                continue
                
            col_fault_mask = fault_mask[:, col]
            if col_fault_mask.any():
                # 计算该列的中位数（排除故障值）
                valid_values = data_np[~col_fault_mask, col]
                if len(valid_values) > 0:
                    median_val = np.median(valid_values)
                    data_np[col_fault_mask, col] = median_val
                else:
                    # 如果该列全部为故障值，用0替换
                    data_np[col_fault_mask, col] = 0.0
    
    # 4. 最终检查
    final_nan_count = np.isnan(data_np).sum()
    final_inf_count = np.isinf(data_np).sum()
    
    if final_nan_count > 0 or final_inf_count > 0:
        # 最后的安全处理
        data_np[np.isnan(data_np)] = 0.0
        data_np[np.isinf(data_np)] = 0.0
    
    # 转换为tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    
    return data_tensor

# 中文注释：加载MC-AE模型输入特征（vin_2.pkl和vin_3.pkl）
# 合并所有训练样本的vin_2和vin_3数据
all_vin2_data = []
all_vin3_data = []

# 汇总统计信息
sample_summary = {
    'total_samples': len(train_samples),
    'processed_samples': 0,
    'error_samples': 0,
    'total_vin2_issues_fixed': 0,
    'total_vin3_issues_fixed': 0
}

print("="*60)
print("📥 开始数据加载和质量检查")
print("="*60)

for sample_id in train_samples:
    vin2_path = f'../QAS/{sample_id}/vin_2.pkl'
    vin3_path = f'../QAS/{sample_id}/vin_3.pkl'
    
    # 加载原始vin_2数据
    try:
        with open(vin2_path, 'rb') as file:
            vin2_data = pickle.load(file)
        
        # 基于物理约束的数据处理（静默模式）
        vin2_tensor = physics_based_data_processing_silent(vin2_data, feature_type='vin2')
        
    except Exception as e:
        print(f"❌ 加载样本 {sample_id} 的vin_2数据失败: {e}")
        sample_summary['error_samples'] += 1
        continue
    
    # 加载原始vin_3数据
    try:
        with open(vin3_path, 'rb') as file:
            vin3_data = pickle.load(file)
        
        # 基于物理约束的数据处理（静默模式）
        vin3_tensor = physics_based_data_processing_silent(vin3_data, feature_type='vin3')
        
    except Exception as e:
        print(f"❌ 加载样本 {sample_id} 的vin_3数据失败: {e}")
        sample_summary['error_samples'] += 1
        continue
    
    # 添加到列表
    all_vin2_data.append(vin2_tensor)
    all_vin3_data.append(vin3_tensor)
    sample_summary['processed_samples'] += 1
    
    # 每处理10个样本输出一次进度
    if sample_summary['processed_samples'] % 10 == 0:
        print(f"📊 已处理 {sample_summary['processed_samples']}/{sample_summary['total_samples']} 个样本")

# 输出处理汇总信息
print(f"\n✅ 数据加载完成:")
print(f"   总样本数: {sample_summary['total_samples']}")
print(f"   成功处理: {sample_summary['processed_samples']}")
print(f"   处理失败: {sample_summary['error_samples']}")
print(f"   成功率: {sample_summary['processed_samples']/sample_summary['total_samples']*100:.1f}%")

# 合并数据
print("\n" + "="*60)
print("🔗 合并所有样本数据")
print("="*60)

combined_tensor = torch.cat(all_vin2_data, dim=0)
combined_tensorx = torch.cat(all_vin3_data, dim=0)

print(f"合并后vin_2数据形状: {combined_tensor.shape}")
print(f"合并后vin_3数据形状: {combined_tensorx.shape}")

# 简要检查合并后的数据质量
print("\n🔍 合并后数据质量简要检查:")
vin2_nan_count = torch.isnan(combined_tensor).sum().item()
vin2_inf_count = torch.isinf(combined_tensor).sum().item()
vin3_nan_count = torch.isnan(combined_tensorx).sum().item()
vin3_inf_count = torch.isinf(combined_tensorx).sum().item()

print(f"   vin_2: NaN={vin2_nan_count}, Inf={vin2_inf_count}")
print(f"   vin_3: NaN={vin3_nan_count}, Inf={vin3_inf_count}")

# 检查是否有异常值需要处理
vin2_has_issues = (vin2_nan_count > 0 or vin2_inf_count > 0)
vin3_has_issues = (vin3_nan_count > 0 or vin3_inf_count > 0)

if vin2_has_issues or vin3_has_issues:
    print("⚠️  检测到数据问题，进行修复...")
    
    # 修复NaN和Inf值
    if vin2_has_issues:
        combined_tensor = torch.where(torch.isnan(combined_tensor) | torch.isinf(combined_tensor), 
                                     torch.zeros_like(combined_tensor), combined_tensor)
        print("   ✅ vin_2数据修复完成")
    
    if vin3_has_issues:
        combined_tensorx = torch.where(torch.isnan(combined_tensorx) | torch.isinf(combined_tensorx), 
                                      torch.zeros_like(combined_tensorx), combined_tensorx)
        print("   ✅ vin_3数据修复完成")
else:
    print("✅ 数据质量良好，无需修复")

#----------------------------------------MC-AE多通道自编码器训练--------------------------
print("="*50)
print("阶段2: 训练MC-AE异常检测模型（使用原始BiLSTM数据）")
print("="*50)

# 中文注释：定义特征切片维度
# vin_2.pkl
dim_x = 2
dim_y = 110
dim_z = 110
dim_q = 3

# 中文注释：分割特征张量
x_recovered = combined_tensor[:, :dim_x]
y_recovered = combined_tensor[:, dim_x:dim_x + dim_y]
z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z]
q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:]

# vin_3.pkl
dim_x2 = 2
dim_y2 = 110
dim_z2 = 110
dim_q2= 4

x_recovered2 = combined_tensorx[:, :dim_x2]
y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]

# 训练超参数配置（已在前面定义）

# 用于记录训练损失
train_losses_mcae1 = []
train_losses_mcae2 = []

# 中文注释：自定义多输入数据集类（本地定义，非Class_.py中的Dataset）
class Dataset(Dataset):
    def __init__(self, x, y, z, q):
        self.x = x.to(torch.float32)
        self.y = y.to(torch.float32)
        self.z = z.to(torch.float32)
        self.q = q.to(torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]

# 中文注释：用DataLoader批量加载多通道特征数据
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=BATCHSIZE, shuffle=False)

# 中文注释：初始化MC-AE模型（使用float32）
net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device).to(torch.float32)

netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device).to(torch.float32)

# 使用更稳定的权重初始化
def stable_weight_init(model):
    """使用更稳定的权重初始化方法"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化，但限制权重范围
            nn.init.xavier_uniform_(module.weight, gain=0.3)  # 降低gain值避免梯度爆炸
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# 应用稳定的权重初始化
stable_weight_init(net)
stable_weight_init(netx)
print("✅ 应用稳定的权重初始化")

# 启用数据并行
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    netx = torch.nn.DataParallel(netx, device_ids=[0, 1])
    print(f"✅ 启用数据并行，使用 {torch.cuda.device_count()} 张GPU (device_ids=[0,1])")
else:
    print("⚠️  单GPU模式")

optimizer = torch.optim.Adam(net.parameters(), lr=INIT_LR)
l1_lambda = 0.01
loss_f = nn.MSELoss()

# 启用混合精度训练
scaler = torch.cuda.amp.GradScaler()
print("✅ 启用混合精度训练 (AMP)")

# 启用CUDA性能优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
print("✅ 启用CUDA性能优化 (cudnn.benchmark)")
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    
    # 更新学习率
    current_lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for iteration, (x, y, z, q) in enumerate(train_loader_u):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        
        # 检查输入数据范围
        if torch.isnan(x).any() or torch.isinf(x).any() or torch.isnan(y).any() or torch.isinf(y).any():
            print(f"警告：第{epoch}轮第{iteration}批次输入数据包含NaN/Inf，跳过此批次")
            continue
        
        # 检查输入数据范围是否合理
        if x.abs().max() > 1000 or y.abs().max() > 1000:
            print(f"警告：第{epoch}轮第{iteration}批次输入数据范围过大，跳过此批次")
            print(f"x范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"y范围: [{y.min():.4f}, {y.max():.4f}]")
            continue
        
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            recon_im, recon_p = net(x, z, q)
            loss_u = loss_f(y, recon_im)
            
                    # 检查损失值是否为NaN
        if torch.isnan(loss_u) or torch.isinf(loss_u):
            print(f"警告：第{epoch}轮第{iteration}批次检测到NaN/Inf损失值")
            print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"输出范围: [{recon_im.min():.4f}, {recon_im.max():.4f}]")
            print(f"损失值: {loss_u.item()}")
            print("跳过此批次，不进行反向传播")
            continue
        
        # 检查输入数据范围是否合理
        if x.abs().max() > 1000 or y.abs().max() > 1000:
            print(f"警告：第{epoch}轮第{iteration}批次输入数据范围过大，跳过此批次")
            print(f"x范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"y范围: [{y.min():.4f}, {y.max():.4f}]")
            continue
        
        total_loss += loss_u.item()
        num_batches += 1
        optimizer.zero_grad()
        
        # 使用混合精度训练
        scaler.scale(loss_u).backward()
        
        # 检查梯度是否为NaN或无穷大
        grad_norm = 0
        has_grad_issue = False
        
        # 安全地处理梯度
        try:
            # 在检查梯度前unscale
            scaler.unscale_(optimizer)
            
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"警告：参数 {name} 的梯度出现NaN或无穷大，跳过此批次")
                        has_grad_issue = True
                        break
                    grad_norm += param.grad.data.norm(2).item() ** 2
            
            if has_grad_issue:
                # 重置scaler状态
                scaler.update()
                continue
                
            grad_norm = grad_norm ** 0.5
            
            # 渐进式梯度裁剪 - 只显示异常情况
            if grad_norm > MAX_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                print(f"⚠️  梯度裁剪: {grad_norm:.4f} -> {MAX_GRAD_NORM}")
            elif grad_norm < MIN_GRAD_NORM:
                print(f"⚠️  梯度过小: {grad_norm:.4f} < {MIN_GRAD_NORM}")
            # 移除"梯度正常"的输出，减少日志噪音
            
            # 执行优化器步骤
            scaler.step(optimizer)
            scaler.update()
            
        except Exception as e:
            print(f"优化器步骤失败: {e}")
            print("跳过此批次并重置scaler状态")
            # 重置scaler状态
            scaler.update()
            continue
    
    avg_loss = total_loss / num_batches
    train_losses_mcae1.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE1 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))
    elif epoch % 10 == 0:  # 每10个epoch输出一次进度
        print('MC-AE1 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))
        # GPU利用率监控
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU显存: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB ({gpu_memory_used/gpu_memory_total*100:.1f}%)")

# 中文注释：全量推理，获得重构误差
train_loader2 = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=len(x_recovered), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loader2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    with torch.cuda.amp.autocast():
        recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = AA - yTrainU

# 中文注释：第二组特征的MC-AE训练
train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=BATCHSIZE, shuffle=False)
optimizer = torch.optim.Adam(netx.parameters(), lr=INIT_LR)
loss_f = nn.MSELoss()

# 为第二个模型创建新的scaler
scaler2 = torch.cuda.amp.GradScaler()

avg_loss_list_x = []
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    
    # 更新学习率
    current_lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            recon_im, z = netx(x, z, q)
            loss_x = loss_f(y, recon_im)
            
                    # 检查损失值是否为NaN
        if torch.isnan(loss_x) or torch.isinf(loss_x):
            print(f"警告：第{epoch}轮第{iteration}批次检测到NaN/Inf损失值")
            print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"输出范围: [{recon_im.min():.4f}, {recon_im.max():.4f}]")
            print(f"损失值: {loss_x.item()}")
            print("跳过此批次，不进行反向传播")
            continue
        
        # 检查输入数据范围是否合理
        if x.abs().max() > 1000 or y.abs().max() > 1000:
            print(f"警告：第{epoch}轮第{iteration}批次输入数据范围过大，跳过此批次")
            print(f"x范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"y范围: [{y.min():.4f}, {y.max():.4f}]")
            continue
        
        total_loss += loss_x.item()
        num_batches += 1
        optimizer.zero_grad()
        scaler2.scale(loss_x).backward()
        
        # 检查梯度是否为NaN或无穷大
        grad_norm = 0
        has_grad_issue = False
        
        # 安全地处理梯度
        try:
            # 在检查梯度前unscale
            scaler2.unscale_(optimizer)
            
            for name, param in netx.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"警告：参数 {name} 的梯度出现NaN或无穷大，跳过此批次")
                        has_grad_issue = True
                        break
                    grad_norm += param.grad.data.norm(2).item() ** 2
            
            if has_grad_issue:
                # 重置scaler状态
                scaler2.update()
                continue
                
            grad_norm = grad_norm ** 0.5
            
            # 渐进式梯度裁剪 - 只显示异常情况
            if grad_norm > MAX_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(netx.parameters(), MAX_GRAD_NORM)
                print(f"⚠️  梯度裁剪: {grad_norm:.4f} -> {MAX_GRAD_NORM}")
            elif grad_norm < MIN_GRAD_NORM:
                print(f"⚠️  梯度过小: {grad_norm:.4f} < {MIN_GRAD_NORM}")
            # 移除"梯度正常"的输出，减少日志噪音
            
            # 执行优化器步骤
            scaler2.step(optimizer)
            scaler2.update()
            
        except Exception as e:
            print(f"优化器步骤失败: {e}")
            print("跳过此批次并重置scaler状态")
            # 重置scaler状态
            scaler2.update()
            continue
    
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    train_losses_mcae2.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE2 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))
    elif epoch % 10 == 0:  # 每10个epoch输出一次进度
        print('MC-AE2 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))
        # GPU利用率监控
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU显存: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB ({gpu_memory_used/gpu_memory_total*100:.1f}%)")

train_loaderx2 = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=len(x_recovered2), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loaderx2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    with torch.cuda.amp.autocast():
        recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = BB - yTrainX

# 创建结果目录
result_dir = './models'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 中文注释：诊断特征提取与PCA分析
df_data = DiagnosisFeature(ERRORU,ERRORX)

v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data,0.95,0.95)

# 训练结束后自动保存模型和分析结果
print("="*50)
print("保存BiLSTM基准训练结果")
print("="*50)

# 绘制训练结果
print("📈 绘制BiLSTM训练曲线...")

# Linux环境matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Linux环境字体设置 - 修复中文显示问题
import matplotlib.font_manager as fm
import os

# 尝试多种字体方案
font_options = [
    'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
    'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS'
]

# 检查可用字体
available_fonts = []
for font in font_options:
    try:
        fm.findfont(font)
        available_fonts.append(font)
    except:
        continue

# 设置字体
if available_fonts:
    plt.rcParams['font.sans-serif'] = available_fonts
    print(f"✅ 使用字体: {available_fonts[0]}")
else:
    # 如果都不可用，使用英文标签
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️  未找到中文字体，将使用英文标签")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 子图1: MC-AE1训练损失曲线
ax1 = axes[0, 0]
epochs = range(1, len(train_losses_mcae1) + 1)
ax1.plot(epochs, train_losses_mcae1, 'b-', linewidth=2, label='MC-AE1 Training Loss')
ax1.set_xlabel('Training Epochs / 训练轮数')
ax1.set_ylabel('MSE Loss / MSE损失')
ax1.set_title('MC-AE1 Training Loss / MC-AE1训练损失曲线')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_yscale('log')

# 子图2: MC-AE2训练损失曲线 
ax2 = axes[0, 1]
ax2.plot(epochs, train_losses_mcae2, 'r-', linewidth=2, label='MC-AE2 Training Loss')
ax2.set_xlabel('Training Epochs / 训练轮数')
ax2.set_ylabel('MSE Loss / MSE损失')
ax2.set_title('MC-AE2 Training Loss / MC-AE2训练损失曲线')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_yscale('log')

# 子图3: MC-AE1重构误差分布
ax3 = axes[1, 0]
reconstruction_errors_1 = ERRORU.flatten()
mean_error_1 = np.mean(np.abs(reconstruction_errors_1))
ax3.hist(np.abs(reconstruction_errors_1), bins=50, alpha=0.7, color='blue', 
         label=f'MC-AE1 Reconstruction Error (Mean: {mean_error_1:.4f}) / MC-AE1重构误差 (均值: {mean_error_1:.4f})')
ax3.set_xlabel('Absolute Reconstruction Error / 绝对重构误差')
ax3.set_ylabel('Frequency / 频数')
ax3.set_title('MC-AE1 Reconstruction Error Distribution / MC-AE1重构误差分布')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4: MC-AE2重构误差分布
ax4 = axes[1, 1]
reconstruction_errors_2 = ERRORX.flatten()
mean_error_2 = np.mean(np.abs(reconstruction_errors_2))
ax4.hist(np.abs(reconstruction_errors_2), bins=50, alpha=0.7, color='red',
         label=f'MC-AE2 Reconstruction Error (Mean: {mean_error_2:.4f}) / MC-AE2重构误差 (均值: {mean_error_2:.4f})')
ax4.set_xlabel('Absolute Reconstruction Error / 绝对重构误差')
ax4.set_ylabel('Frequency / 频数')
ax4.set_title('MC-AE2 Reconstruction Error Distribution / MC-AE2重构误差分布')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{result_dir}/bilstm_training_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ BiLSTM训练结果图已保存: {result_dir}/bilstm_training_results.png")

# 确保结果目录存在
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 2. 保存诊断特征DataFrame
df_data.to_excel(f'{result_dir}/diagnosis_feature_bilstm_baseline.xlsx', index=False)
df_data.to_csv(f'{result_dir}/diagnosis_feature_bilstm_baseline.csv', index=False)
print(f"✓ 保存诊断特征: {result_dir}/diagnosis_feature_bilstm_baseline.xlsx/csv")

# 3. 保存PCA分析主要结果
np.save(f'{result_dir}/v_I_bilstm_baseline.npy', v_I)
np.save(f'{result_dir}/v_bilstm_baseline.npy', v)
np.save(f'{result_dir}/v_ratio_bilstm_baseline.npy', v_ratio)
np.save(f'{result_dir}/p_k_bilstm_baseline.npy', p_k)
np.save(f'{result_dir}/data_mean_bilstm_baseline.npy', data_mean)
np.save(f'{result_dir}/data_std_bilstm_baseline.npy', data_std)
np.save(f'{result_dir}/T_95_limit_bilstm_baseline.npy', T_95_limit)
np.save(f'{result_dir}/T_99_limit_bilstm_baseline.npy', T_99_limit)
np.save(f'{result_dir}/SPE_95_limit_bilstm_baseline.npy', SPE_95_limit)
np.save(f'{result_dir}/SPE_99_limit_bilstm_baseline.npy', SPE_99_limit)
np.save(f'{result_dir}/P_bilstm_baseline.npy', P)
np.save(f'{result_dir}/k_bilstm_baseline.npy', k)
np.save(f'{result_dir}/P_t_bilstm_baseline.npy', P_t)
np.save(f'{result_dir}/X_bilstm_baseline.npy', X)
np.save(f'{result_dir}/data_nor_bilstm_baseline.npy', data_nor)
print(f"✓ 保存PCA分析结果: {result_dir}/*_bilstm_baseline.npy")

# 4. 保存CombinedAE模型参数
torch.save(net.state_dict(), f'{result_dir}/net_model_bilstm_baseline.pth')
torch.save(netx.state_dict(), f'{result_dir}/netx_model_bilstm_baseline.pth')
print(f"✓ 保存MC-AE模型: {result_dir}/net_model_bilstm_baseline.pth, {result_dir}/netx_model_bilstm_baseline.pth")

# 5. 保存训练历史
training_history = {
    'mcae1_losses': train_losses_mcae1,
    'mcae2_losses': train_losses_mcae2,
    'final_mcae1_loss': train_losses_mcae1[-1],
    'final_mcae2_loss': train_losses_mcae2[-1],
    'mcae1_reconstruction_error_mean': np.mean(np.abs(ERRORU)),
    'mcae1_reconstruction_error_std': np.std(np.abs(ERRORU)),
    'mcae2_reconstruction_error_mean': np.mean(np.abs(ERRORX)),
    'mcae2_reconstruction_error_std': np.std(np.abs(ERRORX)),
    'training_samples': len(train_samples),
    'epochs': EPOCH,
    'learning_rate': LR,
    'batch_size': BATCHSIZE
}

import pickle
with open(f'{result_dir}/bilstm_training_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)
print(f"✓ 保存训练历史: {result_dir}/bilstm_training_history.pkl")

print("="*50)
print("🎉 BiLSTM基准训练完成！")
print("="*50)
print("BiLSTM基准模式总结：")
print("1. ✅ 跳过Transformer训练阶段")
print("2. ✅ 直接使用原始vin_2[x[0]]和vin_3[x[0]]数据")
print("3. ✅ 保持Pack Modeling输出vin_2[x[1]]和vin_3[x[1]]不变")
print("4. ✅ MC-AE使用原始BiLSTM数据进行训练")
print("5. ✅ 所有模型和结果文件添加'_bilstm_baseline'后缀")
print("")
print("📊 比对说明：")
print("   - 此模式建立BiLSTM基准性能")
print("   - 可与Transformer模式进行公平对比")
print("   - 便于评估Transformer替换的效果")
print("   - 训练时间更短，适合快速验证") 