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
from scipy import ndimage  # 添加用于时序平滑的导入

# 导入新的数据加载器
from data_loader_transformer import TransformerBatteryDataset, create_transformer_dataloader

# 内存监控函数
def print_gpu_memory():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.1f}GB / {cached:.1f}GB / {total:.1f}GB (已用/缓存/总计)")

# 混合精度训练配置
def setup_mixed_precision():
    """设置混合精度训练"""
    scaler = torch.cuda.amp.GradScaler()
    print("✅ 启用混合精度训练 (AMP)")
    return scaler

# 数据处理函数（从BiLSTM脚本复制）
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
        
        # 索引2-221：220个特征值 - 统一限制在[-5,5]范围内
        voltage_columns = list(range(2, 222))
        for col in voltage_columns:
            col_valid_mask = (data_np[:, col] >= -5) & (data_np[:, col] <= 5)
            col_invalid_count = (~col_valid_mask).sum()
            if col_invalid_count > 0:
                print(f"       电压相关列{col}: 检测到 {col_invalid_count} 个超出范围[-5,5]的异常值")
                data_np[data_np[:, col] < -5, col] = -5
                data_np[data_np[:, col] > 5, col] = 5
        
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
        
        # 索引2-221：220个特征值 - 统一限制在[-5,5]范围内
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
            
    elif feature_type == 'current':
        # 电流物理约束：-100A到100A
        valid_mask = (data_np >= -100) & (data_np <= 100)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            data_np[data_np < -100] = -100
            data_np[data_np > 100] = 100
            
    elif feature_type == 'temperature':
        # 温度物理约束：-40°C到80°C
        valid_mask = (data_np >= -40) & (data_np <= 80)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            data_np[data_np < -40] = -40
            data_np[data_np > 80] = 80
    
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

# GPU设备配置 - 使用GPU0和GPU1进行数据并行
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用GPU0和GPU1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 打印GPU信息
if torch.cuda.is_available():
    print(f"\n🖥️ 双GPU并行配置:")
    print(f"   可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i} ({props.name}): {props.total_memory/1024**3:.1f}GB")
    print(f"   主GPU设备: cuda:0")
    print(f"   数据并行模式: 启用")
else:
    print("⚠️  未检测到GPU，使用CPU训练")

# 中文注释：忽略警告信息
warnings.filterwarnings('ignore')

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

#----------------------------------------重要说明：使用预计算真实值------------------------------
# 新策略：直接使用预计算的真实Terminal Voltage和Pack SOC值进行训练
# 
# 数据流程：
# 1. 输入：vin_1前5维 + 当前时刻真实电压 + 当前时刻真实SOC (7维)
# 2. 输出：下一时刻真实电压 + 下一时刻真实SOC (2维)
# 3. 无需数值转换，直接预测物理值
#
# 优势：
# - 避免了复杂的数值范围转换
# - 直接学习物理量之间的关系
# - 训练目标明确，收敛更快

#----------------------------------------Transformer模型定义------------------------------
class TransformerPredictor(nn.Module):
    """时序预测Transformer模型 - 直接预测真实物理值"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=3, output_size=2):
        super(TransformerPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model, dtype=torch.float32))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 - 直接输出物理值，不使用Sigmoid
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, output_size)
            # 移除Sigmoid，直接输出物理值
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: [batch, input_size] - 2维输入
        if len(x.shape) == 2:
            # 添加序列维度：[batch, input_size] -> [batch, 1, input_size]
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Transformer编码
        transformer_out = self.transformer(x)  # [batch, seq_len, d_model]
        
        # 输出层（使用最后一个时间步）
        output = self.output_layer(transformer_out[:, -1, :])  # [batch, output_size]
        
        return output  # [batch, output_size] 直接返回2维

def main():
    """主训练函数"""
    print("="*60)
    print("🚀 Transformer训练 - Linux环境版本")
    print("="*60)
    
    # Linux环境检查
    import platform
    print(f"🖥️  运行环境: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {platform.python_version()}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"🚀 GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"🔢 GPU数量: {torch.cuda.device_count()}")
    else:
        print("⚠️  GPU不可用，使用CPU训练")
    
    #----------------------------------------数据加载------------------------------
    # 训练样本ID（使用QAS 0-200样本）
    def load_train_samples():
        """从Labels.xls加载训练样本ID"""
        try:
            import pandas as pd
            labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
            df = pd.read_excel(labels_path)
            
            # 提取0-200范围的样本
            all_samples = df['Num'].tolist()
            train_samples = [i for i in all_samples if 0 <= i <= 200]
            
            print(f"📋 从Labels.xls加载训练样本:")
            print(f"   训练样本范围: 0-200")
            print(f"   实际可用样本: {len(train_samples)} 个")
            
            return train_samples
        except Exception as e:
            print(f"❌ 加载Labels.xls失败: {e}")
            print("⚠️  使用默认样本范围 0-20")
            return list(range(21))
    
    train_samples = load_train_samples()
    print(f"📊 使用QAS目录中的{len(train_samples)}个样本")
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU数量: {torch.cuda.device_count()}")
        print(f"🔧 当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 使用新的数据加载器
    print("\n📥 加载预计算数据...")
    try:
        # 创建数据集
        dataset = TransformerBatteryDataset(data_path='/mnt/bz25t/bzhy/zhanglikang/project/QAS', sample_ids=train_samples)
        
        if len(dataset) == 0:
            print("❌ 没有加载到任何训练数据")
            print("请确保已运行 precompute_targets.py 生成预计算数据")
            return
        
        print(f"✅ 成功加载 {len(dataset)} 个训练数据对")
        
        # 创建数据加载器
        BATCH_SIZE = 4000  # 从2000增加到4000
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=4, pin_memory=True)
        print(f"📦 数据加载器创建完成，批次大小: {BATCH_SIZE}, num_workers: 4")
        
        # 显示数据统计
        sample_input, sample_target = dataset[0]
        print(f"📊 数据格式:")
        print(f"   输入维度: {sample_input.shape} (前5维vin_1 + 电压 + SOC)")
        print(f"   目标维度: {sample_target.shape} (下一时刻电压 + SOC)")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保已运行 precompute_targets.py 生成预计算数据")
        return
    
    #----------------------------------------模型初始化------------------------------
    # 初始化Transformer模型
    transformer = TransformerPredictor(
        input_size=7,      # vin_1前5维 + 电压 + SOC
        d_model=128,       # 模型维度
        nhead=8,           # 注意力头数
        num_layers=3,      # Transformer层数
        output_size=2      # 输出：电压 + SOC
    ).to(device).float()
    
    # 启用数据并行
    if torch.cuda.device_count() > 1:
        transformer = torch.nn.DataParallel(transformer)
        print(f"✅ 启用数据并行，使用 {torch.cuda.device_count()} 张GPU")
    else:
        print("⚠️  单GPU模式")
    
    print(f"🧠 Transformer模型初始化完成")
    print(f"📈 模型参数量: {sum(p.numel() for p in transformer.parameters()):,}")
    
    #----------------------------------------训练参数设置------------------------------
    LR = 1.5e-3            # 学习率从1e-3增加到1.5e-3
    EPOCH = 40             # 训练轮数从30增加到40
    lr_decay_freq = 15     # 学习率衰减频率从10增加到15
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=0.9)
    criterion = nn.MSELoss()
    
    # 设置混合精度训练
    scaler = setup_mixed_precision()
    
    print(f"⚙️  训练参数（保守优化版本）:")
    print(f"   学习率: {LR} (从1e-3增加到1.5e-3)")
    print(f"   训练轮数: {EPOCH} (从30增加到40)")
    print(f"   批次大小: {BATCH_SIZE} (从2000增加到4000)")
    print(f"   学习率衰减频率: {lr_decay_freq} (从10增加到15)")
    print(f"   预测批次大小: 15000 (从10000增加到15000)")
    print(f"   MC-AE批次大小: 6000 (从3000增加到6000)")
    print(f"   MC-AE训练轮数: 250 (从300减少到250)")
    print(f"   MC-AE学习率: 7e-4 (从5e-4增加到7e-4)")
    print(f"   混合精度训练: 启用")
    print(f"   DataLoader优化: num_workers=4, pin_memory=True")
    
    #----------------------------------------开始训练------------------------------
    print("\n" + "="*60)
    print("🎯 开始Transformer训练")
    print("="*60)
    
    transformer.train()
    train_losses = []
    
    for epoch in range(EPOCH):
        epoch_loss = 0
        batch_count = 0
        
        for batch_input, batch_target in train_loader:
            # 数据移到设备
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                pred_output = transformer(batch_input)
                loss = criterion(pred_output, batch_target)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # 打印训练进度
        if epoch % 5 == 0 or epoch == EPOCH - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}')
    
    print("\n✅ Transformer训练完成!")
    
    #----------------------------------------训练结果分析------------------------------
    print("\n" + "="*60)
    print("📊 训练结果分析")
    print("="*60)
    
    # 最终损失
    final_loss = train_losses[-1]
    print(f"🎯 最终训练损失: {final_loss:.6f}")
    
    # 损失改善
    initial_loss = train_losses[0]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    print(f"📈 损失改善: {improvement:.2f}% (从 {initial_loss:.6f} 到 {final_loss:.6f})")
    
    # 评估模型性能
    transformer.eval()
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        voltage_errors = []
        soc_errors = []
        
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            pred_output = transformer(batch_input)
            
            # 计算总损失
            loss = criterion(pred_output, batch_target)
            total_loss += loss.item()
            total_samples += batch_input.size(0)
            
            # 分别计算电压和SOC误差
            voltage_error = torch.abs(pred_output[:, 0] - batch_target[:, 0])
            soc_error = torch.abs(pred_output[:, 1] - batch_target[:, 1])
            
            voltage_errors.extend(voltage_error.cpu().numpy())
            soc_errors.extend(soc_error.cpu().numpy())
        
        avg_test_loss = total_loss / len(train_loader)
        avg_voltage_error = np.mean(voltage_errors)
        avg_soc_error = np.mean(soc_errors)
        
        print(f"🔍 模型评估结果:")
        print(f"   平均测试损失: {avg_test_loss:.6f}")
        print(f"   平均电压误差: {avg_voltage_error:.4f} V")
        print(f"   平均SOC误差: {avg_soc_error:.4f}")
        print(f"   电压误差标准差: {np.std(voltage_errors):.4f} V")
        print(f"   SOC误差标准差: {np.std(soc_errors):.4f}")
    
    #----------------------------------------保存模型------------------------------
    print("\n" + "="*60)
    print("💾 保存模型")
    print("="*60)
    
    # 确保models目录存在
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存Transformer模型
    model_path = 'models/transformer_model.pth'
    torch.save(transformer.state_dict(), model_path)
    print(f"✅ Transformer模型已保存: {model_path}")
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'final_loss': final_loss,
        'avg_voltage_error': avg_voltage_error,
        'avg_soc_error': avg_soc_error,
        'training_samples': train_samples,
        'model_params': {
            'input_size': 7,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'output_size': 2
        }
    }
    
    history_path = 'models/transformer_training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    print(f"✅ 训练历史已保存: {history_path}")
    
    #----------------------------------------绘制训练曲线------------------------------
    print("\n📈 绘制训练曲线...")
    
    plt.figure(figsize=(12, 4))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Transformer Training Loss / Transformer训练损失曲线', fontsize=14)
    plt.xlabel('Training Epochs / 训练轮数', fontsize=12)
    plt.ylabel('MSE Loss / MSE损失', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标
    
    # 误差分布
    plt.subplot(1, 2, 2)
    plt.hist(voltage_errors, bins=50, alpha=0.7, label=f'Voltage Error (Mean: {avg_voltage_error:.4f}V) / 电压误差 (均值: {avg_voltage_error:.4f}V)', color='red')
    plt.hist(soc_errors, bins=50, alpha=0.7, label=f'SOC Error (Mean: {avg_soc_error:.4f}) / SOC误差 (均值: {avg_soc_error:.4f})', color='blue')
    plt.title('Prediction Error Distribution / 预测误差分布', fontsize=14)
    plt.xlabel('Absolute Error / 绝对误差', fontsize=12)
    plt.ylabel('Frequency / 频次', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'models/transformer_training_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 训练结果图已保存: {plot_path}")
    
    #----------------------------------------训练完成总结------------------------------
    print("\n" + "="*60)
    print("🎉 Transformer训练完成总结")
    print("="*60)
    print("✅ 主要成果:")
    print("   1. 使用预计算的真实Terminal Voltage和Pack SOC数据")
    print("   2. 7维输入 → 2维物理值输出，无需数值转换")
    print("   3. 直接学习物理量之间的时序关系")
    print("   4. 模型和训练历史已保存到models/目录")
    print("")
    print("📊 性能指标:")
    print(f"   最终训练损失: {final_loss:.6f}")
    print(f"   平均电压预测误差: {avg_voltage_error:.4f} V")
    print(f"   平均SOC预测误差: {avg_soc_error:.4f}")
    print("")
    print("🔄 下一步:")
    print("   1. 可以运行Test_combine.py进行性能对比")
    print("   2. 检查预测结果的物理合理性")
    print("   3. 与BiLSTM基准进行详细对比")
    
    #----------------------------------------阶段2: 加载vin_2和vin_3数据，进行Transformer预测替换------------------------
    print("\n" + "="*60)
    print("🔄 阶段2: 加载vin_2和vin_3数据，进行Transformer预测替换")
    print("="*60)
    
    # 加载所有训练样本的vin_2和vin_3数据
    all_vin1_data = []
    all_vin2_data = []
    all_vin3_data = []
    
    print("📥 加载原始vin_2和vin_3数据...")
    processed_count = 0
    failed_count = 0
    
    for sample_id in train_samples:
        try:
            # 加载vin_1数据（用于Transformer预测）
            vin1_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_1.pkl'
            with open(vin1_path, 'rb') as file:
                vin1_data = pickle.load(file)
                if isinstance(vin1_data, torch.Tensor):
                    vin1_data = vin1_data.cpu()
                else:
                    vin1_data = torch.tensor(vin1_data)
                all_vin1_data.append(vin1_data)
            
            # 加载vin_2数据并进行静默处理
            vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
            with open(vin2_path, 'rb') as file:
                vin2_data = pickle.load(file)
            
            # 基于物理约束的数据处理（静默模式）
            vin2_processed = physics_based_data_processing_silent(vin2_data, feature_type='vin2')
            all_vin2_data.append(vin2_processed)
            
            # 加载vin_3数据并进行静默处理
            vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
            with open(vin3_path, 'rb') as file:
                vin3_data = pickle.load(file)
            
            # 基于物理约束的数据处理（静默模式）
            vin3_processed = physics_based_data_processing_silent(vin3_data, feature_type='vin3')
            all_vin3_data.append(vin3_processed)
            
            processed_count += 1
            
            # 每10个样本显示一次进度
            if processed_count % 10 == 0:
                print(f"📊 已处理 {processed_count}/{len(train_samples)} 个样本")
                
        except Exception as e:
            print(f"❌ 样本 {sample_id} 处理失败: {e}")
            failed_count += 1
            continue
    
    # 显示汇总信息
    print(f"\n✅ 数据加载完成:")
    print(f"   总样本数: {len(train_samples)}")
    print(f"   成功处理: {processed_count}")
    print(f"   处理失败: {failed_count}")
    print(f"   成功率: {processed_count/len(train_samples)*100:.1f}%")
    
    # 合并所有数据
    combined_vin1 = torch.cat(all_vin1_data, dim=0).float()
    combined_vin2 = torch.cat(all_vin2_data, dim=0).float()
    combined_vin3 = torch.cat(all_vin3_data, dim=0).float()
    
    print(f"📊 合并后数据形状:")
    print(f"   vin_1: {combined_vin1.shape}")
    print(f"   vin_2: {combined_vin2.shape}")
    print(f"   vin_3: {combined_vin3.shape}")
    
    # 合并后的数据质量检查（简化版）
    print("\n🔍 合并后数据质量检查:")
    print(f"   vin_2 NaN数量: {torch.isnan(combined_vin2).sum()}")
    print(f"   vin_2 Inf数量: {torch.isinf(combined_vin2).sum()}")
    print(f"   vin_3 NaN数量: {torch.isnan(combined_vin3).sum()}")
    print(f"   vin_3 Inf数量: {torch.isinf(combined_vin3).sum()}")
    
    # 检查是否有异常值需要处理
    vin2_has_issues = (torch.isnan(combined_vin2).any() or 
                       torch.isinf(combined_vin2).any() or 
                       combined_vin2.min() < -1e6 or 
                       combined_vin2.max() > 1e6)
    
    vin3_has_issues = (torch.isnan(combined_vin3).any() or 
                       torch.isinf(combined_vin3).any() or 
                       combined_vin3.min() < -1e6 or 
                       combined_vin3.max() > 1e6)
    
    if vin2_has_issues or vin3_has_issues:
        print("\n⚠️  检测到数据问题，进行修复...")
        
        # 修复NaN和Inf值
        if torch.isnan(combined_vin2).any() or torch.isinf(combined_vin2).any():
            print("   修复vin_2中的NaN和Inf值")
            combined_vin2 = torch.where(torch.isnan(combined_vin2) | torch.isinf(combined_vin2), 
                                       torch.zeros_like(combined_vin2), combined_vin2)
        
        if torch.isnan(combined_vin3).any() or torch.isinf(combined_vin3).any():
            print("   修复vin_3中的NaN和Inf值")
            combined_vin3 = torch.where(torch.isnan(combined_vin3) | torch.isinf(combined_vin3), 
                                        torch.zeros_like(combined_vin3), combined_vin3)
        
        # 检查修复后的数据
        print("\n🔍 修复后数据质量检查:")
        print(f"   vin_2 NaN数量: {torch.isnan(combined_vin2).sum()}")
        print(f"   vin_2 Inf数量: {torch.isinf(combined_vin2).sum()}")
        print(f"   vin_3 NaN数量: {torch.isnan(combined_vin3).sum()}")
        print(f"   vin_3 Inf数量: {torch.isinf(combined_vin3).sum()}")
    else:
        print("\n✅ 数据质量良好，无需修复")
    
    # 使用Transformer进行预测和替换
    print("\n🔄 使用Transformer预测替换vin_2[:,0]和vin_3[:,0]...")
    transformer.eval()
    
    # 预先加载所有targets数据（优化I/O）
    print("📥 预先加载所有targets数据...")
    all_targets = {}
    for sample_id in train_samples:
        targets_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/targets.pkl'
        with open(targets_path, 'rb') as f:
            all_targets[sample_id] = pickle.load(f)
    print(f"✅ 已加载 {len(all_targets)} 个样本的targets数据")
    
    # 批量构建Transformer输入数据
    print("🔧 批量构建Transformer输入数据...")
    transformer_inputs = torch.zeros(len(combined_vin1), 7, dtype=torch.float32)
    
    # 计算每个样本的起始和结束索引
    sample_indices = []
    current_idx = 0
    for sample_idx, sample_id in enumerate(train_samples):
        sample_len = len(all_vin1_data[sample_idx])
        sample_indices.append((current_idx, current_idx + sample_len, sample_id))
        current_idx += sample_len
    
    # 批量填充输入数据
    for start_idx, end_idx, sample_id in sample_indices:
        targets = all_targets[sample_id]
        terminal_voltages = np.array(targets['terminal_voltages'])
        pack_socs = np.array(targets['pack_socs'])
        
        # 填充vin_1前5维
        transformer_inputs[start_idx:end_idx, 0:5] = combined_vin1[start_idx:end_idx, 0, 0:5]
        # 填充当前时刻真实电压
        transformer_inputs[start_idx:end_idx, 5] = torch.tensor(terminal_voltages[:end_idx-start_idx], dtype=torch.float32)
        # 填充当前时刻真实SOC
        transformer_inputs[start_idx:end_idx, 6] = torch.tensor(pack_socs[:end_idx-start_idx], dtype=torch.float32)
    
    print(f"✅ 输入数据构建完成: {transformer_inputs.shape}")
    
    # 显示GPU内存使用情况
    print("📊 GPU内存使用情况:")
    print_gpu_memory()
    
    # 批量预测（使用更大的批次大小）
    print("🚀 开始批量预测...")
    transformer_inputs = transformer_inputs.to(device)
    
    with torch.no_grad():
        # 使用更大的批次大小以提高GPU利用率
        batch_size = 15000  # 从10000增加到15000
        predictions = []
        total_batches = (len(transformer_inputs) + batch_size - 1) // batch_size
        
        for i in range(0, len(transformer_inputs), batch_size):
            batch_idx = i // batch_size + 1
            batch_data = transformer_inputs[i:i+batch_size]
            batch_pred = transformer(batch_data)
            predictions.append(batch_pred.cpu())
            
            # 显示进度
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"   进度: {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.1f}%)")
        
        transformer_predictions = torch.cat(predictions, dim=0)
    
    print(f"✅ Transformer预测完成: {transformer_predictions.shape}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU内存已清理，当前使用: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # 替换vin_2[:,0]和vin_3[:,0]
    vin2_modified = combined_vin2.clone()
    vin3_modified = combined_vin3.clone()
    
    # 确保长度匹配
    min_len = min(len(transformer_predictions), len(vin2_modified), len(vin3_modified))
    
    # 替换BiLSTM预测值为Transformer预测值
    vin2_modified[:min_len, 0] = transformer_predictions[:min_len, 0]  # 电压预测
    vin3_modified[:min_len, 0] = transformer_predictions[:min_len, 1]  # SOC预测
    
    print(f"🔄 替换完成:")
    print(f"   原始vin_2[:,0]范围: [{combined_vin2[:, 0].min():.4f}, {combined_vin2[:, 0].max():.4f}]")
    print(f"   Transformer vin_2[:,0]范围: [{vin2_modified[:, 0].min():.4f}, {vin2_modified[:, 0].max():.4f}]")
    print(f"   原始vin_3[:,0]范围: [{combined_vin3[:, 0].min():.4f}, {combined_vin3[:, 0].max():.4f}]")
    print(f"   Transformer vin_3[:,0]范围: [{vin3_modified[:, 0].min():.4f}, {vin3_modified[:, 0].max():.4f}]")
    
    #----------------------------------------阶段3: 训练MC-AE异常检测模型------------------------
    print("\n" + "="*60)
    print("🧠 阶段3: 训练MC-AE异常检测模型（使用Transformer增强数据）")
    print("="*60)
    
    # 参考Train_BILSTM.py的MC-AE训练逻辑
    from Function_ import custom_activation
    
    # 定义特征切片维度（与Train_BILSTM.py一致）
    # vin_2.pkl
    dim_x = 2
    dim_y = 110
    dim_z = 110
    dim_q = 3
    
    # 分割vin_2特征张量
    x_recovered = vin2_modified[:, :dim_x]
    y_recovered = vin2_modified[:, dim_x:dim_x + dim_y]
    z_recovered = vin2_modified[:, dim_x + dim_y: dim_x + dim_y + dim_z]
    q_recovered = vin2_modified[:, dim_x + dim_y + dim_z:]
    
    # vin_3.pkl
    dim_x2 = 2
    dim_y2 = 110
    dim_z2 = 110
    dim_q2 = 4
    
    x_recovered2 = vin3_modified[:, :dim_x2]
    y_recovered2 = vin3_modified[:, dim_x2:dim_x2 + dim_y2]
    z_recovered2 = vin3_modified[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
    q_recovered2 = vin3_modified[:, dim_x2 + dim_y2 + dim_z2:]
    
    print(f"📊 MC-AE训练数据准备:")
    print(f"   vin_2特征: x{x_recovered.shape}, y{y_recovered.shape}, z{z_recovered.shape}, q{q_recovered.shape}")
    print(f"   vin_3特征: x{x_recovered2.shape}, y{y_recovered2.shape}, z{z_recovered2.shape}, q{q_recovered2.shape}")
    
    # MC-AE训练参数（保守优化）
    EPOCH_MCAE = 250       # 从300减少到250
    LR_MCAE = 7e-4         # 从5e-4增加到7e-4
    BATCHSIZE_MCAE = 6000  # 从3000增加到6000
    
    # 自定义多输入数据集类
    class MCDataset(Dataset):
        def __init__(self, x, y, z, q):
            self.x = x.to(torch.double)
            self.y = y.to(torch.double)
            self.z = z.to(torch.double)
            self.q = q.to(torch.double)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.z[idx], self.q[idx]
    
    # 第一组特征（vin_2）的MC-AE训练
    print("\n🔧 训练第一组MC-AE模型（vin_2）...")
    train_loader_u = DataLoader(MCDataset(x_recovered, y_recovered, z_recovered, q_recovered), 
                               batch_size=BATCHSIZE_MCAE, shuffle=False, 
                               num_workers=4, pin_memory=True)
    
    net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, 
                    activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    
    optimizer_mcae = torch.optim.Adam(net.parameters(), lr=LR_MCAE)
    loss_f = nn.MSELoss()
    
    # 记录训练损失
    train_losses_mcae1 = []
    
    for epoch in range(EPOCH_MCAE):
        total_loss = 0
        num_batches = 0
        for iteration, (x, y, z, q) in enumerate(train_loader_u):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            q = q.to(device)
            net = net.double()
            recon_im, recon_p = net(x, z, q)
            loss_u = loss_f(y, recon_im)
            total_loss += loss_u.item()
            num_batches += 1
            optimizer_mcae.zero_grad()
            loss_u.backward()
            optimizer_mcae.step()
        avg_loss = total_loss / num_batches
        train_losses_mcae1.append(avg_loss)
        if epoch % 50 == 0:
            print(f'MC-AE1 Epoch: {epoch:3d} | Average Loss: {avg_loss:.6f}')
    
    # 获取第一组重构误差
    train_loader2 = DataLoader(MCDataset(x_recovered, y_recovered, z_recovered, q_recovered), 
                              batch_size=len(x_recovered), shuffle=False,
                              num_workers=4, pin_memory=True)
    for iteration, (x, y, z, q) in enumerate(train_loader2):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        net = net.double()
        recon_imtest, recon = net(x, z, q)
    
    AA = recon_imtest.cpu().detach().numpy()
    yTrainU = y_recovered.cpu().detach().numpy()
    ERRORU = AA - yTrainU
    
    # 第二组特征（vin_3）的MC-AE训练
    print("\n🔧 训练第二组MC-AE模型（vin_3）...")
    train_loader_soc = DataLoader(MCDataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), 
                                 batch_size=BATCHSIZE_MCAE, shuffle=False,
                                 num_workers=4, pin_memory=True)
    
    netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                     activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
    
    optimizer_mcae2 = torch.optim.Adam(netx.parameters(), lr=LR_MCAE)
    
    # 记录训练损失
    train_losses_mcae2 = []
    
    for epoch in range(EPOCH_MCAE):
        total_loss = 0
        num_batches = 0
        for iteration, (x, y, z, q) in enumerate(train_loader_soc):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            q = q.to(device)
            netx = netx.double()
            recon_im, z = netx(x, z, q)
            loss_x = loss_f(y, recon_im)
            total_loss += loss_x.item()
            num_batches += 1
            optimizer_mcae2.zero_grad()
            loss_x.backward()
            optimizer_mcae2.step()
        avg_loss = total_loss / num_batches
        train_losses_mcae2.append(avg_loss)
        if epoch % 50 == 0:
            print(f'MC-AE2 Epoch: {epoch:3d} | Average Loss: {avg_loss:.6f}')
    
    # 获取第二组重构误差
    train_loaderx2 = DataLoader(MCDataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), 
                               batch_size=len(x_recovered2), shuffle=False,
                               num_workers=4, pin_memory=True)
    for iteration, (x, y, z, q) in enumerate(train_loaderx2):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        netx = netx.double()
        recon_imtestx, z = netx(x, z, q)
    
    BB = recon_imtestx.cpu().detach().numpy()
    yTrainX = y_recovered2.cpu().detach().numpy()
    ERRORX = BB - yTrainX
    
    # 保存中间结果，避免重新训练
    print("\n💾 保存中间结果...")
    model_suffix = "_transformer"
    
    # 确保models目录存在
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存重构误差数据
    np.save(f'models/ERRORU{model_suffix}.npy', ERRORU)
    np.save(f'models/ERRORX{model_suffix}.npy', ERRORX)
    print(f"✅ 中间结果已保存: ERRORU{model_suffix}.npy, ERRORX{model_suffix}.npy")
    
    # 保存MC-AE训练历史（用于断点续算）
    mcae_intermediate_history = {
        'train_losses_mcae1': train_losses_mcae1,
        'train_losses_mcae2': train_losses_mcae2,
        'final_mcae1_loss': train_losses_mcae1[-1] if train_losses_mcae1 else None,
        'final_mcae2_loss': train_losses_mcae2[-1] if train_losses_mcae2 else None,
        'training_samples': len(train_samples),
        'epochs': EPOCH_MCAE,
        'learning_rate': LR_MCAE,
        'batch_size': BATCHSIZE_MCAE
    }
    
    with open(f'models/mcae_intermediate_history{model_suffix}.pkl', 'wb') as f:
        pickle.dump(mcae_intermediate_history, f)
    print(f"✅ MC-AE中间训练历史已保存: mcae_intermediate_history{model_suffix}.pkl")
    
    print("✅ MC-AE训练完成!")
    
    #----------------------------------------MC-AE训练结果可视化------------------------
    print("\n📈 绘制MC-AE训练结果...")
    
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
    plot_path = f'models/transformer_mcae_training_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ MC-AE训练结果图已保存: {plot_path}")
    
    #----------------------------------------阶段4: PCA分析和保存模型------------------------
    print("\n" + "="*60)
    print("📊 阶段4: PCA分析，保存模型和参数")
    print("="*60)
    
    # 诊断特征提取与PCA分析
    from Function_ import DiagnosisFeature, PCA
    
    print("🔍 提取诊断特征...")
    df_data = DiagnosisFeature(ERRORU, ERRORX)
    print(f"   诊断特征数据形状: {df_data.shape}")
    
    print("🔍 进行PCA分析...")
    v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data, 0.95, 0.95)
    
    print(f"✅ PCA分析完成:")
    print(f"   主成分数量: {k}")
    print(f"   解释方差比: {v_ratio}")
    print(f"   T²控制限: 95%={T_95_limit:.4f}, 99%={T_99_limit:.4f}")
    print(f"   SPE控制限: 95%={SPE_95_limit:.4f}, 99%={SPE_99_limit:.4f}")
    
    # 保存所有模型和分析结果
    print("\n💾 保存Transformer增强训练结果...")
    model_suffix = "_transformer"
    
    # 1. 保存Transformer模型（已保存）
    
    # 2. 保存MC-AE模型
    torch.save(net.state_dict(), f'models/net_model{model_suffix}.pth')
    torch.save(netx.state_dict(), f'models/netx_model{model_suffix}.pth')
    print(f"✅ MC-AE模型已保存: models/net_model{model_suffix}.pth, models/netx_model{model_suffix}.pth")
    
    # 3. 保存诊断特征（分块保存，避免Excel文件过大）
    print(f"💾 保存诊断特征（数据量: {df_data.shape}）...")
    
    # CSV文件保存（无大小限制）
    csv_path = f'models/diagnosis_feature{model_suffix}.csv'
    df_data.to_csv(csv_path, index=False)
    print(f"✅ 诊断特征CSV已保存: {csv_path}")
    
    # Excel文件分块保存（避免超过Excel行数限制）
    excel_path = f'models/diagnosis_feature{model_suffix}.xlsx'
    max_rows_per_sheet = 1000000  # Excel限制约104万行，留些余量
    
    if len(df_data) > max_rows_per_sheet:
        print(f"⚠️  数据量过大({len(df_data)}行)，进行分块保存...")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 计算需要多少个工作表
            num_sheets = (len(df_data) + max_rows_per_sheet - 1) // max_rows_per_sheet
            
            for i in range(num_sheets):
                start_idx = i * max_rows_per_sheet
                end_idx = min((i + 1) * max_rows_per_sheet, len(df_data))
                chunk = df_data.iloc[start_idx:end_idx]
                
                sheet_name = f'Sheet_{i+1}' if i > 0 else 'Sheet_1'
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"   工作表 {i+1}/{num_sheets}: {start_idx+1}-{end_idx} 行")
        
        print(f"✅ 诊断特征Excel已分块保存: {excel_path} ({num_sheets}个工作表)")
    else:
        # 数据量不大，直接保存
        df_data.to_excel(excel_path, index=False)
        print(f"✅ 诊断特征Excel已保存: {excel_path}")
    
    # 4. 保存PCA分析结果
    np.save(f'models/v_I{model_suffix}.npy', v_I)
    np.save(f'models/v{model_suffix}.npy', v)
    np.save(f'models/v_ratio{model_suffix}.npy', v_ratio)
    np.save(f'models/p_k{model_suffix}.npy', p_k)
    np.save(f'models/data_mean{model_suffix}.npy', data_mean)
    np.save(f'models/data_std{model_suffix}.npy', data_std)
    np.save(f'models/T_95_limit{model_suffix}.npy', T_95_limit)
    np.save(f'models/T_99_limit{model_suffix}.npy', T_99_limit)
    np.save(f'models/SPE_95_limit{model_suffix}.npy', SPE_95_limit)
    np.save(f'models/SPE_99_limit{model_suffix}.npy', SPE_99_limit)
    np.save(f'models/P{model_suffix}.npy', P)
    np.save(f'models/k{model_suffix}.npy', k)
    np.save(f'models/P_t{model_suffix}.npy', P_t)
    np.save(f'models/X{model_suffix}.npy', X)
    np.save(f'models/data_nor{model_suffix}.npy', data_nor)
    print(f"✅ PCA分析结果已保存: models/*{model_suffix}.npy")
    
    # 5. 保存MC-AE训练历史
    mcae_training_history = {
        'mcae1_losses': train_losses_mcae1,
        'mcae2_losses': train_losses_mcae2,
        'final_mcae1_loss': train_losses_mcae1[-1],
        'final_mcae2_loss': train_losses_mcae2[-1],
        'mcae1_reconstruction_error_mean': np.mean(np.abs(ERRORU)),
        'mcae1_reconstruction_error_std': np.std(np.abs(ERRORU)),
        'mcae2_reconstruction_error_mean': np.mean(np.abs(ERRORX)),
        'mcae2_reconstruction_error_std': np.std(np.abs(ERRORX)),
        'training_samples': len(train_samples),
        'epochs': EPOCH_MCAE,
        'learning_rate': LR_MCAE,
        'batch_size': BATCHSIZE_MCAE
    }
    
    with open(f'models/transformer_mcae_training_history.pkl', 'wb') as f:
        pickle.dump(mcae_training_history, f)
    print(f"✅ MC-AE训练历史已保存: models/transformer_mcae_training_history.pkl")
    
    #----------------------------------------最终训练完成总结------------------------------
    print("\n" + "="*60)
    print("🎉 Transformer完整训练流程完成！")
    print("="*60)
    print("✅ 训练流程总结:")
    print("   1. ✅ 训练Transformer时序预测模型")
    print("   2. ✅ 使用Transformer预测替换vin_2[:,0]和vin_3[:,0]")
    print("   3. ✅ 保持Pack Modeling输出vin_2[:,1]和vin_3[:,1]不变")
    print("   4. ✅ MC-AE使用Transformer增强数据进行训练")
    print("   5. ✅ 完整的PCA分析和诊断特征提取")
    print("   6. ✅ 所有模型和结果文件添加'_transformer'后缀")
    print("   7. ✅ MC-AE训练结果可视化图表")
    print("")
    print("📊 关键改进:")
    print("   - Transformer替换BiLSTM进行时序预测")
    print("   - 直接使用真实物理值训练，无复杂转换")
    print("   - 保持与原始MC-AE训练流程完全兼容")
    print("   - 便于与BiLSTM基准进行公平对比")
    print("   - 保守优化：批次大小翻倍，启用混合精度训练")
    print("   - 双GPU数据并行，充分利用A100显存")
    print("   - DataLoader优化：num_workers=4, pin_memory=True")
    print("")
    print("📈 性能指标:")
    print(f"   Transformer电压预测误差: {avg_voltage_error:.4f} V")
    print(f"   Transformer SOC预测误差: {avg_soc_error:.4f}")
    print(f"   PCA主成分数量: {k}")
    print("")
    print("🔄 下一步可以:")
    print("   1. 运行Test_combine.py进行详细性能对比")
    print("   2. 分析Transformer vs BiLSTM的故障检测效果")
    print("   3. 检查三窗口检测机制的改进效果")

if __name__ == "__main__":
    main() 