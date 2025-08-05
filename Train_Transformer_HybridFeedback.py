# 中文注释：混合反馈策略版Transformer训练脚本 - 基于Train_Transformer.py架构
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
from Comprehensive_calculation import Comprehensive_calculation
import math
import math
from create_dataset import series_to_supervised
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
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
from scipy import ndimage
import copy
import time


# 导入Transformer数据加载器
from data_loader_transformer import TransformerBatteryDataset, create_transformer_dataloader

# 激进反馈策略配置 - 专注降低假阳率
HYBRID_FEEDBACK_CONFIG = {
    # 数据分组配置（严格按照README规范）
    'train_samples': list(range(8)),        # QAS 0-7 (8个正常样本)
    'feedback_samples': [8, 9],             # QAS 8-9 (2个正常反馈样本)
    
    # 激进反馈机制配置
    'feedback_frequency': 3,                # 每3个epoch检查一次（大幅提高）
    'use_feedback': True,                   # 启用反馈机制
    'feedback_start_epoch': 30,             # 第30轮就开始启用反馈（提前介入）
    
    # 极严格的反馈触发阈值（目标：正常样本接近0%假阳率）
    'false_positive_thresholds': {
        'warning': 0.001,       # 0.1%预警（极低阈值）
        'standard': 0.002,      # 0.2%标准反馈（激进）
        'enhanced': 0.005,      # 0.5%强化反馈（激进）
        'emergency': 0.01       # 1%紧急反馈（原来的标准）
    },
    
    # 激进的混合权重配置
    'mcae_weight': 0.6,                     # 降低MC-AE权重
    'transformer_weight': 0.4,             # 提高Transformer权重（增强预测精度）
    
    # 激进的自适应学习率配置
    'adaptive_lr_factors': {
        'standard': 0.5,        # 标准反馈：LR * 0.5（激进调整）
        'enhanced': 0.3,        # 强化反馈：LR * 0.3（激进调整）
        'emergency': 0.1        # 紧急反馈：LR * 0.1（极激进调整）
    },
    
    # 新增：动态反馈强度配置
    'dynamic_feedback_weights': {
        'min_feedback_weight': 0.1,        # 最小反馈权重
        'max_feedback_weight': 2.0,        # 最大反馈权重（可超过基础训练）
        'weight_increment': 0.2,           # 每次反馈增强幅度
        'consecutive_trigger_boost': 1.5   # 连续触发时的权重提升倍数
    },
    
    # 新增：正常样本特化训练配置（基于阈值相对优化）
    'normal_sample_focus': {
        'enable': True,                     # 启用正常样本特化训练
        'focus_weight': 3.0,               # 正常样本的损失权重倍数
        'threshold_margin': 0.8,           # 目标：FAI < threshold1 * 0.8 (保持20%安全边距)
        'relative_penalty': True,          # 启用相对阈值惩罚
        'penalty_factor': 5.0              # 超出目标阈值的惩罚因子
    }
}

# 复用Train_Transformer.py的内存监控和混合精度设置
def print_gpu_memory():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.1f}GB / {cached:.1f}GB / {total:.1f}GB (已用/缓存/总计)")

def setup_mixed_precision():
    """设置混合精度训练"""
    scaler = torch.cuda.amp.GradScaler()
    print("✅ 启用混合精度训练 (AMP)")
    return scaler

# 复用Train_Transformer.py的数据处理函数
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

# 复用Train_Transformer.py的数据预处理（重要：保持完全一致）
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

# GPU配置优化：小样本训练使用单GPU避免跨卡通信开销
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用GPU0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 打印GPU信息
if torch.cuda.is_available():
    print(f"\n🖥️ 单GPU优化配置:")
    print(f"   可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i} ({props.name}): {props.total_memory/1024**3:.1f}GB")
    print(f"   主GPU设备: cuda:0 (物理GPU0)")
    print(f"   优化模式: 小样本训练，避免跨卡通信开销")
else:
    print("⚠️  未检测到GPU，使用CPU训练")

# 忽略警告信息
warnings.filterwarnings('ignore')

# Linux环境matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Linux环境字体设置 - 修复中文显示问题
import matplotlib.font_manager as fm
import os

# 更全面的字体检测和设置
def setup_chinese_fonts():
    """设置中文字体，如果不可用则使用英文"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
        'Noto Sans CJK JP', 'Noto Sans CJK TC', 'Source Han Sans CN',
        'Droid Sans Fallback', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
    ]
    
    # 检查系统字体
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"🔍 系统可用字体数量: {len(system_fonts)}")
    
    # 查找可用的中文字体
    available_chinese = []
    for font in chinese_fonts:
        if font in system_fonts:
            available_chinese.append(font)
            print(f"✅ 找到中文字体: {font}")
    
    if available_chinese:
        # 使用第一个可用的中文字体
        plt.rcParams['font.sans-serif'] = available_chinese
        plt.rcParams['axes.unicode_minus'] = False
        print(f"🎨 使用中文字体: {available_chinese[0]}")
        return True
    else:
        # 没有中文字体，使用英文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️  未找到中文字体，将使用英文标签")
        return False

# 设置字体
use_chinese = setup_chinese_fonts()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

#----------------------------------------重要说明：混合反馈策略核心架构------------------------------
# 混合反馈策略架构：
# 
# 阶段1: 基础Transformer训练 (样本0-7, epoch 0-20)
# - 使用QAS 0-7训练样本进行标准Transformer训练
# - 不启用反馈机制，建立基础预测能力
#
# 阶段2: MC-AE训练 (使用Transformer增强数据)
# - 使用Transformer预测替换vin_2[:,0]和vin_3[:,0]
# - 训练MC-AE异常检测模型
#
# 阶段3: 混合反馈训练 (样本8-9, epoch 21-40)
# - 使用QAS 8-9反馈样本进行假阳性检测
# - 实时监控假阳性率，触发多级反馈机制
# - 自适应调整训练策略和学习率
#
# 阶段4: PCA分析和模型保存
# - 使用Transformer增强数据训练MC-AE
# - 进行PCA分析，保存模型和参数

#----------------------------------------复用Train_Transformer.py的TransformerPredictor模型------------------------------
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

#----------------------------------------混合反馈策略核心函数------------------------------
def calculate_comprehensive_fault_indicator(sample_data, mcae_net1, mcae_net2, pca_params, device):
    """
    计算综合故障指示器fai（基于Comprehensive_calculation）
    
    参数:
        sample_data: 样本数据 (vin_2, vin_3)
        mcae_net1, mcae_net2: 训练好的MC-AE模型
        pca_params: PCA参数字典
        device: 计算设备
    
    返回:
        fai: 综合故障指示器数组
    """
    mcae_net1.eval()
    mcae_net2.eval()
    
    with torch.no_grad():
        # 1. 准备MC-AE输入数据
        vin2_data, vin3_data = sample_data
        
        # 确保数据是tensor并移动到正确设备
        if not isinstance(vin2_data, torch.Tensor):
            vin2_data = torch.tensor(vin2_data, dtype=torch.float32)
        if not isinstance(vin3_data, torch.Tensor):
            vin3_data = torch.tensor(vin3_data, dtype=torch.float32)
        
        vin2_data = vin2_data.to(device)
        vin3_data = vin3_data.to(device)
        
        # 2. 分割特征（与训练时保持一致）
        dim_x, dim_y, dim_z = 2, 110, 110
        x_recovered = vin2_data[:, :dim_x]
        y_recovered = vin2_data[:, dim_x:dim_x + dim_y] 
        z_recovered = vin2_data[:, dim_x + dim_y: dim_x + dim_y + dim_z]
        q_recovered = vin2_data[:, dim_x + dim_y + dim_z:]
        
        dim_x2, dim_y2, dim_z2 = 2, 110, 110
        x_recovered2 = vin3_data[:, :dim_x2]
        y_recovered2 = vin3_data[:, dim_x2:dim_x2 + dim_y2]
        z_recovered2 = vin3_data[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
        q_recovered2 = vin3_data[:, dim_x2 + dim_y2 + dim_z2:]
        
        # 3. MC-AE重构（确保所有数据在同一设备上）
        recon_im1, _ = mcae_net1(x_recovered.double(), z_recovered.double(), q_recovered.double())
        recon_im2, _ = mcae_net2(x_recovered2.double(), z_recovered2.double(), q_recovered2.double())
        
        # 4. 计算重构误差（在CPU上进行numpy操作）
        ERRORU = recon_im1.cpu().detach().numpy() - y_recovered.cpu().detach().numpy()
        ERRORX = recon_im2.cpu().detach().numpy() - y_recovered2.cpu().detach().numpy()
        
        # 5. 诊断特征提取（复用Function_.py）
        df_data = DiagnosisFeature(ERRORU, ERRORX)
        
        # 6. 综合诊断计算（复用Comprehensive_calculation.py）
        time = np.arange(df_data.shape[0])
        
        try:
            # 调用综合计算函数
            lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std = Comprehensive_calculation(
                df_data.values,
                pca_params['data_mean'],
                pca_params['data_std'], 
                pca_params['v'].reshape(len(pca_params['v']), 1),
                pca_params['p_k'],
                pca_params['v_I'],
                pca_params['T_99_limit'],
                pca_params['SPE_99_limit'],
                pca_params['P'],
                time
            )
            
            # 按照源代码方式重新计算阈值（与Test_.py保持一致）
            nm = 3000  # 固定值，与源代码一致
            mm = len(fai)  # 数据总长度
            
            # 确保数据长度足够
            if mm > nm:
                # 使用后半段数据计算阈值
                threshold1 = np.mean(fai[nm:mm]) + 3*np.std(fai[nm:mm])
                threshold2 = np.mean(fai[nm:mm]) + 4.5*np.std(fai[nm:mm])
                threshold3 = np.mean(fai[nm:mm]) + 6*np.std(fai[nm:mm])
            else:
                # 数据太短，使用全部数据
                threshold1 = np.mean(fai) + 3*np.std(fai)
                threshold2 = np.mean(fai) + 4.5*np.std(fai)
                threshold3 = np.mean(fai) + 6*np.std(fai)
            
            return fai, threshold1, threshold2, threshold3
            
        except Exception as e:
            print(f"   ⚠️ 综合诊断计算失败: {e}")
            # 返回基于重构误差的简单指标作为后备
            simple_fai = np.mean(np.abs(ERRORU), axis=1) + np.mean(np.abs(ERRORX), axis=1)
            # 使用简单统计方法计算后备阈值（仿照源代码方式）
            nm = 3000
            mm = len(simple_fai)
            if mm > nm:
                default_threshold1 = np.mean(simple_fai[nm:mm]) + 3*np.std(simple_fai[nm:mm])
                default_threshold2 = np.mean(simple_fai[nm:mm]) + 4.5*np.std(simple_fai[nm:mm])
                default_threshold3 = np.mean(simple_fai[nm:mm]) + 6*np.std(simple_fai[nm:mm])
            else:
                default_threshold1 = np.mean(simple_fai) + 3*np.std(simple_fai)
                default_threshold2 = np.mean(simple_fai) + 4.5*np.std(simple_fai)
                default_threshold3 = np.mean(simple_fai) + 6*np.std(simple_fai)
            return simple_fai, default_threshold1, default_threshold2, default_threshold3

def calculate_training_threshold(train_samples, mcae_net1, mcae_net2, pca_params, device):
    """
    基于训练样本计算故障检测阈值（按照测试脚本的方法）
    
    参数:
        train_samples: 训练样本ID列表
        mcae_net1, mcae_net2: 训练好的MC-AE模型
        pca_params: PCA参数字典
        device: 计算设备
    
    返回:
        threshold1, threshold2, threshold3: 三级阈值
    """
    print("🔧 计算训练阶段故障检测阈值...")
    
    all_training_fai = []
    
    for sample_id in train_samples:
        try:
            # 加载样本数据
            vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
            vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
            
            with open(vin2_path, 'rb') as f:
                vin2_data = pickle.load(f)
            with open(vin3_path, 'rb') as f:
                vin3_data = pickle.load(f)
            
            # 数据预处理
            vin2_processed = physics_based_data_processing_silent(vin2_data, feature_type='vin2')
            vin3_processed = physics_based_data_processing_silent(vin3_data, feature_type='vin3')
            
            # 计算该样本的综合故障指示器
            sample_data = (vin2_processed, vin3_processed)
            fai, _, _, _ = calculate_comprehensive_fault_indicator(sample_data, mcae_net1, mcae_net2, pca_params, device)
            
            all_training_fai.extend(fai)
            
            if (len(all_training_fai) // 1000) > ((len(all_training_fai) - len(fai)) // 1000):
                print(f"   已处理 {len(all_training_fai)} 个数据点")
                
        except Exception as e:
            print(f"   ❌ 样本 {sample_id} 处理失败: {e}")
            continue
    
    all_training_fai = np.array(all_training_fai)
    print(f"   训练数据总计: {len(all_training_fai)} 个数据点")
    
    # 检查数据是否为空
    if len(all_training_fai) == 0:
        print("   ❌ 没有成功处理任何训练数据，使用默认阈值")
        # 使用默认阈值
        default_threshold = 1.0
        return default_threshold, default_threshold * 1.5, default_threshold * 2.0
    
    # 检查数据是否包含NaN或Inf
    if np.any(np.isnan(all_training_fai)) or np.any(np.isinf(all_training_fai)):
        print("   ⚠️ 检测到NaN或Inf值，清理数据...")
        all_training_fai = all_training_fai[~np.isnan(all_training_fai)]
        all_training_fai = all_training_fai[~np.isinf(all_training_fai)]
        print(f"   清理后数据点: {len(all_training_fai)}")
        
        if len(all_training_fai) == 0:
            print("   ❌ 清理后没有有效数据，使用默认阈值")
            default_threshold = 1.0
            return default_threshold, default_threshold * 1.5, default_threshold * 2.0
    
    # 按照测试脚本的方法计算阈值
    nm = 3000  # 固定分割点
    mm = len(all_training_fai)
    
    if mm > nm:
        # 使用后半段数据计算基准统计量
        fai_baseline = all_training_fai[nm:mm]
        print(f"   使用后半段数据 ({nm}:{mm}) 计算阈值")
    else:
        # 数据不足，使用全部数据
        fai_baseline = all_training_fai
        print(f"   ⚠️ 数据长度({mm})不足{nm}，使用全部数据计算阈值")
    
    # 计算三级阈值
    fai_mean = np.mean(fai_baseline)
    fai_std = np.std(fai_baseline)
    
    # 检查统计量是否有效
    if np.isnan(fai_mean) or np.isnan(fai_std) or fai_std == 0:
        print("   ⚠️ 统计量无效，使用数据范围计算阈值")
        fai_range = np.max(fai_baseline) - np.min(fai_baseline)
        fai_mean = np.median(fai_baseline)
        fai_std = fai_range / 6.0  # 使用范围估计标准差
    
    threshold1 = fai_mean + 3 * fai_std      # 3σ
    threshold2 = fai_mean + 4.5 * fai_std    # 4.5σ  
    threshold3 = fai_mean + 6 * fai_std      # 6σ
    
    print(f"   阈值统计: 均值={fai_mean:.4f}, 标准差={fai_std:.4f}")
    print(f"   计算得到阈值: T1={threshold1:.4f}, T2={threshold2:.4f}, T3={threshold3:.4f}")
    
    return threshold1, threshold2, threshold3

def calculate_false_positive_rate_comprehensive(feedback_samples, mcae_net1, mcae_net2, 
                                              pca_params, threshold, device):
    """
    基于综合诊断指标计算假阳性率
    
    参数:
        feedback_samples: 反馈样本ID列表（已知正常样本）
        mcae_net1, mcae_net2: 训练好的MC-AE模型
        pca_params: PCA参数字典
        threshold: 故障检测阈值
        device: 计算设备
    
    返回:
        false_positive_rate: 假阳性率
        false_positives: 假阳性数量
        total_normals: 总正常样本数
    """
    print(f"🔍 计算反馈样本 {feedback_samples} 的假阳性率...")
    
    all_fai = []
    
    for sample_id in feedback_samples:  # [8, 9] 都是正常样本
        try:
            # 加载样本数据
            vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
            vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
            
            with open(vin2_path, 'rb') as f:
                vin2_data = pickle.load(f)
            with open(vin3_path, 'rb') as f:
                vin3_data = pickle.load(f)
            
            # 数据预处理
            vin2_processed = physics_based_data_processing_silent(vin2_data, feature_type='vin2')
            vin3_processed = physics_based_data_processing_silent(vin3_data, feature_type='vin3')
            
            # 计算该样本的综合故障指示器
            sample_data = (vin2_processed, vin3_processed)
            fai, _, _, _ = calculate_comprehensive_fault_indicator(sample_data, mcae_net1, mcae_net2, pca_params, device)
            
            all_fai.extend(fai)
            print(f"   样本{sample_id}: {len(fai)}个数据点")
            
        except Exception as e:
            print(f"   ❌ 反馈样本 {sample_id} 处理失败: {e}")
            continue
    
    if len(all_fai) == 0:
        print("   ❌ 没有成功加载任何反馈样本数据")
        return 0.0, 0, 0
    
    all_fai = np.array(all_fai)
    
    # 计算假阳性率：正常样本中被误判为故障的比例
    false_positives = (all_fai > threshold).sum()
    total_normals = len(all_fai)
    false_positive_rate = false_positives / total_normals
    
    print(f"   反馈样本总计: {total_normals} 个数据点")
    print(f"   超过阈值({threshold:.4f}): {false_positives} 个")
    print(f"   假阳性率: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    
    return false_positive_rate, false_positives, total_normals

def detect_feedback_trigger(false_positive_rate, epoch, config, consecutive_triggers=0):
    """
    激进反馈触发检测（专注降低假阳率）
    
    参数:
        false_positive_rate: 当前假阳性率
        epoch: 当前训练轮数
        config: 反馈配置
        consecutive_triggers: 连续触发次数
    
    返回:
        trigger_level: 触发等级 ('none', 'warning', 'standard', 'enhanced', 'emergency')
        lr_factor: 学习率调整因子
        feedback_weight: 反馈权重
    """
    thresholds = config['false_positive_thresholds']
    dynamic_config = config['dynamic_feedback_weights']
    
    # 动态计算反馈权重（随连续触发次数增强）
    base_weight = min(
        dynamic_config['max_feedback_weight'],
        dynamic_config['min_feedback_weight'] + consecutive_triggers * dynamic_config['weight_increment']
    )
    
    # 连续触发时的权重提升
    if consecutive_triggers > 0:
        consecutive_boost = dynamic_config['consecutive_trigger_boost']
        base_weight *= (1 + consecutive_boost * min(consecutive_triggers / 3, 1.0))
    
    # 激进反馈策略：极低阈值触发
    if false_positive_rate >= thresholds['emergency']:
        # 紧急反馈：假阳率 >= 1%
        trigger_level = 'emergency'
        lr_factor = config['adaptive_lr_factors']['emergency']
        feedback_weight = min(base_weight * 2.0, dynamic_config['max_feedback_weight'])
    elif false_positive_rate >= thresholds['enhanced']:
        # 强化反馈：假阳率 >= 0.5%
        trigger_level = 'enhanced'
        lr_factor = config['adaptive_lr_factors']['enhanced']
        feedback_weight = min(base_weight * 1.5, dynamic_config['max_feedback_weight'])
    elif false_positive_rate >= thresholds['standard']:
        # 标准反馈：假阳率 >= 0.2%
        trigger_level = 'standard'
        lr_factor = config['adaptive_lr_factors']['standard']
        feedback_weight = base_weight
    elif false_positive_rate >= thresholds['warning']:
        # 轻度反馈：假阳率 >= 0.1%（不再仅记录，开始轻度干预）
        trigger_level = 'warning'
        lr_factor = 0.8  # 轻度学习率调整
        feedback_weight = base_weight * 0.3
    else:
        trigger_level = 'none'
        lr_factor = 1.0
        feedback_weight = 0.0
    
    return trigger_level, lr_factor, feedback_weight

def prepare_feedback_data(feedback_samples, device, batch_size=1000):
    """
    准备反馈数据
    
    参数:
        feedback_samples: 反馈样本ID列表 [8, 9]
        device: 计算设备
        batch_size: 批次大小
    
    返回:
        feedback_data: (vin1_batch, targets_batch) 用于反馈计算
    """
    try:
        print(f"🔧 准备反馈数据（样本 {feedback_samples}）...")
        
        all_vin1_data = []
        all_targets = []
        sample_lengths = []
        
        for sample_id in feedback_samples:
            # 加载vin_1数据
            vin1_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_1.pkl'
            with open(vin1_path, 'rb') as f:
                vin1_data = pickle.load(f)
                if isinstance(vin1_data, torch.Tensor):
                    vin1_data = vin1_data.cpu()
                else:
                    vin1_data = torch.tensor(vin1_data)
                
                # 记录数据长度和格式
                sample_length = len(vin1_data)
                sample_lengths.append(sample_length)
                print(f"   样本{sample_id}: {sample_length}个数据点, 格式{vin1_data.shape}")
                
                all_vin1_data.append(vin1_data)
            
            # 加载目标数据
            targets_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/targets.pkl'
            with open(targets_path, 'rb') as f:
                targets = pickle.load(f)
                terminal_voltages = np.array(targets['terminal_voltages'])
                pack_socs = np.array(targets['pack_socs'])
                
                print(f"   样本{sample_id} targets: 电压{len(terminal_voltages)}点, SOC{len(pack_socs)}点")
                
                # 组合目标数据：下一时刻的电压和SOC
                targets_combined = np.column_stack([terminal_voltages[1:], pack_socs[1:]])
                targets_tensor = torch.tensor(targets_combined, dtype=torch.float32)
                all_targets.append(targets_tensor)
        
        # 检查数据长度一致性
        if len(set(sample_lengths)) > 1:
            print(f"   ⚠️ 警告: 样本长度不一致 {sample_lengths}")
            print(f"   使用最小长度: {min(sample_lengths)}")
            
            # 统一截取到最小长度
            min_length = min(sample_lengths)
            for i in range(len(all_vin1_data)):
                all_vin1_data[i] = all_vin1_data[i][:min_length]
                all_targets[i] = all_targets[i][:min_length-1]  # targets少一个点
        
        # 分别处理每个样本，避免长度不匹配问题
        all_feedback_inputs = []
        all_feedback_targets = []
        
        for i, (vin1_data, targets_data) in enumerate(zip(all_vin1_data, all_targets)):
            # 构建输入数据：vin_1前5维 + 当前时刻真实电压 + 当前时刻真实SOC
            feedback_inputs = torch.zeros(len(vin1_data), 7, dtype=torch.float32)
            
            # 根据数据格式调整索引
            if len(vin1_data.shape) == 3:  # [time, 1, features]
                feedback_inputs[:, 0:5] = vin1_data[:, 0, 0:5]  # vin_1前5维
            elif len(vin1_data.shape) == 2:  # [time, features]
                feedback_inputs[:, 0:5] = vin1_data[:, 0:5]  # vin_1前5维
            else:
                print(f"   ⚠️ 未知的vin1_data格式: {vin1_data.shape}")
                continue
            
            # 添加当前时刻的真实值（从targets中获取前一时刻的值）
            if len(targets_data) > 0:
                # 确保维度匹配：feedback_inputs[1:] 对应 targets_data[:-1]
                # 因为feedback_inputs[0]对应targets_data[0]，但feedback_inputs[1]对应targets_data[0]
                feedback_inputs[1:, 5] = targets_data[:-1, 0]  # 当前时刻电压
                feedback_inputs[1:, 6] = targets_data[:-1, 1]  # 当前时刻SOC
                
                # 对应的目标是下一时刻的值
                feedback_targets = targets_data[1:]
                feedback_inputs = feedback_inputs[1:]
                
                # 确保截断后的维度匹配
                min_length = min(len(feedback_inputs), len(feedback_targets))
                if min_length > 0:
                    feedback_inputs = feedback_inputs[:min_length]
                    feedback_targets = feedback_targets[:min_length]
                    
                    all_feedback_inputs.append(feedback_inputs)
                    all_feedback_targets.append(feedback_targets)
        
        # 合并所有样本的数据
        if all_feedback_inputs:
            combined_inputs = torch.cat(all_feedback_inputs, dim=0)
            combined_targets = torch.cat(all_feedback_targets, dim=0)
            
            # 随机采样一个批次
            total_samples = len(combined_inputs)
            if total_samples > batch_size:
                indices = torch.randperm(total_samples)[:batch_size]
                combined_inputs = combined_inputs[indices]
                combined_targets = combined_targets[indices]
            
            print(f"   ✅ 反馈数据准备完成: 输入{combined_inputs.shape}, 目标{combined_targets.shape}")
            return (combined_inputs, combined_targets)
        else:
            print("   ❌ 没有有效的反馈数据")
            return None
        
    except Exception as e:
        print(f"   ❌ 反馈数据准备失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def apply_normal_sample_focus_training(transformer, feedback_data, optimizer, criterion, config, device, current_threshold=None):
    """
    正常样本特化训练 - 基于阈值相对优化，使FAI低于threshold1
    
    参数:
        transformer: Transformer模型
        feedback_data: 正常样本反馈数据
        optimizer: 优化器
        criterion: 损失函数
        config: 配置
        device: 计算设备
        current_threshold: 当前threshold1值
    
    返回:
        focus_loss: 特化训练损失
        avg_prediction_error: 平均预测误差
        threshold_info: 阈值相关信息
    """
    if not config['normal_sample_focus']['enable'] or feedback_data is None:
        return torch.tensor(0.0, device=device), 0.0, "未启用"
    
    try:
        vin1_batch, targets_batch = feedback_data
        vin1_batch = vin1_batch.to(device)
        targets_batch = targets_batch.to(device)
        
        transformer.train()
        
        # 前向传播
        predictions = transformer(vin1_batch)
        
        # 基础预测损失
        base_loss = criterion(predictions, targets_batch)
        
        # 计算预测误差（用于FAI估算）
        prediction_errors = torch.abs(predictions - targets_batch)
        avg_prediction_error = prediction_errors.mean().item()
        
        # 基于阈值的相对优化
        if current_threshold is not None and config['normal_sample_focus']['relative_penalty']:
            # 目标：FAI < threshold1 * margin（例如：< threshold1 * 0.8）
            target_threshold = current_threshold * config['normal_sample_focus']['threshold_margin']
            
            # 预测误差越大，FAI越可能超过目标阈值，施加相对惩罚
            # 使用sigmoid函数平滑惩罚，避免梯度突变
            error_ratio = prediction_errors / target_threshold
            relative_penalty = torch.sigmoid(error_ratio - 1.0) * config['normal_sample_focus']['penalty_factor']
            threshold_penalty = torch.mean(relative_penalty * prediction_errors)
            
            threshold_info = f"目标阈值={target_threshold:.4f}, 当前误差={avg_prediction_error:.4f}"
        else:
            # 如果没有阈值信息，使用基础FAI惩罚
            threshold_penalty = torch.mean(prediction_errors * config['normal_sample_focus']['penalty_factor'])
            threshold_info = f"基础惩罚模式, 预测误差={avg_prediction_error:.4f}"
        
        # 总损失 = 基础损失 * 权重 + 阈值相对惩罚
        focus_weight = config['normal_sample_focus']['focus_weight']
        total_loss = base_loss * focus_weight + threshold_penalty
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return total_loss, avg_prediction_error, threshold_info
        
    except Exception as e:
        print(f"   ❌ 正常样本特化训练失败: {e}")
        return torch.tensor(0.0, device=device), 0.0, f"失败: {e}"

def apply_hybrid_feedback(transformer, mcae_net1, mcae_net2, feedback_data, 
                         feedback_weight, mcae_weight, transformer_weight, device):
    """
    应用混合反馈机制（基于实际预测误差）
    
    参数:
        transformer: Transformer模型
        mcae_net1, mcae_net2: MC-AE模型
        feedback_data: 反馈数据 (vin1_batch, targets_batch)
        feedback_weight: 反馈强度权重
        mcae_weight: MC-AE权重
        transformer_weight: Transformer权重
        device: 计算设备
    
    返回:
        feedback_loss: 反馈损失
        feedback_info: 反馈信息
    """
    if feedback_weight == 0.0 or feedback_data is None:
        return torch.tensor(0.0, device=device), "无反馈"
    
    try:
        vin1_batch, targets_batch = feedback_data
        vin1_batch = vin1_batch.to(device)
        targets_batch = targets_batch.to(device)
        
        # 计算Transformer在反馈样本上的预测误差
        transformer.eval()
        with torch.no_grad():
            pred_output = transformer(vin1_batch)
            
        # 计算预测误差（MSE）
        prediction_error = torch.nn.functional.mse_loss(pred_output, targets_batch)
        
        # 基于实际预测误差计算反馈损失
        # 误差越大，反馈损失越大，训练调整越强烈
        feedback_loss = prediction_error * feedback_weight
        
        # 添加正则化项，防止过度调整
        l2_reg = sum(p.pow(2.0).sum() for p in transformer.parameters())
        regularization_loss = 1e-6 * l2_reg
        
        total_feedback_loss = feedback_loss + regularization_loss
        
        feedback_info = f"预测误差: {prediction_error:.6f}, 反馈权重: {feedback_weight:.2f}, 反馈损失: {feedback_loss:.6f}"
        
        return total_feedback_loss, feedback_info
        
    except Exception as e:
        print(f"   ⚠️ 反馈计算失败: {e}")
        # 降级为简化反馈
        fallback_loss = torch.tensor(feedback_weight * 0.001, device=device)
        return fallback_loss, f"降级反馈: {feedback_weight:.2f}"

#----------------------------------------主训练函数------------------------------
def main():
    """混合反馈策略主训练函数"""
    print("="*80)
    print("🚀 混合反馈策略Transformer训练 - Linux环境版本")
    print("="*80)
    
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
    
    #----------------------------------------配置显示------------------------------
    print("\n" + "="*60)
    print("⚙️  混合反馈策略配置")
    print("="*60)
    config = HYBRID_FEEDBACK_CONFIG
    print(f"📊 数据分组:")
    print(f"   训练样本: {config['train_samples']} (QAS 0-7)")
    print(f"   反馈样本: {config['feedback_samples']} (QAS 8-9)")
    print(f"🔧 激进反馈机制（专注降低假阳率）:")
    print(f"   反馈频率: 每{config['feedback_frequency']}个epoch （大幅提高）")
    print(f"   反馈启动轮数: 第{config['feedback_start_epoch']}轮 （提前介入）")
    print(f"   假阳性阈值（极严格）: {config['false_positive_thresholds']}")
    print(f"   自适应学习率因子（激进）: {config['adaptive_lr_factors']}")
    print(f"   MC-AE权重: {config['mcae_weight']}, Transformer权重: {config['transformer_weight']}")
    print(f"   动态反馈权重: {config['dynamic_feedback_weights']}")
    print(f"   正常样本特化训练: 目标FAI < threshold1 * {config['normal_sample_focus']['threshold_margin']}")
    
    #----------------------------------------阶段1: 基础Transformer训练------------------------------
    print("\n" + "="*60)
    print("🎯 阶段1: 基础Transformer训练（样本0-7）")
    print("="*60)
    
    # 加载训练样本
    train_samples = config['train_samples']
    print(f"📊 使用QAS目录中的{len(train_samples)}个训练样本: {train_samples}")
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU数量: {torch.cuda.device_count()}")
        print(f"🔧 当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 验证训练样本数据是否存在
    print("\n🔍 验证训练样本数据...")
    valid_samples = []
    for sample_id in train_samples:
        vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
        vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
        
        if os.path.exists(vin2_path) and os.path.exists(vin3_path):
            valid_samples.append(sample_id)
            print(f"   ✅ 样本 {sample_id}: 数据文件存在")
        else:
            print(f"   ❌ 样本 {sample_id}: 数据文件缺失")
    
    if len(valid_samples) == 0:
        print("❌ 没有找到有效的训练样本数据")
        print("请检查数据路径: /mnt/bz25t/bzhy/zhanglikang/project/QAS/")
        return
    
    print(f"✅ 找到 {len(valid_samples)} 个有效训练样本: {valid_samples}")
    
    # 使用复用的数据加载器
    print("\n📥 加载预计算数据...")
    try:
        # 创建数据集
        dataset = TransformerBatteryDataset(data_path='/mnt/bz25t/bzhy/zhanglikang/project/QAS', sample_ids=valid_samples)
        
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
    
    # 初始化Transformer模型（精简配置，匹配保守训练参数）
    transformer = TransformerPredictor(
        input_size=7,      # vin_1前5维 + 电压 + SOC
        d_model=128,       # 模型维度（精简规模）
        nhead=8,           # 注意力头数（精简规模）
        num_layers=3,      # Transformer层数（精简规模）
        output_size=2      # 输出：电压 + SOC
    ).to(device).float()
    
    # 单GPU优化模式（小样本训练，避免跨卡通信开销）
    print("🔧 单GPU优化模式：避免数据并行开销，专注于小样本训练")
    
    print(f"🧠 Transformer模型初始化完成")
    print(f"📈 模型参数量: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # 训练参数设置
    LR = 1.5e-3            # 学习率从1e-3增加到1.5e-3
    EPOCH_PHASE1 = config['feedback_start_epoch']  # 阶段1训练轮数
    EPOCH_PHASE2 = 120     # 阶段2总轮数（修正：必须大于EPOCH_PHASE1）
    lr_decay_freq = 15     # 学习率衰减频率从10增加到15
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=0.9)
    criterion = nn.MSELoss()
    
    # 设置混合精度训练
    scaler = setup_mixed_precision()
    
    print(f"⚙️  训练参数配置（精简模型 + 保守训练 + 改进反馈）:")
    print(f"   模型规模: d_model=128, nhead=8, layers=3（精简配置）")
    print(f"   学习率: {LR}（保守学习率，匹配精简模型）")
    print(f"   阶段1训练轮数: {EPOCH_PHASE1}")
    print(f"   总训练轮数: {EPOCH_PHASE2}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   学习率衰减频率: {lr_decay_freq}")
    print(f"   混合精度训练: 启用")
    print(f"   反馈机制: 基于实际预测误差 + 更严格阈值")
    
    # 开始阶段1训练
    print("\n🎯 开始阶段1训练...")
    transformer.train()
    train_losses_phase1 = []
    
    for epoch in range(EPOCH_PHASE1):
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
        train_losses_phase1.append(avg_loss)
        
        # 打印训练进度
        if epoch % 5 == 0 or epoch == EPOCH_PHASE1 - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'阶段1 Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}')
    
    print(f"\n✅ 阶段1训练完成! 最终损失: {train_losses_phase1[-1]:.6f}")
    
    #----------------------------------------阶段2: MC-AE训练（复用Train_Transformer.py逻辑）------------------------------
    print("\n" + "="*60)
    print("🔄 阶段2: 加载vin_2和vin_3数据，进行Transformer预测替换")
    print("="*60)
    
    # 复用Train_Transformer.py的逻辑加载和处理数据
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
            if processed_count % 5 == 0:
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
    
    # 使用Transformer进行预测和替换（复用Train_Transformer.py逻辑）
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
            if batch_idx % 5 == 0 or batch_idx == total_batches:
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
    
    # 训练MC-AE模型（复用Train_Transformer.py的完整逻辑）
    print("\n🧠 训练MC-AE异常检测模型（使用Transformer增强数据）...")
    
    # 复用Class_.py中的CombinedAE和Function_.py中的custom_activation
    from Function_ import custom_activation
    
    # 定义特征切片维度（与Train_Transformer.py一致）
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
    
    # MC-AE训练参数（与源代码Train_.py完全一致）
    EPOCH_MCAE = 300       # 恢复源代码的300轮训练
    LR_MCAE = 5e-4         # 恢复源代码的5e-4学习率
    BATCHSIZE_MCAE = 100   # 恢复源代码的100批次大小
    
    print(f"\n🔧 MC-AE训练参数（与源代码Train_.py完全对齐）:")
    print(f"   训练轮数: {EPOCH_MCAE} (源代码: 300)")
    print(f"   学习率: {LR_MCAE} (源代码: 5e-4)")
    print(f"   批次大小: {BATCHSIZE_MCAE} (源代码: 100)")
    print(f"   优化器: Adam")
    print(f"   损失函数: MSELoss")
    print(f"   激活函数: MC-AE1用custom_activation, MC-AE2用sigmoid")
    
    # 自定义多输入数据集类（复用Class_.py中的定义）
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
                               batch_size=BATCHSIZE_MCAE, shuffle=False)
    
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
                              batch_size=len(x_recovered), shuffle=False)
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
                                 batch_size=BATCHSIZE_MCAE, shuffle=False)
    
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
                               batch_size=len(x_recovered2), shuffle=False)
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
    
    print("✅ MC-AE训练完成! (与源代码Train_.py参数完全一致)")
    print(f"   MC-AE1最终损失: {train_losses_mcae1[-1]:.6f}")
    print(f"   MC-AE2最终损失: {train_losses_mcae2[-1]:.6f}")
    print(f"   训练参数: 轮数{EPOCH_MCAE}, 学习率{LR_MCAE}, 批次{BATCHSIZE_MCAE}")
    
    #----------------------------------------阶段3: 混合反馈训练------------------------------
    print("\n" + "="*60)
    print("🔮 阶段3: 混合反馈训练（样本8-9，轮数21-40）")
    print("="*60)
    
    # 阶段3在阶段4之后进行，因为需要PCA参数计算阈值
    print("⚠️ 阶段3将在PCA分析完成后进行，需要先获取故障检测阈值")
    train_losses_phase2 = []
    feedback_history = []
    consecutive_triggers = 0  # 连续触发计数器
    
    #----------------------------------------阶段4: PCA分析和保存模型（复用Train_Transformer.py逻辑）------------------------------
    print("\n" + "="*60)
    print("📊 阶段4: PCA分析，保存模型和参数")
    print("="*60)
    
    # 诊断特征提取与PCA分析（复用Function_.py中的函数）
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
    print("\n💾 保存混合反馈训练结果...")
    model_suffix = "_hybrid_feedback"
    
    # 确保models目录存在
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 1. 保存Transformer模型
    transformer_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/Transformer/models/transformer_model{model_suffix}.pth',  # 用户指定路径
        f'/tmp/transformer_model{model_suffix}.pth',
        f'./transformer_model{model_suffix}.pth',
        f'/mnt/bz25t/bzhy/zhanglikang/project/transformer_model{model_suffix}.pth',
        f'models/transformer_model{model_suffix}.pth'
    ]
    
    transformer_saved = False
    for save_path in transformer_save_paths:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            torch.save(transformer.state_dict(), save_path)
            print(f"✅ Transformer模型已保存: {save_path}")
            transformer_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存Transformer模型到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not transformer_saved:
        print("❌ 警告: Transformer模型保存失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
    
    # 2. 保存MC-AE模型
    mcae_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/Transformer/models/net_model{model_suffix}.pth',  # 用户指定路径
        f'/tmp/net_model{model_suffix}.pth',
        f'./net_model{model_suffix}.pth',
        f'/mnt/bz25t/bzhy/zhanglikang/project/net_model{model_suffix}.pth',
        f'models/net_model{model_suffix}.pth'
    ]
    
    mcae1_saved = False
    for save_path in mcae_save_paths:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            torch.save(net.state_dict(), save_path)
            print(f"✅ MC-AE1模型已保存: {save_path}")
            mcae1_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存MC-AE1模型到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    mcae2_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/Transformer/models/netx_model{model_suffix}.pth',  # 用户指定路径
        f'/tmp/netx_model{model_suffix}.pth',
        f'./netx_model{model_suffix}.pth',
        f'/mnt/bz25t/bzhy/zhanglikang/project/netx_model{model_suffix}.pth',
        f'models/netx_model{model_suffix}.pth'
    ]
    
    mcae2_saved = False
    for save_path in mcae2_save_paths:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            torch.save(netx.state_dict(), save_path)
            print(f"✅ MC-AE2模型已保存: {save_path}")
            mcae2_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存MC-AE2模型到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not mcae1_saved or not mcae2_saved:
        print("❌ 警告: 部分MC-AE模型保存失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
    
    # 3. 保存重构误差数据
    error_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/ERRORU{model_suffix}.npy',  # 用户指定路径
        f'/tmp/ERRORU{model_suffix}.npy',
        f'./ERRORU{model_suffix}.npy',
        f'/mnt/bz25t/bzhy/zhanglikang/project/ERRORU{model_suffix}.npy',
        f'models/ERRORU{model_suffix}.npy'
    ]
    
    erroru_saved = False
    for save_path in error_save_paths:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            np.save(save_path, ERRORU)
            print(f"✅ ERRORU已保存: {save_path}")
            erroru_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存ERRORU到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    errorx_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/ERRORX{model_suffix}.npy',  # 用户指定路径
        f'/tmp/ERRORX{model_suffix}.npy',
        f'./ERRORX{model_suffix}.npy',
        f'/mnt/bz25t/bzhy/zhanglikang/project/ERRORX{model_suffix}.npy',
        f'models/ERRORX{model_suffix}.npy'
    ]
    
    errorx_saved = False
    for save_path in errorx_save_paths:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            np.save(save_path, ERRORX)
            print(f"✅ ERRORX已保存: {save_path}")
            errorx_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存ERRORX到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not erroru_saved or not errorx_saved:
        print("❌ 警告: 部分重构误差数据保存失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
    
    # 4. 保存PCA分析结果（简化版本，只保存关键参数）
    pca_files = [
        ('v_I', v_I), ('v', v), ('v_ratio', v_ratio), ('p_k', p_k),
        ('data_mean', data_mean), ('data_std', data_std),
        ('T_95_limit', T_95_limit), ('T_99_limit', T_99_limit),
        ('SPE_95_limit', SPE_95_limit), ('SPE_99_limit', SPE_99_limit),
        ('P', P), ('k', k), ('P_t', P_t), ('X', X), ('data_nor', data_nor)
    ]
    
    pca_save_paths = [
        f'/mnt/bz25t/bzhy/datasave/',  # 用户指定路径
        f'/tmp/',
        f'./',
        f'/mnt/bz25t/bzhy/zhanglikang/project/',
        f'models/'
    ]
    
    pca_saved_count = 0
    for save_dir in pca_save_paths:
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(save_dir)
                print(f"📊 路径 {save_dir} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            for name, data in pca_files:
                save_path = f'{save_dir}{name}{model_suffix}.npy'
                np.save(save_path, data)
            print(f"✅ PCA分析结果已保存到: {save_dir}")
            pca_saved_count += 1
            break
        except OSError as e:
            print(f"⚠️ 保存PCA结果到 {save_dir} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if pca_saved_count == 0:
        print("❌ 警告: PCA分析结果保存失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
        print("💡 尝试手动创建目录: mkdir -p /mnt/bz25t/bzhy/zhanglikang/project")
    
    # 5. 保存PCA参数字典（用于反馈阶段）
    pca_params = {
        'v_I': v_I,
        'v': v,
        'v_ratio': v_ratio,
        'p_k': p_k,
        'data_mean': data_mean,
        'data_std': data_std,
        'T_95_limit': T_95_limit,
        'T_99_limit': T_99_limit,
        'SPE_95_limit': SPE_95_limit,
        'SPE_99_limit': SPE_99_limit,
        'P': P,
        'k': k,
        'P_t': P_t,
        'X': X,
        'data_nor': data_nor
    }
    
    # 尝试多个保存路径，处理磁盘空间不足问题
    save_paths = [
                f'/mnt/bz25t/bzhy/datasave/Transformer/models/pca_params{model_suffix}.pkl',  # 用户指定路径
        f'/tmp/pca_params{model_suffix}.pkl',
        f'./pca_params{model_suffix}.pkl',
        f'/mnt/bz25t/bzhy/zhanglikang/project/pca_params{model_suffix}.pkl',
        f'models/pca_params{model_suffix}.pkl'
    ]
    
    saved = False
    for save_path in save_paths:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            with open(save_path, 'wb') as f:
                pickle.dump(pca_params, f)
            print(f"✅ PCA参数字典已保存: {save_path}")
            saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not saved:
        print("❌ 警告: 所有保存路径都失败，PCA参数未保存")
        print("💡 建议: 检查目录权限或使用其他存储位置")
        print("💡 尝试手动创建目录: mkdir -p /mnt/bz25t/bzhy/zhanglikang/project")
        # 将PCA参数保存到内存中，供后续使用
        global_saved_pca_params = pca_params
    
    # 6. 计算训练阶段故障检测阈值
    if len(valid_samples) == 0:
        print("❌ 没有有效的训练样本，无法计算故障检测阈值")
        return
    
    # 确保MC-AE模型在正确的设备上
    net = net.to(device)
    netx = netx.to(device)
    
    print(f"🔍 设备检查: net在{next(net.parameters()).device}, netx在{next(netx.parameters()).device}, 目标设备{device}")
    
    threshold1, threshold2, threshold3 = calculate_training_threshold(
        valid_samples, net, netx, pca_params, device)
    
    # 保存阈值
    thresholds = {
        'threshold1': threshold1,  # 3σ阈值
        'threshold2': threshold2,  # 4.5σ阈值
        'threshold3': threshold3   # 6σ阈值
    }
    
    # 尝试多个保存路径，处理磁盘空间不足问题
    threshold_save_paths = [
                f'/mnt/bz25t/bzhy/datasave/fault_thresholds{model_suffix}.pkl',  # 用户指定路径
        f'/tmp/fault_thresholds{model_suffix}.pkl',
        f'./fault_thresholds{model_suffix}.pkl',
        f'/mnt/bz25t/bzhy/zhanglikang/project/fault_thresholds{model_suffix}.pkl',
        f'models/fault_thresholds{model_suffix}.pkl'
    ]
    
    threshold_saved = False
    for save_path in threshold_save_paths:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            with open(save_path, 'wb') as f:
                pickle.dump(thresholds, f)
            print(f"✅ 故障检测阈值已保存: {save_path}")
            threshold_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存阈值到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not threshold_saved:
        print("❌ 警告: 所有阈值保存路径都失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
        print("💡 尝试手动创建目录: mkdir -p /mnt/bz25t/bzhy/zhanglikang/project")
        # 将阈值保存到内存中，供后续使用
        global_saved_thresholds = thresholds
    
    #----------------------------------------现在开始阶段3: 混合反馈训练------------------------------
    print("\n" + "="*60)
    print("🔮 现在开始阶段3: 混合反馈训练（使用计算出的阈值）")
    print("="*60)
    
    # 使用计算出的阈值进行反馈训练
    current_threshold = threshold1
    print(f"✅ 使用计算得到的阈值: {current_threshold:.4f}")
    
    # 继续训练（阶段3：混合反馈）
    transformer.train()
    
    print(f"\n🎯 开始阶段3训练（epoch {EPOCH_PHASE1+1}-{EPOCH_PHASE2}）...")
    
    for epoch in range(EPOCH_PHASE1, EPOCH_PHASE2):
        epoch_loss = 0
        batch_count = 0
        feedback_triggered = False
        trigger_info = "无反馈"
        
        # 检查是否启用反馈
        if (epoch >= config['feedback_start_epoch'] and 
            epoch % config['feedback_frequency'] == 0):
            
            print(f"\n🔍 Epoch {epoch}: 检查反馈触发条件...")
            
            try:
                # 确保MC-AE模型在正确的设备上
                net = net.to(device)
                netx = netx.to(device)
                
                # 计算当前的假阳性率（基于综合诊断指标）
                false_positive_rate, false_positives, total_normals = calculate_false_positive_rate_comprehensive(
                    config['feedback_samples'], net, netx, pca_params, current_threshold, device)
                
                print(f"   当前假阳性率: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
                
                # 检测反馈触发（集成连续触发追踪）
                trigger_level, lr_factor, feedback_weight = detect_feedback_trigger(
                    false_positive_rate, epoch, config, consecutive_triggers)
                
                if trigger_level != 'none':
                    feedback_triggered = True
                    consecutive_triggers += 1  # 增加连续触发计数
                    trigger_info = f"{trigger_level}反馈 (权重:{feedback_weight:.2f}, LR因子:{lr_factor:.2f}, 连续:{consecutive_triggers})"
                    
                    # 调整学习率
                    if lr_factor != 1.0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_factor
                        print(f"   学习率调整: {param_group['lr']:.6f}")
                    
                    # 准备反馈数据（基于实际预测误差）
                    feedback_data = prepare_feedback_data(config['feedback_samples'], device, batch_size=500)
                    
                    if feedback_data is not None:
                        # 1. 应用正常样本特化训练（基于阈值相对优化）
                        focus_loss, avg_pred_error, threshold_info = apply_normal_sample_focus_training(
                            transformer, feedback_data, optimizer, criterion, config, device, current_threshold)
                        print(f"   正常样本特化训练: 损失={focus_loss:.6f}, {threshold_info}")
                        
                        # 2. 应用混合反馈
                        feedback_loss, feedback_info = apply_hybrid_feedback(
                            transformer, net, netx, feedback_data, 
                            feedback_weight, config['mcae_weight'], config['transformer_weight'], device)
                        
                        print(f"   {feedback_info}")
                    else:
                        print(f"   ⚠️ 反馈数据准备失败，跳过反馈训练")
                    
                    # 记录反馈历史
                    feedback_history.append({
                        'epoch': epoch,
                        'false_positive_rate': false_positive_rate,
                        'trigger_level': trigger_level,
                        'feedback_weight': feedback_weight,
                        'lr_factor': lr_factor,
                        'false_positives': false_positives,
                        'total_normals': total_normals,
                        'consecutive_triggers': consecutive_triggers
                    })
                else:
                    consecutive_triggers = 0  # 重置连续触发计数
                    print(f"   无需反馈 (假阳性率: {false_positive_rate:.4f})")
                    
            except Exception as e:
                print(f"   ❌ 反馈计算失败: {e}")
                print("   继续正常训练...")
                feedback_triggered = False
        
        # 正常训练循环
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
                
                # 如果有反馈，添加反馈损失
                if feedback_triggered and 'feedback_loss' in locals():
                    total_loss = loss + 0.1 * feedback_loss  # 反馈损失权重为0.1
                else:
                    total_loss = loss
            
            # 混合精度反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            batch_count += 1
        
        # 学习率调度（在反馈调整之后）
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        train_losses_phase2.append(avg_loss)
        
        # 打印训练进度
        if epoch % 2 == 0 or epoch == EPOCH_PHASE2 - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'阶段3 Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | {trigger_info}')
    
    if train_losses_phase2:
        print(f"\n✅ 阶段3混合反馈训练完成! 最终损失: {train_losses_phase2[-1]:.6f}")
    else:
        print(f"\n⚠️ 阶段3混合反馈训练完成! 但训练损失列表为空")
    print(f"📊 反馈触发次数: {len(feedback_history)}")
    if feedback_history:
        avg_fpr = np.mean([h['false_positive_rate'] for h in feedback_history])
        print(f"📊 平均假阳性率: {avg_fpr:.4f} ({avg_fpr*100:.2f}%)")
    
    # 5. 保存混合反馈训练历史
    hybrid_feedback_history = {
        'phase1_losses': train_losses_phase1,
        'phase2_losses': train_losses_phase2,
        'mcae1_losses': train_losses_mcae1,
        'mcae2_losses': train_losses_mcae2,
        'feedback_history': feedback_history,
        'final_phase1_loss': train_losses_phase1[-1] if train_losses_phase1 else None,
        'final_phase2_loss': train_losses_phase2[-1] if train_losses_phase2 else None,
        'config': config,
        'feedback_triggers': len(feedback_history),
        'model_params': {
            'input_size': 7,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'output_size': 2
        }
    }
    
    # 尝试多个保存路径，处理磁盘空间不足问题
    history_save_paths = [
                f'/mnt/bz25t/bzhy/datasave/hybrid_feedback_training_history.pkl',  # 用户指定路径
        f'/tmp/hybrid_feedback_training_history.pkl',
        f'./hybrid_feedback_training_history.pkl',
        f'/mnt/bz25t/bzhy/zhanglikang/project/hybrid_feedback_training_history.pkl',
        f'models/hybrid_feedback_training_history.pkl'
    ]
    
    history_saved = False
    for save_path in history_save_paths:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查磁盘空间（如果可能）
            try:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(save_path))
                print(f"📊 路径 {os.path.dirname(save_path)} 可用空间: {free / (1024**3):.2f} GB")
            except:
                pass
            
            with open(save_path, 'wb') as f:
                pickle.dump(hybrid_feedback_history, f)
            print(f"✅ 混合反馈训练历史已保存: {save_path}")
            history_saved = True
            break
        except OSError as e:
            print(f"⚠️ 保存训练历史到 {save_path} 失败: {e}")
            print(f"   错误代码: {e.errno}, 错误信息: {e.strerror}")
            continue
    
    if not history_saved:
        print("❌ 警告: 所有训练历史保存路径都失败")
        print("💡 建议: 检查目录权限或使用其他存储位置")
        print("💡 尝试手动创建目录: mkdir -p /mnt/bz25t/bzhy/zhanglikang/project")
        # 将训练历史保存到内存中，供后续使用
        global_saved_training_history = hybrid_feedback_history
    
    # 6. 保存诊断特征
    print(f"💾 保存诊断特征（数据量: {df_data.shape}）...")
    csv_path = f'models/diagnosis_feature{model_suffix}.csv'
    df_data.to_csv(csv_path, index=False)
    print(f"✅ 诊断特征CSV已保存: {csv_path}")
    
    #----------------------------------------绘制混合反馈训练结果------------------------------
    print("\n📈 绘制混合反馈训练结果...")
    
    # 创建综合图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 子图1: 两阶段训练损失对比
    ax1 = axes[0, 0]
    epochs_phase1 = range(1, len(train_losses_phase1) + 1)
    epochs_phase2 = range(len(train_losses_phase1) + 1, len(train_losses_phase1) + len(train_losses_phase2) + 1)
    
    ax1.plot(epochs_phase1, train_losses_phase1, 'b-', linewidth=2, label='阶段1: 基础训练')
    if train_losses_phase2:
        ax1.plot(epochs_phase2, train_losses_phase2, 'r-', linewidth=2, label='阶段3: 混合反馈')
    ax1.axvline(x=len(train_losses_phase1), color='gray', linestyle='--', alpha=0.7, label='反馈启动点')
    
    if use_chinese:
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('MSE损失')
        ax1.set_title('混合反馈策略训练损失')
    else:
        ax1.set_xlabel('Training Epochs')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Hybrid Feedback Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 子图2: MC-AE1训练损失
    ax2 = axes[0, 1]
    epochs = range(1, len(train_losses_mcae1) + 1)
    ax2.plot(epochs, train_losses_mcae1, 'g-', linewidth=2, label='MC-AE1 Training Loss')
    if use_chinese:
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('MSE损失')
        ax2.set_title('MC-AE1训练损失曲线')
    else:
        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('MC-AE1 Training Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 子图3: MC-AE2训练损失
    ax3 = axes[0, 2]
    ax3.plot(epochs, train_losses_mcae2, 'orange', linewidth=2, label='MC-AE2 Training Loss')
    if use_chinese:
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('MSE损失')
        ax3.set_title('MC-AE2训练损失曲线')
    else:
        ax3.set_xlabel('Training Epochs')
        ax3.set_ylabel('MSE Loss')
        ax3.set_title('MC-AE2 Training Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')
    
    # 子图4: 反馈触发历史
    ax4 = axes[1, 0]
    if feedback_history:
        feedback_epochs = [item['epoch'] for item in feedback_history]
        feedback_rates = [item['false_positive_rate'] for item in feedback_history]
        feedback_levels = [item['trigger_level'] for item in feedback_history]
        
        # 用不同颜色表示不同的反馈等级
        color_map = {'standard': 'yellow', 'enhanced': 'orange', 'emergency': 'red'}
        for i, (epoch, rate, level) in enumerate(zip(feedback_epochs, feedback_rates, feedback_levels)):
            color = color_map.get(level, 'gray')
            ax4.scatter(epoch, rate, c=color, s=100, alpha=0.7, 
                       label=level if level not in [item.get_text() for item in ax4.get_legend_handles_labels()[1]] else "")
        
        # 添加阈值线
        thresholds = config['false_positive_thresholds']
        ax4.axhline(y=thresholds['standard'], color='yellow', linestyle='--', alpha=0.5, label='标准阈值')
        ax4.axhline(y=thresholds['enhanced'], color='orange', linestyle='--', alpha=0.5, label='强化阈值')
        ax4.axhline(y=thresholds['emergency'], color='red', linestyle='--', alpha=0.5, label='紧急阈值')
    
    if use_chinese:
        ax4.set_xlabel('训练轮数')
        ax4.set_ylabel('假阳性率')
        ax4.set_title('反馈触发历史')
    else:
        ax4.set_xlabel('Training Epochs')
        ax4.set_ylabel('False Positive Rate')
        ax4.set_title('Feedback Trigger History')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: MC-AE1重构误差分布
    ax5 = axes[1, 1]
    reconstruction_errors_1 = ERRORU.flatten()
    mean_error_1 = np.mean(np.abs(reconstruction_errors_1))
    if use_chinese:
        ax5.hist(np.abs(reconstruction_errors_1), bins=50, alpha=0.7, color='blue', 
                label=f'MC-AE1重构误差 (均值: {mean_error_1:.4f})')
        ax5.set_xlabel('绝对重构误差')
        ax5.set_ylabel('频数')
        ax5.set_title('MC-AE1重构误差分布')
    else:
        ax5.hist(np.abs(reconstruction_errors_1), bins=50, alpha=0.7, color='blue', 
                label=f'MC-AE1 Reconstruction Error (Mean: {mean_error_1:.4f})')
        ax5.set_xlabel('Absolute Reconstruction Error')
        ax5.set_ylabel('Frequency')
        ax5.set_title('MC-AE1 Reconstruction Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图6: MC-AE2重构误差分布
    ax6 = axes[1, 2]
    reconstruction_errors_2 = ERRORX.flatten()
    mean_error_2 = np.mean(np.abs(reconstruction_errors_2))
    if use_chinese:
        ax6.hist(np.abs(reconstruction_errors_2), bins=50, alpha=0.7, color='red',
                label=f'MC-AE2重构误差 (均值: {mean_error_2:.4f})')
        ax6.set_xlabel('绝对重构误差')
        ax6.set_ylabel('频数')
        ax6.set_title('MC-AE2重构误差分布')
    else:
        ax6.hist(np.abs(reconstruction_errors_2), bins=50, alpha=0.7, color='red',
                label=f'MC-AE2 Reconstruction Error (Mean: {mean_error_2:.4f})')
        ax6.set_xlabel('Absolute Reconstruction Error')
        ax6.set_ylabel('Frequency')
        ax6.set_title('MC-AE2 Reconstruction Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'/mnt/bz25t/bzhy/datasave/Transformer/models/hybrid_feedback_training_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混合反馈训练结果图已保存: {plot_path}")
    
    #----------------------------------------最终训练完成总结------------------------------
    print("\n" + "="*80)
    print("🎉 混合反馈策略训练完成！")
    print("="*80)
    print("✅ 训练流程总结:")
    print("   阶段1: ✅ Transformer基础训练 (样本0-7, epoch 0-20)")
    print("   阶段2: ✅ MC-AE训练 (使用Transformer增强数据)")
    print("   阶段3: ✅ 混合反馈训练 (样本8-9, epoch 21-40)")
    print("   阶段4: ✅ PCA分析和模型保存")
    print("")
    print("🔧 关键修复 (与源代码Train_.py对齐):")
    print(f"   - MC-AE训练轮数: {EPOCH_MCAE} (源代码: 300)")
    print(f"   - MC-AE学习率: {LR_MCAE} (源代码: 5e-4)")
    print(f"   - MC-AE批次大小: {BATCHSIZE_MCAE} (源代码: 100)")
    print("   - 激活函数: MC-AE1用custom_activation, MC-AE2用sigmoid")
    print("")
    print("📊 关键创新:")
    print("   - 数据隔离策略：训练/反馈/测试样本严格分离")
    print("   - 多级反馈触发：1%预警、3%标准、5%强化、10%紧急")
    print("   - 混合权重机制：MC-AE权重0.8，Transformer权重0.2")
    print("   - 自适应学习率：根据假阳性率动态调整")
    print("   - 实时反馈监控：每15个epoch检查触发条件")
    print("")
    print("📈 性能指标:")
    print(f"   阶段1最终损失: {train_losses_phase1[-1]:.6f}")
    if train_losses_phase2:
        print(f"   阶段3最终损失: {train_losses_phase2[-1]:.6f}")
    else:
        print(f"   阶段3最终损失: 无数据")
    print(f"   反馈触发次数: {len(feedback_history)}")
    print(f"   PCA主成分数量: {k}")
    print("")
    print("🔄 下一步可以:")
    print("   1. 运行Test_combine_transonly.py进行性能评估")
    print("   2. 与BiLSTM基准进行详细对比分析")
    print("   3. 分析混合反馈策略的改进效果")
    print("   4. 调整反馈参数进行进一步优化")

if __name__ == "__main__":
    main()