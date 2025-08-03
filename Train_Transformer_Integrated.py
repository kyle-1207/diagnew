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
from Comprehensive_calculation import Comprehensive_calculation
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
from tqdm import tqdm
import json
import time
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Comprehensive_calculation import Comprehensive_calculation

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

# GPU设备配置
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

# 更全面的字体检测和设置
def setup_chinese_fonts():
    """设置中文字体，如果不可用则使用英文"""
    try:
        # 尝试多种中文字体
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
            'Noto Sans CJK JP', 'Noto Sans CJK TC', 'Source Han Sans CN',
            'Droid Sans Fallback', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
        ]
        
        # 检查系统字体
        try:
            system_fonts = [f.name for f in fm.fontManager.ttflist]
            print(f"🔍 系统可用字体数量: {len(system_fonts)}")
        except:
            system_fonts = []
            print("⚠️  无法获取系统字体列表")
        
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
    except Exception as e:
        print(f"⚠️  字体设置出现问题: {e}")
        # 使用最基本的字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 设置字体
use_chinese = setup_chinese_fonts()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

#----------------------------------------实验配置------------------------------
# 训练和测试样本配置
TRAIN_SAMPLES = list(range(10))  # 训练集：QAS 0-9（10个正常样本）
TEST_SAMPLES = {
    'normal': ['10', '11'],      # 测试正常样本
    'fault': ['335', '336']      # 测试故障样本
}
ALL_TEST_SAMPLES = TEST_SAMPLES['normal'] + TEST_SAMPLES['fault']

# 三窗口固定参数
WINDOW_CONFIG = {
    "detection_window": 100,     # 检测窗口：100个采样点
    "verification_window": 50,   # 验证窗口：50个采样点  
    "marking_window": 50        # 标记窗口：前后各50个采样点
}

# 两种实验配置
EXPERIMENT_CONFIGS = {
    'original': {
        'name': '原始参数规模',
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 512,
        'dropout': 0.1,
        'save_suffix': '_original'
    },
    'enhanced': {
        'name': '增强参数规模(+50%)',
        'd_model': 192,      # 128 * 1.5
        'n_heads': 12,       # 8 * 1.5
        'n_layers': 6,       # 保持不变
        'd_ff': 768,         # 512 * 1.5
        'dropout': 0.2,      # 增加dropout防止过拟合
        'save_suffix': '_enhanced'
    }
}

# 反向传播机制参数
FEEDBACK_CONFIG = {
    'use_feedback': True,
    'feedback_frequency': 5,     # 每5个epoch执行一次反馈
    'feedback_alpha': 0.1,       # 反馈损失权重
    'diagnosis_threshold': 0.5,  # 诊断阈值
    'min_false_positives': 10    # 最少假阳性样本数才执行反馈
}

print("="*60)
print("🚀 Transformer集成训练测试系统")
print("="*60)
print(f"📊 实验配置:")
print(f"   训练样本: {TRAIN_SAMPLES} (共{len(TRAIN_SAMPLES)}个)")
print(f"   测试样本: {ALL_TEST_SAMPLES} (正常:{len(TEST_SAMPLES['normal'])}, 故障:{len(TEST_SAMPLES['fault'])})")
print(f"   三窗口参数: {WINDOW_CONFIG}")
print(f"   反向传播: {'启用' if FEEDBACK_CONFIG['use_feedback'] else '禁用'}")
for config_name, config in EXPERIMENT_CONFIGS.items():
    print(f"   {config['name']}: d_model={config['d_model']}, n_heads={config['n_heads']}")

#----------------------------------------Transformer模型定义------------------------------
class TransformerPredictor(nn.Module):
    """时序预测Transformer模型 - 支持可配置参数"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=3, d_ff=None, dropout=0.1, output_size=2):
        super(TransformerPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        if d_ff is None:
            d_ff = d_model * 4  # 默认为d_model的4倍
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model, dtype=torch.float32))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 - 直接输出物理值，不使用Sigmoid
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, output_size)
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

#----------------------------------------反向传播机制设计------------------------------
def compute_feedback_loss(transformer_output, false_positive_mask, normal_center):
    """计算反向传播的反馈损失"""
    if not false_positive_mask.any():
        return torch.tensor(0.0, device=transformer_output.device, requires_grad=True)
    
    false_positive_features = transformer_output[false_positive_mask]
    
    # 距离损失：让假阳性样本的特征向量更接近正常样本的平均特征
    if len(false_positive_features) > 0:
        distance_loss = F.mse_loss(
            false_positive_features, 
            normal_center.expand_as(false_positive_features)
        )
        
        # 对比损失：增强正常样本与异常样本的区分度
        # 简化版本：使用L2距离
        contrastive_loss = torch.mean(torch.norm(false_positive_features - normal_center, dim=1))
        
        return distance_loss + 0.1 * contrastive_loss
    else:
        return torch.tensor(0.0, device=transformer_output.device, requires_grad=True)

#----------------------------------------三窗口故障检测机制------------------------------
def three_window_fault_detection(fai_values, threshold1, sample_id):
    """
    三窗口故障检测机制：检测→验证→标记
    
    Args:
        fai_values: 综合诊断指标序列
        threshold1: 一级预警阈值
        sample_id: 样本ID（用于调试）
    
    Returns:
        fault_labels: 故障标签序列 (0=正常, 1=故障)
        detection_info: 检测过程详细信息
    """
    detection_window = WINDOW_CONFIG["detection_window"]
    verification_window = WINDOW_CONFIG["verification_window"] 
    marking_window = WINDOW_CONFIG["marking_window"]
    
    fault_labels = np.zeros(len(fai_values), dtype=int)
    detection_info = {
        'candidate_points': [],
        'verified_points': [],
        'marked_regions': [],
        'window_stats': {}
    }
    
    # 阶段1：检测窗口 - 寻找候选故障点
    candidate_points = []
    for i in range(len(fai_values)):
        if fai_values[i] > threshold1:
            candidate_points.append(i)
    
    detection_info['candidate_points'] = candidate_points
    
    if len(candidate_points) == 0:
        # 没有候选点，直接返回
        return fault_labels, detection_info
    
    # 阶段2：验证窗口 - 检查持续性
    verified_points = []
    for candidate in candidate_points:
        # 定义验证窗口范围
        start_verify = max(0, candidate - verification_window//2)
        end_verify = min(len(fai_values), candidate + verification_window//2)
        verify_data = fai_values[start_verify:end_verify]
        
        # 持续性判断：验证窗口内超阈值点比例
        continuous_ratio = np.sum(verify_data > threshold1) / len(verify_data)
        
        # 30%以上超阈值认为持续异常
        if continuous_ratio >= 0.3:
            verified_points.append({
                'point': candidate,
                'continuous_ratio': continuous_ratio,
                'verify_range': (start_verify, end_verify)
            })
    
    detection_info['verified_points'] = verified_points
    
    # 阶段3：标记窗口 - 标记故障区域
    marked_regions = []
    for verified in verified_points:
        candidate = verified['point']
        
        # 定义标记窗口范围
        start_mark = max(0, candidate - marking_window)
        end_mark = min(len(fai_values), candidate + marking_window)
        
        # 标记故障区域
        fault_labels[start_mark:end_mark] = 1
        
        marked_regions.append({
            'center': candidate,
            'range': (start_mark, end_mark),
            'length': end_mark - start_mark
        })
    
    detection_info['marked_regions'] = marked_regions
    
    # 统计信息
    detection_info['window_stats'] = {
        'total_candidates': len(candidate_points),
        'verified_candidates': len(verified_points),
        'total_fault_points': np.sum(fault_labels),
        'fault_ratio': np.sum(fault_labels) / len(fault_labels)
    }
    
    return fault_labels, detection_info

#----------------------------------------诊断阈值计算------------------------------
def calculate_thresholds(fai):
    """按照Test_combine_transonly.py的方法计算诊断阈值"""
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
    
    return threshold1, threshold2, threshold3

#----------------------------------------数据加载函数------------------------------
def load_train_samples():
    """加载训练样本ID"""
    try:
        import pandas as pd
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/data/QAS/Labels.xls'
        
        # 检查文件是否存在
        if not os.path.exists(labels_path):
            print(f"⚠️  Labels.xls文件不存在: {labels_path}")
            print("⚠️  使用默认样本范围 0-9")
            return TRAIN_SAMPLES
        
        df = pd.read_excel(labels_path)
        print(f"📋 成功读取Labels.xls, 共{len(df)}行数据")
        
        # 检查DataFrame列名
        print(f"📋 可用列名: {df.columns.tolist()}")
        
        # 尝试不同的列名
        if 'Num' in df.columns:
            all_samples = df['Num'].tolist()
        elif 'num' in df.columns:
            all_samples = df['num'].tolist()
        elif df.columns.size > 0:
            # 使用第一列
            all_samples = df.iloc[:, 0].tolist()
        else:
            raise ValueError("无法找到样本ID列")
        
        # 提取0-9范围的样本
        train_samples = [i for i in all_samples if i in TRAIN_SAMPLES]
        
        print(f"📋 从Labels.xls加载训练样本:")
        print(f"   训练样本范围: 0-9")
        print(f"   实际可用样本: {len(train_samples)} 个")
        print(f"   样本列表: {train_samples}")
        
        return train_samples if train_samples else TRAIN_SAMPLES
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认样本范围 0-9")
        return TRAIN_SAMPLES

def load_test_sample(sample_id):
    """加载测试样本"""
    base_path = f'/mnt/bz25t/bzhy/zhanglikang/project/data/QAS/{sample_id}'
    
    # 检查样本目录是否存在
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"测试样本目录不存在: {base_path}")
    
    # 加载vin_1, vin_2, vin_3数据
    try:
        with open(f'{base_path}/vin_1.pkl', 'rb') as f:
            vin1_data = pickle.load(f)
        with open(f'{base_path}/vin_2.pkl', 'rb') as f:
            vin2_data = pickle.load(f) 
        with open(f'{base_path}/vin_3.pkl', 'rb') as f:
            vin3_data = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"样本 {sample_id} 数据文件缺失: {e}")
        
    return vin1_data, vin2_data, vin3_data

#----------------------------------------主训练函数------------------------------
def train_experiment(config_name, config):
    """训练单个实验配置"""
    
    print(f"\n{'='*60}")
    print(f"🚀 开始训练实验: {config['name']}")
    print(f"{'='*60}")
    
    # 创建保存目录
    save_suffix = config['save_suffix']
    save_dir = f"modelsfl{save_suffix}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否存在中间结果，支持断点续算
    checkpoint_path = f"{save_dir}/checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        print(f"🔄 发现断点文件，尝试恢复训练...")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"✅ 成功加载断点，阶段: {checkpoint.get('stage', 'unknown')}")
            if checkpoint.get('stage') == 'completed':
                print(f"⚠️  实验已完成，跳过训练")
                return checkpoint
        except Exception as e:
            print(f"❌ 断点文件损坏: {e}，重新开始训练")
    
    # 初始化结果字典
    experiment_results = {
        'config': config,
        'stage': 'starting',
        'transformer_results': {},
        'mcae_results': {},
        'pca_results': {},
        'test_results': {},
        'timing': {}
    }
    
    # 阶段1: 训练Transformer
    print(f"\n🎯 阶段1: 训练Transformer模型")
    start_time = time.time()
    
    try:
        # 加载训练样本
        train_samples = load_train_samples()
        print(f"📊 使用{len(train_samples)}个训练样本")
        
        # 创建数据集
        dataset = TransformerBatteryDataset(
            data_path='/mnt/bz25t/bzhy/zhanglikang/project/QAS', 
            sample_ids=train_samples
        )
        
        if len(dataset) == 0:
            raise ValueError("没有加载到任何训练数据")
        
        print(f"✅ 成功加载 {len(dataset)} 个训练数据对")
        
        # 创建数据加载器
        BATCH_SIZE = 4000
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=4, pin_memory=True)
        
        # 初始化Transformer模型
        transformer = TransformerPredictor(
            input_size=7,
            d_model=config['d_model'],
            nhead=config['n_heads'],
            num_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            output_size=2
        ).to(device).float()
        
        # 启用数据并行
        if torch.cuda.device_count() > 1:
            transformer = torch.nn.DataParallel(transformer)
            print(f"✅ 启用数据并行，使用 {torch.cuda.device_count()} 张GPU")
        
        print(f"🧠 Transformer模型初始化完成")
        print(f"📈 模型参数量: {sum(p.numel() for p in transformer.parameters()):,}")
        
        # 训练参数
        LR = 1.5e-3
        EPOCH = 40
        lr_decay_freq = 15
        
        optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=0.9)
        criterion = nn.MSELoss()
        scaler = setup_mixed_precision()
        
        # 训练循环
        transformer.train()
        train_losses = []
        
        for epoch in range(EPOCH):
            epoch_loss = 0
            batch_count = 0
            
            for batch_input, batch_target in train_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    pred_output = transformer(batch_input)
                    loss = criterion(pred_output, batch_target)
                
                # 反向传播机制（每5个epoch执行一次）
                if FEEDBACK_CONFIG['use_feedback'] and epoch % FEEDBACK_CONFIG['feedback_frequency'] == 0:
                    # 简化的反馈机制：这里我们暂时跳过，因为需要MC-AE的结果
                    pass
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            scheduler.step()
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            
            if epoch % 5 == 0 or epoch == EPOCH - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}')
        
        # 保存Transformer模型
        transformer_path = f'{save_dir}/transformer_model.pth'
        torch.save(transformer.state_dict(), transformer_path)
        print(f"✅ Transformer模型已保存: {transformer_path}")
        
        # 记录Transformer训练结果
        experiment_results['transformer_results'] = {
            'train_losses': train_losses,
            'final_loss': train_losses[-1],
            'model_path': transformer_path,
            'config': config
        }
        experiment_results['timing']['transformer'] = time.time() - start_time
        experiment_results['stage'] = 'transformer_completed'
        
        # 保存中间结果
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(experiment_results, f)
        
        print(f"✅ Transformer训练完成，用时: {experiment_results['timing']['transformer']:.2f}秒")
        
    except Exception as e:
        print(f"❌ Transformer训练失败: {e}")
        raise e
    
    return experiment_results

# 继续添加MC-AE训练和测试函数
# 由于篇幅限制，这里先实现一个简化版本的完整流程

def main():
    """主执行函数"""
    
    print("\n🚀 开始集成训练测试实验...")
    
    all_results = {}
    
    # 执行两个实验配置
    for config_name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"🔬 实验配置: {config['name']}")
        print(f"{'='*80}")
        
        try:
            # 训练实验（这里只是Transformer部分，完整版本需要添加MC-AE训练）
            experiment_results = train_experiment(config_name, config)
            
            # TODO: 添加MC-AE训练和PCA分析
            # TODO: 添加测试功能
            # TODO: 添加可视化功能
            
            all_results[config_name] = experiment_results
            
            print(f"✅ {config['name']} 实验阶段1完成")
            
        except Exception as e:
            print(f"❌ 实验 {config['name']} 失败: {e}")
            continue
    
    print(f"\n🎉 Transformer训练阶段完成！")
    print(f"📝 注意: 这是简化版本，完整版本需要添加:")
    print(f"   1. MC-AE训练功能")
    print(f"   2. PCA分析功能") 
    print(f"   3. 测试和三窗口检测功能")
    print(f"   4. 可视化对比功能")
    print(f"   5. 反向传播机制的完整实现")

if __name__ == "__main__":
    main()