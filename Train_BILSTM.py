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
import psutil  # 系统内存监控

# GPU设备配置 - A100环境
import os
# 使用指定的GPU设备（A100环境）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 只使用GPU2

# CUDA调试功能（在A100环境中可能导致初始化问题，暂时禁用）
# os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 可能导致A100初始化错误
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 仅在需要详细调试时启用
print("🔧 CUDA调试模式已禁用（避免A100初始化冲突）")

# device将在init_cuda_device()函数中初始化

# 打印GPU信息
if torch.cuda.is_available():
    print("\n🖥️ A100 GPU配置信息:")
    print(f"   可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i} ({props.name}):")
        print(f"      总显存: {props.total_memory/1024**3:.1f}GB")
    print(f"\n   当前使用: 仅GPU2 (A100 80GB优化)")
    print(f"   主GPU设备: cuda:0 (物理GPU2)")
    print(f"   备注: 单卡A100优化，充分利用80GB显存")
else:
    print("⚠️  未检测到GPU，使用CPU训练")

# 中文注释：忽略警告信息
warnings.filterwarnings('ignore')

# Linux环境matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Linux环境字体设置 - 修复中文显示问题
import matplotlib.font_manager as fm

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
    print(f"✅ Linux字体配置完成，使用字体: {available_fonts[0]}")
else:
    # 如果都不可用，使用英文标签
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️  未找到中文字体，将使用英文标签")

plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

#----------------------------------------BiLSTM A100优化训练配置------------------------------
print("="*50)
print("BiLSTM A100优化训练模式")
print("直接使用原始vin_2[x[0]]和vin_3[x[0]]数据")
print("跳过Transformer训练，直接进行MC-AE训练")
print("启用A100 80GB优化和混合精度训练")
print("="*50)

#----------------------------------------数据加载------------------------------
# 从Labels.xls加载训练样本ID（0-200号）
def load_train_samples():
    """从Labels.xls加载训练样本ID"""
    try:
        import pandas as pd
        # Linux路径格式
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取0-100范围的样本
        all_samples = df['Num'].tolist()
        train_samples = [i for i in all_samples if 0 <= i <= 100]
        
        print(f"📋 从Labels.xls加载训练样本:")
        print(f"   训练样本范围: 0-100")
        print(f"   实际可用样本: {len(train_samples)} 个")
        print(f"   样本ID: {train_samples[:10]}..." if len(train_samples) > 10 else f"   样本ID: {train_samples}")
        
        return train_samples
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认样本范围 0-20")
        return list(range(21))

train_samples = load_train_samples()
print(f"使用QAS目录中的{len(train_samples)}个样本进行训练")

# 统一的路径配置 - Linux环境
data_dir = '/mnt/bz25t/bzhy/zhanglikang/project/QAS'  # 数据目录
save_dir = '/mnt/bz25t/bzhy/datasave/BILSTM_train'  # 模型保存目录

# 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ 创建模型保存目录: {save_dir}")
else:
    print(f"✅ 模型保存目录已存在: {save_dir}")

# CUDA设备检查和初始化
def init_cuda_device():
    """安全的CUDA设备初始化"""
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，将使用CPU模式")
            return torch.device('cpu'), False
        
        # 检查CUDA设备数量
        device_count = torch.cuda.device_count()
        print(f"🚀 检测到 {device_count} 个CUDA设备")
        
        # 设置默认设备
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        
        # 测试CUDA初始化
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        
        # 获取设备信息
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / 1024**3
        print(f"🖥️  GPU: {props.name}")
        print(f"💾 GPU内存: {memory_gb:.1f}GB")
        
        return device, True
        
    except Exception as e:
        print(f"❌ CUDA初始化失败: {e}")
        print("🔄 回退到CPU模式")
        return torch.device('cpu'), False

def get_dataloader_config(device, cuda_available):
    """
    获取安全的DataLoader配置 - 完全禁用多进程避免CUDA初始化问题
    """
    # 🚨 紧急修复：完全禁用多进程DataLoader
    # 这是最安全的配置，避免所有CUDA worker进程问题
    dataloader_workers = 0  # 强制使用主进程，无多进程
    pin_memory_enabled = False  # 禁用pin_memory避免CUDA内存问题
    use_persistent = False  # 禁用持久worker
    
    print("🚨 紧急修复模式：完全禁用DataLoader多进程")
    print("   - Workers: 0 (主进程加载)")
    print("   - Pin Memory: False (避免CUDA内存冲突)")
    print("   - 性能影响: 轻微，但确保稳定性")
    
    return dataloader_workers, pin_memory_enabled, use_persistent

# 初始化CUDA设备
device, cuda_available = init_cuda_device()

# 获取安全的DataLoader配置
dataloader_workers, pin_memory_enabled, use_persistent = get_dataloader_config(device, cuda_available)

# 显示修复状态
print("🔧 CUDA错误修复状态:")
print(f"   - CUDA调试模式: 已禁用（避免A100冲突）")
print(f"   - DataLoader workers: {dataloader_workers}")
print(f"   - Pin memory: {pin_memory_enabled}")
print(f"   - Persistent workers: {use_persistent}")
print("   - 批次大小: 已优化为安全级别")

# MC-AE训练参数（与Transformer保持一致）
EPOCH = 500  # 与Transformer保持一致的MC-AE训练轮数
INIT_LR = 2e-5  # 与Transformer保持一致的初始学习率
MAX_LR = 1e-4   # 与Transformer保持一致的最大学习率

# 根据GPU内存动态调整批次大小 - A100优化（更保守的设置）
if cuda_available:
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 80:  # A100 80GB
        BATCHSIZE = 2000  # 与Transformer保持一致的MC-AE批次大小
    elif gpu_memory_gb >= 40:  # A100 40GB
        BATCHSIZE = 1000
    elif gpu_memory_gb >= 24:  # A100 24GB
        BATCHSIZE = 500
    elif gpu_memory_gb >= 16:  # V100 16GB
        BATCHSIZE = 300
    else:  # 其他GPU
        BATCHSIZE = 150
    print(f"🖥️  设置批次大小: {BATCHSIZE}")
else:
    BATCHSIZE = 50  # CPU模式使用更小的批次
    print("⚠️  CPU模式，批次大小: 50")

WARMUP_EPOCHS = 50  # 与Transformer保持一致的预热轮数

# 添加梯度裁剪
MAX_GRAD_NORM = 1.0  # 调整到更合理的梯度裁剪阈值
MIN_GRAD_NORM = 0.1  # 最小梯度范数阈值

# A100 80GB内存优化参数
MEMORY_CHECK_INTERVAL = 25  # 更频繁检查内存（每25个批次）
CLEAR_CACHE_INTERVAL = 50   # 更频繁清理缓存（每50个批次）
MAX_MEMORY_THRESHOLD = 0.85  # 内存使用率超过85%时采取措施（A100单卡保守策略）
EMERGENCY_MEMORY_THRESHOLD = 0.95  # 内存使用率超过95%时紧急处理

# 学习率预热函数
def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return INIT_LR + (MAX_LR - INIT_LR) * epoch / WARMUP_EPOCHS
    return MAX_LR * (0.9 ** (epoch // 50))  # 每50个epoch衰减到90%

# 内存监控函数
def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            usage_ratio = allocated / total
            print(f"   GPU {i}: {allocated:.1f}GB / {cached:.1f}GB / {total:.1f}GB (已用/缓存/总计) - {usage_ratio*100:.1f}%")
            return usage_ratio
    return 0.0

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   🧹 GPU缓存已清理")

# 显示优化后的训练参数
print(f"\n⚙️  BiLSTM训练参数（A100单卡优化版本）:")
print(f"   批次大小: {BATCHSIZE} (充分利用A100 80GB显存)")
print(f"   训练轮数: {EPOCH}")
print(f"   初始学习率: {INIT_LR}")
print(f"   最大学习率: {MAX_LR}")
print(f"   GPU配置: 单卡A100 80GB (GPU2)")
print(f"   混合精度: 启用 (AMP)")
print(f"   内存监控: 启用 (每{MEMORY_CHECK_INTERVAL}批次检查)")
print(f"   缓存清理: 启用 (每{CLEAR_CACHE_INTERVAL}批次清理)")
print(f"   内存阈值: {MAX_MEMORY_THRESHOLD*100:.0f}% (A100单卡保守策略)")

#----------------------------------------BILSTM训练（观察Loss下降）------------------------
print("="*50)
print("阶段0: BILSTM训练（观察Loss下降情况）")
print("="*50)

# A100 GPU优化参数配置（安全版本）
TIME_STEP = 1  # rnn time step
INPUT_SIZE = 7  # rnn input size

# 显存和内存安全监控函数
def get_gpu_memory_info():
    """获取GPU显存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0

def get_system_memory_info():
    """获取系统内存信息"""
    import psutil
    memory = psutil.virtual_memory()
    used_gb = memory.used / 1024**3
    total_gb = memory.total / 1024**3
    return used_gb, total_gb

def safe_batch_size_calculator(model, sample_data, max_batch_size=4096, safety_margin=0.2):
    """安全的批次大小计算器"""
    print(f"\n🔍 正在计算安全的批次大小...")
    
    # 获取当前显存状态
    allocated_before, reserved_before, total_gpu = get_gpu_memory_info()
    print(f"   当前显存: {allocated_before:.1f}GB / {total_gpu:.1f}GB")
    
    # 二分法查找最大安全批次大小
    min_batch = 32
    max_batch = max_batch_size
    safe_batch = min_batch
    
    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2
        try:
            # 创建测试批次
            if len(sample_data.shape) == 3:  # (samples, time, features)
                test_input = sample_data[:test_batch].clone().to(device)
            else:  # 其他形状
                test_input = sample_data[:test_batch].clone().to(device)
            
            # 测试前向传播
            with torch.no_grad():
                model.eval()
                _ = model(test_input)
            
            # 检查显存使用
            allocated_after, _, _ = get_gpu_memory_info()
            memory_usage = allocated_after / total_gpu
            
            if memory_usage < (1.0 - safety_margin):  # 安全阈值
                safe_batch = test_batch
                min_batch = test_batch + 1
                print(f"   ✅ 批次 {test_batch}: 显存使用 {memory_usage*100:.1f}% - 安全")
            else:
                max_batch = test_batch - 1
                print(f"   ⚠️  批次 {test_batch}: 显存使用 {memory_usage*100:.1f}% - 超限")
            
            # 清理测试数据
            del test_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                max_batch = test_batch - 1
                print(f"   ❌ 批次 {test_batch}: 显存溢出")
                torch.cuda.empty_cache()
            else:
                raise e
        except Exception as e:
            print(f"   ⚠️  批次 {test_batch}: 测试失败 - {str(e)}")
            max_batch = test_batch - 1
    
    return safe_batch

# 内存监控装饰器
def memory_monitor(func):
    """内存监控装饰器"""
    def wrapper(*args, **kwargs):
        # 训练前检查
        gpu_alloc, gpu_reserved, gpu_total = get_gpu_memory_info()
        sys_used, sys_total = get_system_memory_info()
        
        print(f"\n📊 训练前内存状态:")
        print(f"   GPU显存: {gpu_alloc:.1f}GB / {gpu_total:.1f}GB ({gpu_alloc/gpu_total*100:.1f}%)")
        print(f"   系统内存: {sys_used:.1f}GB / {sys_total:.1f}GB ({sys_used/sys_total*100:.1f}%)")
        
        # 安全检查
        if gpu_alloc/gpu_total > 0.85:
            print("⚠️  警告: GPU显存使用率过高，建议清理缓存")
            torch.cuda.empty_cache()
        
        if sys_used/sys_total > 0.90:
            print("⚠️  警告: 系统内存使用率过高")
        
        try:
            result = func(*args, **kwargs)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ 显存不足，正在清理缓存...")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
        finally:
            # 训练后状态
            gpu_alloc_after, _, _ = get_gpu_memory_info()
            sys_used_after, _ = get_system_memory_info()
            print(f"\n📊 训练后内存状态:")
            print(f"   GPU显存: {gpu_alloc_after:.1f}GB / {gpu_total:.1f}GB")
            print(f"   系统内存: {sys_used_after:.1f}GB / {sys_total:.1f}GB")
    
    return wrapper

# A100优化参数（大规模BILSTM适配版 - 稳定训练）
BILSTM_LR = 5e-5  # 与Transformer保持一致的学习率
BILSTM_EPOCH = 800  # 与Transformer保持一致的完整训练轮数
BILSTM_BATCH_SIZE_TARGET = 512  # 与Transformer保持一致的批次大小

# 大模型训练的额外参数（保守策略）
WARMUP_EPOCHS = 5  # 测试配置：快速预热（原值：50）
GRADIENT_CLIP = 0.5  # 更严格的梯度裁剪，防止梯度爆炸
WEIGHT_DECAY = 1e-5  # 降低权重衰减，避免过度正则化

print(f"A100单卡GPU优化配置（安全模式）:")
print(f"   时间步长: {TIME_STEP}")
print(f"   输入维度: {INPUT_SIZE}")
print(f"   学习率: {BILSTM_LR} (针对A100优化)")
print(f"   训练轮数: {BILSTM_EPOCH} (充分训练)")
print(f"   目标批次大小: {BILSTM_BATCH_SIZE_TARGET} (将动态调整)")

# 加载所有样本的vin_1数据进行BILSTM训练
print(f"\n🔄 加载所有{len(train_samples)}个样本的vin_1数据进行BILSTM训练")

# 收集所有样本的数据
all_train_X = []
all_train_y = []
valid_samples = []

for idx, sample_id in enumerate(train_samples):
    vin_1_file = os.path.join(data_dir, f'{sample_id}', 'vin_1.pkl')
    if os.path.exists(vin_1_file):
        try:
            with open(vin_1_file, 'rb') as file:
                sample_data = pickle.load(file)
            
            # 使用源代码的prepare_training_data函数
            sample_train_X, sample_train_y = prepare_training_data(sample_data, INPUT_SIZE, TIME_STEP, device)
            
            all_train_X.append(sample_train_X)
            all_train_y.append(sample_train_y)
            valid_samples.append(sample_id)
            
            if (idx + 1) % 10 == 0:
                print(f"   ✓ 已加载 {idx + 1}/{len(train_samples)} 个样本")
                
        except Exception as e:
            print(f"   ❌ 样本 {sample_id} 加载失败: {e}")
            continue
    else:
        print(f"   ⚠️  样本 {sample_id} 的vin_1.pkl文件不存在")

if len(all_train_X) > 0:
    # 合并所有样本的数据
    train_X = torch.cat(all_train_X, dim=0)
    train_y = torch.cat(all_train_y, dim=0)
    
    print(f"✅ 成功加载 {len(valid_samples)} 个样本的数据")
    print(f"   有效样本ID: {valid_samples}")
    print(f"   合并后训练数据形状:")
    print(f"   输入形状 (train_X): {train_X.shape}")
    print(f"   目标形状 (train_y): {train_y.shape}")
    print(f"   预测目标: 下一时刻索引5和6的数据")
else:
    print("❌ 未能加载任何有效的训练数据")
    print("跳过BILSTM训练步骤")

if len(all_train_X) > 0:
    
    # 创建BILSTM模型（用于批次大小计算）
    try:
        bilstm_model = LSTM()
        if cuda_available:
            bilstm_model = bilstm_model.to(device)
        bilstm_model = bilstm_model.double()
        print(f"✅ BILSTM模型已移动到设备: {device}")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"❌ CUDA设备错误: {e}")
            print("🔄 自动切换到CPU模式")
            device = torch.device('cpu')
            cuda_available = False
            bilstm_model = LSTM().to(device).double()
        else:
            raise e
    
    # 应用专门的大模型权重初始化
    def initialize_bilstm_weights(model):
        """大规模BILSTM专用权重初始化"""
        for name, param in model.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # LSTM权重使用Xavier初始化，降低方差
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'bias' in name:
                # 偏置初始化为0，forget gate偏置设为1
                nn.init.zeros_(param.data)
                if 'bias_hh' in name:
                    # forget gate偏置设为1，帮助记忆
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'weight' in name and 'fc' in name:
                # 全连接层权重
                nn.init.xavier_uniform_(param.data, gain=0.3)
            elif 'bias' in name and 'fc' in name:
                nn.init.zeros_(param.data)
        print("✅ 应用大规模BILSTM专用权重初始化")
    
    initialize_bilstm_weights(bilstm_model)
    
    # 统计模型参数量
    bilstm_params = bilstm_model.count_parameters()
    print(f"\n📊 BILSTM模型参数统计:")
    print(f"   总参数量: {bilstm_params:,}")
    print(f"   模型规模: {bilstm_params/1e6:.2f}M 参数")
    print(f"   对比Transformer: 约 {bilstm_params/920000:.2f}x 规模")
    
    # 打印模型结构
    print(f"   架构: BiLSTM(input=7, hidden=128, layers=3) + FC(256→128→64→2)")
    print(f"   匹配目标: Transformer(d_model=128, layers=3) ≈ 0.92M参数")
    
    # 安全计算最优批次大小
    safe_batch_size = safe_batch_size_calculator(bilstm_model, train_X, BILSTM_BATCH_SIZE_TARGET, safety_margin=0.2)
    print(f"\n🎯 确定安全批次大小: {safe_batch_size}")
    print(f"   原目标: {BILSTM_BATCH_SIZE_TARGET}")
    print(f"   实际使用: {safe_batch_size}")
    
    # 创建数据集和数据加载器（使用全局安全配置）
    train_dataset = MyDataset(train_X, train_y)
    
    bilstm_train_loader = DataLoader(
        train_dataset, 
        batch_size=safe_batch_size, 
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=pin_memory_enabled,
        persistent_workers=use_persistent,
        prefetch_factor=2 if dataloader_workers > 0 else None
    )
    
    # 优化器配置（大模型适配版）
    # 为大模型使用更保守的学习率策略
    actual_lr = BILSTM_LR  # 不再线性缩放，使用固定学习率
    bilstm_optimizer = torch.optim.AdamW(bilstm_model.parameters(), 
                                        lr=actual_lr, 
                                        weight_decay=WEIGHT_DECAY,  # 使用更保守的权重衰减
                                        betas=(0.9, 0.999),  # 标准beta值
                                        eps=1e-8)  # 数值稳定性
    bilstm_loss_func = nn.MSELoss()
    
    # 学习率调度器（支持预热）
    def get_lr_with_warmup(epoch):
        if epoch < WARMUP_EPOCHS:
            # 预热阶段：从0线性增加到目标学习率
            return actual_lr * (epoch + 1) / WARMUP_EPOCHS
        else:
            # 余弦退火阶段
            cos_epoch = epoch - WARMUP_EPOCHS
            cos_max = BILSTM_EPOCH - WARMUP_EPOCHS
            return actual_lr * 0.5 * (1 + np.cos(np.pi * cos_epoch / cos_max))
    
    # 手动学习率调度（更精确控制）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        bilstm_optimizer, 
        lr_lambda=lambda epoch: get_lr_with_warmup(epoch) / actual_lr
    )
    
    print(f"\n🚀 A100大规模BILSTM训练配置 (稳定版):")
    print(f"   模型规模: hidden_size=128, num_layers=3 (匹配Transformer)")
    print(f"   批次大小: {safe_batch_size} (降低以提高稳定性)")
    print(f"   学习率: {actual_lr:.6f} (大幅降低，确保稳定)")
    print(f"   训练轮数: {BILSTM_EPOCH} (增加以补偿低学习率)")
    print(f"   预热轮数: {WARMUP_EPOCHS} (长预热期)")
    print(f"   梯度裁剪: {GRADIENT_CLIP} (严格控制)")
    print(f"   权重衰减: {WEIGHT_DECAY} (保守正则化)")
    print(f"   优化器: AdamW (数值稳定配置)")
    print(f"   学习率调度: 长预热 + CosineAnnealing")
    print(f"   权重初始化: 专用大模型初始化")
    print(f"   数据加载进程: 8")
    
    # 内存监控的BILSTM训练函数
    @memory_monitor
    def bilstm_training_loop(train_loader):
        print(f"\n🏋️ 开始BILSTM训练 (A100优化版本)...")
        
        # 🚨 训练前CUDA状态检查
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize()  # 确保CUDA操作完成
                print(f"✅ CUDA设备状态正常: {torch.cuda.get_device_name()}")
                gpu_alloc, _, gpu_total = get_gpu_memory_info()
                print(f"🔋 当前GPU显存: {gpu_alloc/gpu_total*100:.1f}%")
            except Exception as e:
                print(f"⚠️ CUDA状态检查警告: {e}")
                print("🔄 继续使用当前配置...")
        
        loss_train_100 = []
        bilstm_model.train()
        
        # 训练循环
        for epoch in range(BILSTM_EPOCH):
            epoch_losses = []
            
            # 每个epoch开始前检查内存
            if epoch % 50 == 0:
                gpu_alloc, _, gpu_total = get_gpu_memory_info()
                print(f"   Epoch {epoch}: GPU显存使用 {gpu_alloc/gpu_total*100:.1f}%")
            
            # 🚨 DataLoader枚举保护
            try:
                for step, (b_x, b_y) in enumerate(train_loader):
                    try:
                        # 前向传播
                        output = bilstm_model(b_x)
                        loss = bilstm_loss_func(b_y, output)
                        
                        # 反向传播
                        bilstm_optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪（防止梯度爆炸）
                        torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), max_norm=GRADIENT_CLIP)
                        
                        bilstm_optimizer.step()
                        
                        # 记录损失
                        if step % 20 == 0:  # 更频繁的记录
                            loss_train_100.append(loss.cpu().detach().numpy())
                            epoch_losses.append(loss.item())
                        
                        # 定期清理缓存
                        if step % 100 == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"   ⚠️  Epoch {epoch}, Step {step}: 显存不足，清理缓存后继续")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            except RuntimeError as e:
                print(f"🚨 DataLoader错误 (Epoch {epoch}): {e}")
                if "initialization error" in str(e).lower():
                    print("   原因: CUDA初始化失败")
                    print("   🔄 尝试重新创建DataLoader...")
                    # 强制使用CPU模式重新创建DataLoader
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=safe_batch_size, 
                        shuffle=True,
                        num_workers=0,  # 强制单进程
                        pin_memory=False,
                        persistent_workers=False
                    )
                    print("   ✅ DataLoader重新创建完成，继续训练")
                    # 重新尝试这个epoch
                    epoch -= 1  # 重新尝试当前epoch
                    continue
                else:
                    print(f"   ❌ 未知DataLoader错误: {e}")
                    raise e
            
            # 更新学习率
            scheduler.step()
            
            # 定期输出训练状态
            if epoch % 20 == 0:
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                current_lr = scheduler.get_last_lr()[0]
                print(f"   Epoch {epoch:3d}/{BILSTM_EPOCH}: Loss={avg_loss:.6f}, LR={current_lr:.6f}")
        
        print(f"✅ BILSTM训练完成，共记录 {len(loss_train_100)} 个Loss值")
        return loss_train_100
    
    # 执行训练
    loss_train_100 = bilstm_training_loop(bilstm_train_loader)
    
    # 保存BILSTM模型和Loss记录（基于所有样本训练）
    bilstm_model_path = os.path.join(save_dir, 'bilstm_model_all_samples.pth')
    bilstm_loss_path = os.path.join(save_dir, 'bilstm_loss_record_all_samples.pkl')
    
    torch.save(bilstm_model.state_dict(), bilstm_model_path)
    with open(bilstm_loss_path, 'wb') as f:
        pickle.dump(loss_train_100, f)
    
    # 保存训练信息
    training_info = {
        'valid_samples': valid_samples,
        'total_samples': len(valid_samples),
        'train_data_shape': (train_X.shape, train_y.shape),
        'training_epochs': BILSTM_EPOCH,
        'learning_rate': BILSTM_LR,
        'final_loss': loss_train_100[-1] if loss_train_100 else None
    }
    
    training_info_path = os.path.join(save_dir, 'bilstm_training_info.pkl')
    with open(training_info_path, 'wb') as f:
        pickle.dump(training_info, f)
    
    print(f"✅ BILSTM模型已保存: {bilstm_model_path}")
    print(f"✅ Loss记录已保存: {bilstm_loss_path}")
    print(f"✅ 训练信息已保存: {training_info_path}")
    print(f"✅ 模型基于 {len(valid_samples)} 个样本训练完成")

else:
    print("❌ 未能加载任何有效的训练数据")
    print("跳过BILSTM训练步骤")

print("\n" + "="*50)

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
    vin2_path = os.path.join(data_dir, str(sample_id), 'vin_2.pkl')
    vin3_path = os.path.join(data_dir, str(sample_id), 'vin_3.pkl')
    
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

# A100安全批次大小计算（MC-AE1）
print(f"\n🔍 MC-AE1: 计算安全批次大小...")
print(f"   原设定批次大小: {BATCHSIZE}")

# 创建临时模型用于批次大小测试
temp_net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device).to(torch.float32)

# 创建测试数据样本
test_sample_size = min(BATCHSIZE, len(x_recovered))
sample_x = x_recovered[:test_sample_size]
sample_y = y_recovered[:test_sample_size] 
sample_z = z_recovered[:test_sample_size]
sample_q = q_recovered[:test_sample_size]

# 使用安全批次大小计算器（修改版本适配MC-AE）
def safe_mcae_batch_calculator(model, x, y, z, q, max_batch_size, safety_margin=0.2):
    """MC-AE专用安全批次大小计算器"""
    print(f"   正在测试MC-AE批次大小...")
    
    allocated_before, _, total_gpu = get_gpu_memory_info()
    print(f"   当前显存: {allocated_before:.1f}GB / {total_gpu:.1f}GB")
    
    min_batch = 32
    max_batch = min(max_batch_size, len(x))
    safe_batch = min_batch
    
    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2
        try:
            # 创建测试批次
            test_x = x[:test_batch].to(device)
            test_y = y[:test_batch].to(device)
            test_z = z[:test_batch].to(device)
            test_q = q[:test_batch].to(device)
            
            # 测试前向传播
            with torch.no_grad():
                model.eval()
                _, _ = model(test_x, test_z, test_q)
            
            # 检查显存使用
            allocated_after, _, _ = get_gpu_memory_info()
            memory_usage = allocated_after / total_gpu
            
            if memory_usage < (1.0 - safety_margin):
                safe_batch = test_batch
                min_batch = test_batch + 1
                print(f"   ✅ 批次 {test_batch}: 显存 {memory_usage*100:.1f}% - 安全")
            else:
                max_batch = test_batch - 1
                print(f"   ⚠️  批次 {test_batch}: 显存 {memory_usage*100:.1f}% - 超限")
            
            # 清理
            del test_x, test_y, test_z, test_q
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                max_batch = test_batch - 1
                print(f"   ❌ 批次 {test_batch}: 显存溢出")
                torch.cuda.empty_cache()
            else:
                raise e
    
    return safe_batch

# 计算MC-AE1的安全批次大小
safe_mcae1_batch = safe_mcae_batch_calculator(temp_net, sample_x, sample_y, sample_z, sample_q, BATCHSIZE, safety_margin=0.25)
print(f"🎯 MC-AE1 安全批次大小: {safe_mcae1_batch}")

# 清理临时模型
del temp_net
torch.cuda.empty_cache()

# 中文注释：用DataLoader批量加载多通道特征数据（安全配置）
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), 
                           batch_size=safe_mcae1_batch, shuffle=False, 
                           num_workers=dataloader_workers, pin_memory=pin_memory_enabled)

# 中文注释：初始化MC-AE模型（使用float32）
try:
    net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True)
    if cuda_available:
        net = net.to(device)
    net = net.to(torch.float32)
    print(f"✅ MC-AE1模型已移动到设备: {device}")
    
    netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True)
    if cuda_available:
        netx = netx.to(device)
    netx = netx.to(torch.float32)
    print(f"✅ MC-AE2模型已移动到设备: {device}")
    
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"❌ MC-AE模型CUDA初始化失败: {e}")
        print("🔄 自动切换到CPU模式")
        device = torch.device('cpu')
        cuda_available = False
        
        # 重新创建模型到CPU
        net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device).to(torch.float32)
        netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device).to(torch.float32)
        print("✅ MC-AE模型已移动到CPU设备")
    else:
        raise e

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

# A100单卡优化配置
print("✅ 使用单卡A100优化模式")
print(f"   GPU设备: {device}")
print(f"   显存优化: 针对80GB显存特别优化")

optimizer = torch.optim.Adam(net.parameters(), lr=INIT_LR)
l1_lambda = 0.01
loss_f = nn.MSELoss()

# 启用混合精度训练
scaler = torch.cuda.amp.GradScaler()
print("✅ 启用混合精度训练 (AMP)")
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    
    # 更新学习率
    current_lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # 每个epoch开始时清理缓存
    clear_gpu_cache()
    
        for iteration, (x, y, z, q) in enumerate(train_loader_u):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            q = q.to(device)
            
            # 内存监控 - 定期检查内存使用情况
            if iteration % MEMORY_CHECK_INTERVAL == 0:
                memory_usage = check_gpu_memory()
                if memory_usage > EMERGENCY_MEMORY_THRESHOLD:
                    print(f"🚨  内存使用率过高 ({memory_usage*100:.1f}%)，紧急清理缓存...")
                    clear_gpu_cache()
                    torch.cuda.synchronize()  # 强制同步
                    if memory_usage > 0.98:  # 如果仍然过高，跳过此批次
                        print(f"🚨  内存使用率仍然过高，跳过此批次")
                        continue
                elif memory_usage > MAX_MEMORY_THRESHOLD:
                    print(f"⚠️  内存使用率较高 ({memory_usage*100:.1f}%)，清理缓存...")
                    clear_gpu_cache()
            
            # 定期清理缓存
            if iteration % CLEAR_CACHE_INTERVAL == 0:
                clear_gpu_cache()
            
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
            
            # 使用混合精度训练（带CUDA错误处理）
            try:
                with torch.cuda.amp.autocast():
                    recon_im, recon_p = net(x, z, q)
                    loss_u = loss_f(y, recon_im)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"🚨 CUDA运行时错误: {e}")
                    print("尝试清理GPU缓存并继续...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue
                else:
                    raise e
                    
            # 检查损失值是否为NaN
            if torch.isnan(loss_u) or torch.isinf(loss_u):
                print(f"警告：第{epoch}轮第{iteration}批次检测到NaN/Inf损失值")
                print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
                print(f"输出范围: [{recon_im.min():.4f}, {recon_im.max():.4f}]")
                print(f"损失值: {loss_u.item()}")
                print("跳过此批次，不进行反向传播")
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
                
                # 执行优化器步骤
                scaler.step(optimizer)
                scaler.update()
                
            except Exception as e:
                print(f"优化器步骤失败: {e}")
                print("跳过此批次并重置scaler状态")
                # 重置scaler状态
                scaler.update()
                continue
            
            # 及时释放不需要的张量
            del x, y, z, q, recon_im, recon_p, loss_u
    
    avg_loss = total_loss / num_batches
    train_losses_mcae1.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE1 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))

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

# A100安全批次大小计算（MC-AE2）
print(f"\n🔍 MC-AE2: 计算安全批次大小...")

# 计算MC-AE2的安全批次大小
test_sample_size2 = min(BATCHSIZE, len(x_recovered2))
sample_x2 = x_recovered2[:test_sample_size2]
sample_y2 = y_recovered2[:test_sample_size2]
sample_z2 = z_recovered2[:test_sample_size2]
sample_q2 = q_recovered2[:test_sample_size2]

safe_mcae2_batch = safe_mcae_batch_calculator(netx, sample_x2, sample_y2, sample_z2, sample_q2, BATCHSIZE, safety_margin=0.25)
print(f"🎯 MC-AE2 安全批次大小: {safe_mcae2_batch}")

# 中文注释：第二组特征的MC-AE训练（安全配置）
train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), 
                             batch_size=safe_mcae2_batch, shuffle=False,
                             num_workers=dataloader_workers, pin_memory=pin_memory_enabled)
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
    
    # 每个epoch开始时清理缓存
    clear_gpu_cache()
    
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        
        # 内存监控 - 定期检查内存使用情况
        if iteration % MEMORY_CHECK_INTERVAL == 0:
            memory_usage = check_gpu_memory()
            if memory_usage > EMERGENCY_MEMORY_THRESHOLD:
                print(f"🚨  内存使用率过高 ({memory_usage*100:.1f}%)，紧急清理缓存...")
                clear_gpu_cache()
                torch.cuda.synchronize()  # 强制同步
                if memory_usage > 0.98:  # 如果仍然过高，跳过此批次
                    print(f"🚨  内存使用率仍然过高，跳过此批次")
                    continue
            elif memory_usage > MAX_MEMORY_THRESHOLD:
                print(f"⚠️  内存使用率较高 ({memory_usage*100:.1f}%)，清理缓存...")
                clear_gpu_cache()
        
        # 定期清理缓存
        if iteration % CLEAR_CACHE_INTERVAL == 0:
            clear_gpu_cache()
        
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
            
            # 执行优化器步骤
            scaler2.step(optimizer)
            scaler2.update()
            
        except Exception as e:
            print(f"优化器步骤失败: {e}")
            print("跳过此批次并重置scaler状态")
            # 重置scaler状态
            scaler2.update()
            continue
        
        # 及时释放不需要的张量
        del x, y, z, q, recon_im, loss_x
    
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    train_losses_mcae2.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE2 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))

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

# 使用统一的保存目录
result_dir = save_dir
print(f"📁 结果保存目录: {result_dir}")

# 中文注释：诊断特征提取与PCA分析
df_data = DiagnosisFeature(ERRORU,ERRORX)

v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data,0.95,0.95)

# 训练结束后自动保存模型和分析结果
print("="*50)
print("保存BiLSTM基准训练结果")
print("="*50)

# 绘制训练结果
print("📈 绘制BiLSTM训练曲线...")

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

# 结果目录已在前面创建，无需重复检查

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
    'learning_rate': INIT_LR, # Changed from LR to INIT_LR
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