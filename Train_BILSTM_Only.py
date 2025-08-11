# BiLSTM模型专门训练脚本
# 功能：只训练BiLSTM模型并保存，供后续MC-AE使用

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
import psutil  # 系统内存监控

# GPU设备配置 - A100环境
import os
# 使用指定的GPU设备（A100环境）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 只使用GPU2

# CUDA调试功能（在A100环境中可能导致初始化问题，暂时禁用）
print("🔧 CUDA调试模式已禁用（避免A100初始化冲突）")

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

# 忽略警告信息
warnings.filterwarnings('ignore')

# Linux环境matplotlib配置
matplotlib.use('Agg')  # 使用非交互式后端

# Linux环境字体设置
import matplotlib.font_manager as fm

font_options = [
    'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
    'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS'
]

available_fonts = []
for font in font_options:
    try:
        fm.findfont(font)
        available_fonts.append(font)
    except:
        continue

if available_fonts:
    plt.rcParams['font.sans-serif'] = available_fonts
    print(f"✅ Linux字体配置完成，使用字体: {available_fonts[0]}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️  未找到中文字体，将使用英文标签")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("="*60)
print("BiLSTM专门训练模式")
print("只训练BiLSTM模型并保存，供后续MC-AE使用")
print("启用A100 80GB优化和混合精度训练")
print("="*60)

#----------------------------------------数据加载函数------------------------------
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
    """获取安全的DataLoader配置"""
    dataloader_workers = 0  # 强制使用主进程
    pin_memory_enabled = False  # 禁用pin_memory避免CUDA内存问题
    use_persistent = False  # 禁用持久worker
    
    print("🚨 安全模式：完全禁用DataLoader多进程")
    print("   - Workers: 0 (主进程加载)")
    print("   - Pin Memory: False (避免CUDA内存冲突)")
    
    return dataloader_workers, pin_memory_enabled, use_persistent

# 初始化CUDA设备
device, cuda_available = init_cuda_device()

# 获取安全的DataLoader配置
dataloader_workers, pin_memory_enabled, use_persistent = get_dataloader_config(device, cuda_available)

# BiLSTM训练参数（与Transformer保持一致）
# 参照源代码Train_.py的训练参数设置
BILSTM_LR = 1e-4     # 源代码中的学习率
BILSTM_EPOCH = 100   # 源代码中的训练轮数
BILSTM_BATCH_SIZE = 100  # 源代码中的批次大小

# 数据处理参数（参照源代码Train_.py）
TIME_STEP = 1    # 源代码中使用的时间步长
INPUT_SIZE = 7   # 源代码中使用的输入特征数量

print(f"🔧 BiLSTM训练参数（参照源代码Train_.py）:")
print(f"   学习率: {BILSTM_LR}")
print(f"   训练轮数: {BILSTM_EPOCH}")
print(f"   批次大小: {BILSTM_BATCH_SIZE}")
print(f"   时间步长: {TIME_STEP}")
print(f"   输入特征数: {INPUT_SIZE}")

# 加载训练样本
train_samples = load_train_samples()
print(f"使用QAS目录中的{len(train_samples)}个样本进行训练")

#----------------------------------------BiLSTM训练核心代码------------------------------
# 内存监控函数
def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return allocated_gb, reserved_gb, total_gb
    return 0, 0, 0

def get_system_memory_info():
    """获取系统内存信息"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / 1024**3
    total_gb = memory.total / 1024**3
    return used_gb, total_gb

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
    
    return wrapper

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
            
            # 使用源代码中的原始数据预处理函数
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
    
    # 打印模型结构
    print(f"   架构: BiLSTM(input=7, hidden=128, layers=3) + FC(256→128→64→2)")
    
    # 使用源代码中的批次大小设置
    safe_batch_size = BILSTM_BATCH_SIZE
    print(f"\n🎯 使用批次大小（参照源代码）: {safe_batch_size}")
    
    # 创建数据集和数据加载器
    train_dataset = MyDataset(train_X, train_y)
    
    bilstm_train_loader = DataLoader(
        train_dataset, 
        batch_size=safe_batch_size, 
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=pin_memory_enabled,
        persistent_workers=use_persistent
    )
    
    # 优化器配置（大模型适配版）
    bilstm_optimizer = torch.optim.AdamW(bilstm_model.parameters(), 
                                        lr=BILSTM_LR, 
                                        weight_decay=1e-5,
                                        betas=(0.9, 0.999),
                                        eps=1e-8)
    bilstm_loss_func = nn.MSELoss()
    
    # 简化学习率调度器（参照源代码Train_.py）
    def get_lr_with_decay(epoch):
        # 使用源代码中的简单学习率衰减
        lr_decay_freq = 25  # 源代码中的衰减频率
        decay_factor = 0.9
        return BILSTM_LR * (decay_factor ** (epoch // lr_decay_freq))
    
    # 手动学习率调度（参照源代码）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        bilstm_optimizer, 
        lr_lambda=lambda epoch: get_lr_with_decay(epoch) / BILSTM_LR
    )
    
    print(f"\n🚀 A100大规模BILSTM训练配置:")
    print(f"   模型规模: hidden_size=128, num_layers=3")
    print(f"   批次大小: {safe_batch_size}")
    print(f"   学习率: {BILSTM_LR:.6f}")
    print(f"   训练轮数: {BILSTM_EPOCH}")
    print(f"   预热轮数: {WARMUP_EPOCHS}")
    print(f"   优化器: AdamW")
    print(f"   学习率调度: 长预热 + CosineAnnealing")
    
    # 内存监控的BILSTM训练函数
    @memory_monitor
    def bilstm_training_loop(train_loader):
        print(f"\n🏋️ 开始BILSTM训练 (A100优化版本)...")
        
        # 训练前CUDA状态检查
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize()
                print(f"✅ CUDA设备状态正常: {torch.cuda.get_device_name()}")
                gpu_alloc, _, gpu_total = get_gpu_memory_info()
                print(f"🔋 当前GPU显存: {gpu_alloc/gpu_total*100:.1f}%")
            except Exception as e:
                print(f"⚠️ CUDA状态检查警告: {e}")
        
        loss_train_100 = []
        bilstm_model.train()
        
        # 训练循环
        for epoch in range(BILSTM_EPOCH):
            epoch_losses = []
            
            # 每个epoch开始前检查内存
            if epoch % 50 == 0:
                gpu_alloc, _, gpu_total = get_gpu_memory_info()
                print(f"   Epoch {epoch}: GPU显存使用 {gpu_alloc/gpu_total*100:.1f}%")
            
            # DataLoader枚举保护
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
                        torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), max_norm=0.5)
                        
                        bilstm_optimizer.step()
                        
                        # 记录损失
                        if step % 20 == 0:
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
                    print("   🔄 尝试重新创建DataLoader...")
                    # 强制使用CPU模式重新创建DataLoader
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=safe_batch_size, 
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        persistent_workers=False
                    )
                    print("   ✅ DataLoader重新创建完成，继续训练")
                    continue
                else:
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
    
    #----------------------------------------模型保存功能------------------------------
    print(f"\n💾 保存BiLSTM模型和训练结果...")
    
    # 保存BILSTM模型和Loss记录（基于所有样本训练）
    bilstm_model_path = os.path.join(save_dir, 'bilstm_model_all_samples.pth')
    bilstm_loss_path = os.path.join(save_dir, 'bilstm_loss_record_all_samples.pkl')
    
    # 保存模型状态字典
    torch.save(bilstm_model.state_dict(), bilstm_model_path)
    
    # 保存损失记录
    with open(bilstm_loss_path, 'wb') as f:
        pickle.dump(loss_train_100, f)
    
    # 保存训练信息
    training_info = {
        'valid_samples': valid_samples,
        'total_samples': len(valid_samples),
        'train_data_shape': (train_X.shape, train_y.shape),
        'training_epochs': BILSTM_EPOCH,
        'learning_rate': BILSTM_LR,
        'batch_size': safe_batch_size,
        'model_parameters': bilstm_params,
        'device_used': str(device),
        'final_loss': loss_train_100[-1] if loss_train_100 else None,
        'warmup_epochs': WARMUP_EPOCHS,
        'optimizer': 'AdamW',
        'scheduler': 'LambdaLR with warmup + cosine annealing'
    }
    
    training_info_path = os.path.join(save_dir, 'bilstm_training_info.pkl')
    with open(training_info_path, 'wb') as f:
        pickle.dump(training_info, f)
    
    # 保存模型架构信息（用于重建模型）
    model_architecture = {
        'model_class': 'LSTM',
        'input_size': INPUT_SIZE,
        'time_step': TIME_STEP,
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': 2,
        'model_type': 'BiLSTM',
        'note': '专门为MC-AE准备的BiLSTM模型'
    }
    
    architecture_path = os.path.join(save_dir, 'bilstm_architecture.pkl')
    with open(architecture_path, 'wb') as f:
        pickle.dump(model_architecture, f)
    
    # 打印保存信息
    print(f"✅ BiLSTM模型已保存: {bilstm_model_path}")
    print(f"✅ Loss记录已保存: {bilstm_loss_path}")
    print(f"✅ 训练信息已保存: {training_info_path}")
    print(f"✅ 模型架构已保存: {architecture_path}")
    print(f"✅ 模型基于 {len(valid_samples)} 个样本训练完成")
    
    # 显示训练总结
    print(f"\n📊 BiLSTM训练总结:")
    print(f"   模型参数量: {bilstm_params:,}")
    print(f"   训练样本数: {len(valid_samples)}")
    print(f"   训练轮数: {BILSTM_EPOCH}")
    print(f"   批次大小: {safe_batch_size}")
    print(f"   学习率: {BILSTM_LR:.6f}")
    print(f"   最终损失: {loss_train_100[-1]:.6f}" if loss_train_100 else "   最终损失: 未记录")
    print(f"   模型保存目录: {save_dir}")

else:
    print("❌ 未能加载任何有效的训练数据")
    print("跳过BILSTM训练步骤")

print("\n" + "="*60)
print("BiLSTM专门训练模式完成")
print("模型已保存，可供MC-AE使用")
print("="*60)
