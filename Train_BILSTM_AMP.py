# 导入必要的库
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
from datetime import datetime

# 导入混合精度训练工具
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from Function_ import *
from Class_ import CombinedAE
from distributed_utils import (
    setup_distributed, cleanup_distributed, to_distributed,
    create_distributed_loader, is_main_process, barrier, reduce_value
)

# 设置分布式训练
is_distributed, local_rank, world_size = setup_distributed()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 打印GPU信息
if is_main_process():
    print(f"\n🖥️ 分布式训练配置:")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   当前进程: {local_rank}")
    print(f"   总进程数: {world_size}")
    if torch.cuda.is_available():
        print(f"   GPU型号: {torch.cuda.get_device_name(local_rank)}")
        print(f"   GPU显存: {torch.cuda.get_device_properties(local_rank).total_memory/1024**3:.1f}GB")

# 忽略警告
warnings.filterwarnings('ignore')

# Linux环境配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

class MCDataset(Dataset):
    """MC-AE数据集"""
    def __init__(self, x, y, z, q):
        self.x = torch.tensor(x, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)
        self.z = torch.tensor(z, dtype=torch.float64)
        self.q = torch.tensor(q, dtype=torch.float64)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]

def main():
    """主函数 - 分布式训练 + 混合精度版本"""
    try:
        # 确保分布式环境正确初始化
        if not is_distributed and torch.cuda.device_count() > 1:
            print("⚠️ 警告：检测到多个GPU但未启用分布式训练，请使用torchrun启动脚本")
            return
        
        # 加载训练样本（0-200号）
        train_samples = list(range(201))
        if is_main_process():
            print(f"📋 训练样本范围: 0-200")
            print(f"   样本数量: {len(train_samples)}")
        
        # 加载数据
        all_vin2_data = []
        all_vin3_data = []
        
        for sample_id in train_samples:
            vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
            vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
            
            with open(vin2_path, 'rb') as file:
                vin2_data = pickle.load(file)
                all_vin2_data.append(vin2_data)
            
            with open(vin3_path, 'rb') as file:
                vin3_data = pickle.load(file)
                all_vin3_data.append(vin3_data)
        
        # 合并数据
        all_vin2_data = torch.cat(all_vin2_data, dim=0)
        all_vin3_data = torch.cat(all_vin3_data, dim=0)
        
        if is_main_process():
            print(f"\n📥 数据加载完成:")
            print(f"   vin_2数据形状: {all_vin2_data.shape}")
            print(f"   vin_3数据形状: {all_vin3_data.shape}")
        
        # 定义维度
        dim_x, dim_y, dim_z, dim_q = 2, 110, 110, 3
        dim_x2, dim_y2, dim_z2, dim_q2 = 2, 110, 110, 4
        
        # 分离数据
        x_recovered = all_vin2_data[:, :dim_x]
        y_recovered = all_vin2_data[:, dim_x:dim_x + dim_y]
        z_recovered = all_vin2_data[:, dim_x + dim_y: dim_x + dim_y + dim_z]
        q_recovered = all_vin2_data[:, dim_x + dim_y + dim_z:]
        
        x_recovered2 = all_vin3_data[:, :dim_x2]
        y_recovered2 = all_vin3_data[:, dim_x2:dim_x2 + dim_y2]
        z_recovered2 = all_vin3_data[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
        q_recovered2 = all_vin3_data[:, dim_x2 + dim_y2 + dim_z2:]
        
        # 创建数据集和数据加载器
        batch_size = 2048 * world_size  # 增大总批次大小
        
        # MC-AE1数据集
        dataset1 = MCDataset(x_recovered, y_recovered, z_recovered, q_recovered)
        train_loader1, train_sampler1 = create_distributed_loader(
            dataset1,
            batch_size=batch_size // world_size,
            num_workers=4
        )
        
        # MC-AE2数据集
        dataset2 = MCDataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2)
        train_loader2, train_sampler2 = create_distributed_loader(
            dataset2,
            batch_size=batch_size // world_size,
            num_workers=4
        )
        
        if is_main_process():
            print(f"\n📦 分布式数据加载器创建完成:")
            print(f"   总批次大小: {batch_size}")
            print(f"   每GPU批次大小: {batch_size // world_size}")
        
        # 初始化模型
        net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                        activation_fn=custom_activation, use_dx_in_forward=True)
        netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110,
                         activation_fn=torch.sigmoid, use_dx_in_forward=True)
        
        # 转换为分布式模型
        net = to_distributed(net, local_rank)
        netx = to_distributed(netx, local_rank)
        
        if is_main_process():
            print(f"\n🧠 分布式MC-AE模型初始化完成")
            print(f"   MC-AE1参数量: {sum(p.numel() for p in net.parameters()):,}")
            print(f"   MC-AE2参数量: {sum(p.numel() for p in netx.parameters()):,}")
        
        # 训练参数
        num_epochs = 300
        learning_rate = 5e-4 * 4  # 因为批次大小增加4倍，学习率也相应增加
        
        # 优化器
        optimizer1 = optim.Adam(net.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(netx.parameters(), lr=learning_rate)
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 初始化梯度缩放器
        scaler1 = GradScaler()
        scaler2 = GradScaler()
        
        if is_main_process():
            print("\n⚙️  训练参数:")
            print(f"   学习率: {learning_rate}")
            print(f"   训练轮数: {num_epochs}")
            print(f"   批次大小: {batch_size}")
            print(f"   混合精度训练: 启用")
            print("\n" + "="*60)
            print("🎯 开始MC-AE训练")
            print("="*60)
        
        # 训练MC-AE1
        train_losses_mcae1 = []
        best_loss_mcae1 = float('inf')
        
        if is_main_process():
            print("\n🔧 训练第一组MC-AE模型（vin_2）...")
        
        for epoch in range(num_epochs):
            net.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 设置采样器的epoch
            if train_sampler1:
                train_sampler1.set_epoch(epoch)
            
            # 使用tqdm显示进度（仅在主进程）
            if is_main_process():
                pbar = tqdm(train_loader1, desc=f"MC-AE1 Epoch {epoch:3d}")
            else:
                pbar = train_loader1
            
            for x, y, z, q in pbar:
                x = x.to(local_rank if is_distributed else 'cuda')
                y = y.to(local_rank if is_distributed else 'cuda')
                z = z.to(local_rank if is_distributed else 'cuda')
                q = q.to(local_rank if is_distributed else 'cuda')
                
                optimizer1.zero_grad()
                
                # 使用自动混合精度训练
                with autocast():
                    outputs = net(x, z, q)
                    loss = criterion(outputs[0], y)
                
                # 使用梯度缩放器
                scaler1.scale(loss).backward()
                scaler1.step(optimizer1)
                scaler1.update()
                
                # 收集所有进程的损失
                reduced_loss = reduce_value(loss.detach(), average=True)
                epoch_loss += reduced_loss.item()
                batch_count += 1
                
                if is_main_process():
                    pbar.set_postfix({'loss': f"{reduced_loss.item():.6f}"})
            
            # 计算平均损失
            avg_loss = epoch_loss / batch_count
            train_losses_mcae1.append(avg_loss)
            
            # 保存最佳模型（仅主进程）
            if is_main_process() and avg_loss < best_loss_mcae1:
                best_loss_mcae1 = avg_loss
                torch.save(net.module.state_dict() if is_distributed else net.state_dict(),
                         '/mnt/bz25t/bzhy/zhanglikang/project/models/net_model_bilstm_baseline_best.pth')
            
            # 打印训练信息（仅主进程）
            if is_main_process() and epoch % 50 == 0:
                print(f"MC-AE1 Epoch: {epoch:3d} | Average Loss: {avg_loss:.4f}")
        
        # 训练MC-AE2
        train_losses_mcae2 = []
        best_loss_mcae2 = float('inf')
        
        if is_main_process():
            print("\n🔧 训练第二组MC-AE模型（vin_3）...")
        
        for epoch in range(num_epochs):
            netx.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 设置采样器的epoch
            if train_sampler2:
                train_sampler2.set_epoch(epoch)
            
            # 使用tqdm显示进度（仅在主进程）
            if is_main_process():
                pbar = tqdm(train_loader2, desc=f"MC-AE2 Epoch {epoch:3d}")
            else:
                pbar = train_loader2
            
            for x, y, z, q in pbar:
                x = x.to(local_rank if is_distributed else 'cuda')
                y = y.to(local_rank if is_distributed else 'cuda')
                z = z.to(local_rank if is_distributed else 'cuda')
                q = q.to(local_rank if is_distributed else 'cuda')
                
                optimizer2.zero_grad()
                
                # 使用自动混合精度训练
                with autocast():
                    outputs = netx(x, z, q)
                    loss = criterion(outputs[0], y)
                
                # 使用梯度缩放器
                scaler2.scale(loss).backward()
                scaler2.step(optimizer2)
                scaler2.update()
                
                # 收集所有进程的损失
                reduced_loss = reduce_value(loss.detach(), average=True)
                epoch_loss += reduced_loss.item()
                batch_count += 1
                
                if is_main_process():
                    pbar.set_postfix({'loss': f"{reduced_loss.item():.6f}"})
            
            # 计算平均损失
            avg_loss = epoch_loss / batch_count
            train_losses_mcae2.append(avg_loss)
            
            # 保存最佳模型（仅主进程）
            if is_main_process() and avg_loss < best_loss_mcae2:
                best_loss_mcae2 = avg_loss
                torch.save(netx.module.state_dict() if is_distributed else netx.state_dict(),
                         '/mnt/bz25t/bzhy/zhanglikang/project/models/netx_model_bilstm_baseline_best.pth')
            
            # 打印训练信息（仅主进程）
            if is_main_process() and epoch % 50 == 0:
                print(f"MC-AE2 Epoch: {epoch:3d} | Average Loss: {avg_loss:.4f}")
        
        # 保存最终结果（仅主进程）
        if is_main_process():
            # 保存最终模型
            torch.save(net.module.state_dict() if is_distributed else net.state_dict(),
                     '/mnt/bz25t/bzhy/zhanglikang/project/models/net_model_bilstm_baseline.pth')
            torch.save(netx.module.state_dict() if is_distributed else netx.state_dict(),
                     '/mnt/bz25t/bzhy/zhanglikang/project/models/netx_model_bilstm_baseline.pth')
            
            # 保存训练历史
            history = {
                'mcae1_losses': train_losses_mcae1,
                'mcae2_losses': train_losses_mcae2,
                'mcae1_best_loss': best_loss_mcae1,
                'mcae2_best_loss': best_loss_mcae2,
                'batch_size': batch_size,
                'world_size': world_size,
                'amp_enabled': True
            }
            
            with open('/mnt/bz25t/bzhy/zhanglikang/project/models/bilstm_training_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            
            # 绘制训练曲线
            plt.figure(figsize=(15, 6))
            
            # MC-AE1损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(train_losses_mcae1, label='MC-AE1 Loss')
            plt.title('MC-AE1 Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            
            # MC-AE2损失曲线
            plt.subplot(1, 2, 2)
            plt.plot(train_losses_mcae2, label='MC-AE2 Loss')
            plt.title('MC-AE2 Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('/mnt/bz25t/bzhy/zhanglikang/project/models/bilstm_training_results.png')
            plt.close()
            
            print("\n💾 保存结果:")
            print("   ✓ MC-AE1模型: net_model_bilstm_baseline.pth")
            print("   ✓ MC-AE2模型: netx_model_bilstm_baseline.pth")
            print("   ✓ 最佳MC-AE1: net_model_bilstm_baseline_best.pth")
            print("   ✓ 最佳MC-AE2: netx_model_bilstm_baseline_best.pth")
            print("   ✓ 训练历史: bilstm_training_history.pkl")
            print("   ✓ 训练曲线: bilstm_training_results.png")
            
            print("\n📊 训练结果:")
            print(f"   MC-AE1最终损失: {train_losses_mcae1[-1]:.6f}")
            print(f"   MC-AE2最终损失: {train_losses_mcae2[-1]:.6f}")
            print(f"   MC-AE1最佳损失: {best_loss_mcae1:.6f}")
            print(f"   MC-AE2最佳损失: {best_loss_mcae2:.6f}")
    
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        raise e
    
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    main()