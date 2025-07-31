# 导入必要的库
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
from datetime import datetime

# 导入自定义模块
from data_loader_transformer import TransformerBatteryDataset
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

class TransformerPredictor(nn.Module):
    """时序预测Transformer模型 - 直接预测真实物理值"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=3, output_size=2):
        super(TransformerPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: [batch, input_size]
        x = x.unsqueeze(1)  # [batch, 1, input_size]
        
        # 投影到d_model维度
        x = self.input_projection(x)  # [batch, 1, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoding[:x.size(1)].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)  # [batch, 1, d_model]
        
        # 输出层
        x = x.squeeze(1)  # [batch, d_model]
        out = self.output_layer(x)  # [batch, output_size]
        
        return out

def main():
    """主函数 - 分布式训练版本"""
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
        
        # 创建数据集
        dataset = TransformerBatteryDataset(
            data_path='/mnt/bz25t/bzhy/zhanglikang/project/QAS',
            sample_ids=train_samples
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空，请检查数据加载！")
            return
            
        if is_main_process():
            print(f"\n📥 加载预计算数据...")
            print(f"📊 加载完成: {len(dataset)} 个训练数据对")
        
        # 创建分布式数据加载器
        batch_size = 1000 * world_size  # 总批次大小
        train_loader, train_sampler = create_distributed_loader(
            dataset, 
            batch_size=batch_size // world_size,  # 每个GPU的批次大小
            num_workers=4
        )
        
        if is_main_process():
            print(f"📦 分布式数据加载器创建完成:")
            print(f"   总批次大小: {batch_size}")
            print(f"   每GPU批次大小: {batch_size // world_size}")
        
        # 初始化模型并转换为分布式模型
        model = TransformerPredictor()
        model = to_distributed(model, local_rank)
        
        if is_main_process():
            print(f"🧠 分布式Transformer模型初始化完成")
            print(f"📈 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练参数
        num_epochs = 30
        learning_rate = 0.001
        
        # 优化器和学习率调度器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        criterion = nn.MSELoss()
        
        if is_main_process():
            print("\n⚙️  训练参数:")
            print(f"   学习率: {learning_rate}")
            print(f"   训练轮数: {num_epochs}")
            print(f"   批次大小: {batch_size}")
            print("\n" + "="*60)
            print("🎯 开始Transformer训练")
            print("="*60)
        
        # 训练循环
        train_losses = []
        best_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 设置采样器的epoch
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # 使用tqdm显示进度（仅在主进程）
            if is_main_process():
                pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                inputs, targets = batch
                inputs = inputs.to(local_rank if is_distributed else 'cuda')
                targets = targets.to(local_rank if is_distributed else 'cuda')
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 收集所有进程的损失
                reduced_loss = reduce_value(loss.detach(), average=True)
                epoch_loss += reduced_loss.item()
                batch_count += 1
                
                if is_main_process():
                    pbar.set_postfix({'loss': f"{reduced_loss.item():.6f}"})
            
            # 计算平均损失
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            
            # 更新学习率
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # 保存最佳模型（仅主进程）
            if is_main_process() and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(),
                         '/mnt/bz25t/bzhy/zhanglikang/project/models/transformer_model_best.pth')
            
            # 打印训练信息（仅主进程）
            if is_main_process() and epoch % 5 == 0:
                print(f"Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        
        # 训练完成，保存最终模型和训练历史
        if is_main_process():
            # 保存最终模型
            torch.save(model.module.state_dict() if is_distributed else model.state_dict(),
                     '/mnt/bz25t/bzhy/zhanglikang/project/models/transformer_model.pth')
            
            # 保存训练历史
            history = {
                'losses': train_losses,
                'best_loss': best_loss,
                'training_time': time.time() - training_start_time,
                'final_lr': scheduler.get_last_lr()[0],
                'batch_size': batch_size,
                'world_size': world_size
            }
            
            with open('/mnt/bz25t/bzhy/zhanglikang/project/models/transformer_training_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            
            print("\n✅ Transformer训练完成!")
            print(f"🎯 最终训练损失: {train_losses[-1]:.6f}")
            print(f"📈 损失改善: {((train_losses[0] - train_losses[-1])/train_losses[0]*100):.2f}% (从 {train_losses[0]:.6f} 到 {train_losses[-1]:.6f})")
            print(f"⏱️  训练用时: {time.time() - training_start_time:.2f}秒")
            
            # 绘制训练曲线
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.title('Transformer Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig('/mnt/bz25t/bzhy/zhanglikang/project/models/transformer_training_results.png')
            plt.close()
            
            print("\n💾 保存结果:")
            print("   ✓ 模型权重: transformer_model.pth")
            print("   ✓ 最佳模型: transformer_model_best.pth")
            print("   ✓ 训练历史: transformer_training_history.pkl")
            print("   ✓ 训练曲线: transformer_training_results.png")
    
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        raise e
    
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    main()