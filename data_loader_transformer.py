"""
Transformer训练数据加载器
加载vin_1和targets.pkl构建训练数据对
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TransformerBatteryDataset(Dataset):
    """Transformer电池数据集"""
    
    def __init__(self, data_path='./data/QAS', sample_ids=None):
        self.data_path = data_path
        self.training_pairs = []
        
        if sample_ids is None:
            # 如果没有指定样本，使用0-20作为默认训练集
            sample_ids = list(range(21))
        
        # 加载所有训练数据对
        for sample_id in sample_ids:
            pairs = self.load_sample_pairs(sample_id)
            if pairs:
                self.training_pairs.extend(pairs)
        
        print(f"📊 加载完成: {len(self.training_pairs)} 个训练数据对")
    
    def load_sample_pairs(self, sample_id):
        """加载单个样本的训练数据对"""
        try:
            # 加载vin_1和targets
            with open(f'{self.data_path}/{sample_id}/vin_1.pkl', 'rb') as f:
                vin_1 = pickle.load(f)
            with open(f'{self.data_path}/{sample_id}/targets.pkl', 'rb') as f:
                targets = pickle.load(f)
            
            # 转换为numpy数组
            if isinstance(vin_1, torch.Tensor):
                vin_1 = vin_1.cpu().numpy()
            
            pairs = []
            
            # 构建训练数据对 (k时刻输入 -> k+1时刻目标)
            for k in range(len(vin_1) - 1):
                # 输入：k时刻的7维状态
                input_7d = [
                    vin_1[k, 0, 0],  # 原始维度0-4
                    vin_1[k, 0, 1],
                    vin_1[k, 0, 2],
                    vin_1[k, 0, 3],
                    vin_1[k, 0, 4],
                    targets['terminal_voltages'][k],  # k时刻电压真实值
                    targets['pack_socs'][k]           # k时刻SOC真实值
                ]
                
                # 目标：k+1时刻的2维预测
                target_2d = [
                    targets['terminal_voltages'][k+1],  # k+1时刻电压真实值
                    targets['pack_socs'][k+1]           # k+1时刻SOC真实值
                ]
                
                pairs.append((input_7d, target_2d))
            
            return pairs
            
        except Exception as e:
            print(f"❌ 加载样本 {sample_id} 失败: {e}")
            return []
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        input_data, target_data = self.training_pairs[idx]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)

def create_transformer_dataloader(data_path='./data/QAS', sample_ids=None, batch_size=32, shuffle=True):
    """创建Transformer训练数据加载器"""
    dataset = TransformerBatteryDataset(data_path, sample_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 便捷函数
def load_transformer_data(sample_id, data_path='./data/QAS'):
    """加载单个样本的Transformer训练数据"""
    try:
        with open(f'{data_path}/{sample_id}/vin_1.pkl', 'rb') as f:
            vin_1 = pickle.load(f)
        with open(f'{data_path}/{sample_id}/targets.pkl', 'rb') as f:
            targets = pickle.load(f)
        return vin_1, targets
    except Exception as e:
        print(f"❌ 加载样本 {sample_id} 失败: {e}")
        return None, None

if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试Transformer数据加载器...")
    
    # 创建数据加载器（使用样本0-2进行测试）
    dataloader = create_transformer_dataloader(sample_ids=[0, 1, 2], batch_size=4)
    
    # 测试一个batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  输入形状: {inputs.shape}")  # [batch_size, 7]
        print(f"  目标形状: {targets.shape}")  # [batch_size, 2]
        print(f"  输入样例: {inputs[0]}")
        print(f"  目标样例: {targets[0]}")
        
        if batch_idx >= 2:  # 只显示前3个batch
            break
    
    print(f"\n✅ 数据加载器测试完成!") 