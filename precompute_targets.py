"""
预计算所有QAS样本的Terminal Voltage和Pack SOC
基于真实的110个单体数据计算均值，保存为targets.pkl便于Transformer训练使用
"""

import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
import glob

def load_sample_data(data_path, sample_id):
    """加载样本的vin_2和vin_3数据"""
    try:
        with open(f'{data_path}/{sample_id}/vin_2.pkl', 'rb') as f:
            vin2 = pickle.load(f)
        with open(f'{data_path}/{sample_id}/vin_3.pkl', 'rb') as f:
            vin3 = pickle.load(f)
        
        # 转换为numpy数组
        if isinstance(vin2, torch.Tensor):
            vin2 = vin2.cpu().numpy()
        if isinstance(vin3, torch.Tensor):
            vin3 = vin3.cpu().numpy()
            
        return vin2, vin3
    except Exception as e:
        print(f"❌ 加载样本 {sample_id} 失败: {e}")
        return None, None

def compute_targets(vin2, vin3):
    """计算Terminal Voltage和Pack SOC"""
    # 提取110个单体的真实值 (维度2-111)
    voltages = vin2[:, 2:112].mean(axis=1)  # Terminal Voltage = mean(110个单体电压)
    socs = vin3[:, 2:112].mean(axis=1)      # Pack SOC = mean(110个单体SOC)
    
    return voltages.tolist(), socs.tolist()

def process_sample(data_path, sample_id):
    """处理单个样本"""
    # 加载数据
    vin2, vin3 = load_sample_data(data_path, sample_id)
    if vin2 is None or vin3 is None:
        return False
    
    # 计算目标值
    terminal_voltages, pack_socs = compute_targets(vin2, vin3)
    
    # 构建保存数据
    targets_data = {
        'terminal_voltages': terminal_voltages,
        'pack_socs': pack_socs,
        'sample_id': sample_id,
        'length': len(terminal_voltages)
    }
    
    # 保存到targets.pkl
    save_path = f'{data_path}/{sample_id}/targets.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(targets_data, f)
    
    return True

def load_labels(data_path):
    """从Labels.xls加载样本信息"""
    try:
        import pandas as pd
        labels_path = f'{data_path}/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取样本编号和标签
        sample_ids = df['Num'].tolist()
        labels = df['Label'].tolist()
        
        print(f"📋 从Labels.xls加载样本信息:")
        print(f"   总样本数: {len(sample_ids)}")
        print(f"   正常样本: {labels.count(0)} 个")
        print(f"   故障样本: {labels.count(1)} 个")
        print(f"   样本编号范围: {min(sample_ids)} - {max(sample_ids)}")
        
        # 根据训练需求筛选样本
        train_samples = [i for i in sample_ids if 0 <= i <= 200]
        test_normal_samples = [i for i in sample_ids if 201 <= i <= 334 and labels[sample_ids.index(i)] == 0]
        test_fault_samples = [i for i in sample_ids if 335 <= i <= 392 and labels[sample_ids.index(i)] == 1]
        
        print(f"📊 样本分配:")
        print(f"   训练样本 (0-200): {len(train_samples)} 个")
        print(f"   测试正常样本 (201-334): {len(test_normal_samples)} 个")
        print(f"   测试故障样本 (335-392): {len(test_fault_samples)} 个")
        
        return sample_ids, labels, train_samples, test_normal_samples, test_fault_samples
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        return [], [], [], [], []

def main():
    """主函数"""
    data_path = '../QAS'
    
    # Linux环境检查
    import platform
    print(f"🖥️  运行环境: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {platform.python_version()}")
    
    # 检查数据目录是否存在
    import os
    if not os.path.exists(data_path):
        print(f"❌ 数据目录不存在: {data_path}")
        print("请确保QAS.zip已解压到指定路径")
        return
    
    # 从Labels.xls加载样本信息
    sample_ids, labels, train_samples, test_normal_samples, test_fault_samples = load_labels(data_path)
    
    if not sample_ids:
        print("❌ 未能加载样本信息!")
        return
    
    print(f"\n🚀 开始处理训练样本...")
    print(f"   训练样本: {len(train_samples)} 个")
    print(f"   测试正常样本: {len(test_normal_samples)} 个")
    print(f"   测试故障样本: {len(test_fault_samples)} 个")
    
    # 处理训练样本
    successful = 0
    failed = 0
    
    for sample_id in tqdm(train_samples, desc="处理训练样本"):
        if process_sample(data_path, sample_id):
            successful += 1
        else:
            failed += 1
    
    print(f"\n🎉 处理完成!")
    print(f"   成功: {successful} 个样本")
    print(f"   失败: {failed} 个样本")
    
    # 验证前3个训练样本
    print(f"\n🔍 验证前3个训练样本:")
    for sample_id in train_samples[:3]:
        try:
            with open(f'{data_path}/{sample_id}/targets.pkl', 'rb') as f:
                data = pickle.load(f)
            print(f"   样本 {sample_id}: {data['length']} 个时刻")
        except:
            print(f"   样本 {sample_id}: 验证失败")

if __name__ == "__main__":
    main() 