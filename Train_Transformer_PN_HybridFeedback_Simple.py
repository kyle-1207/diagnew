#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正负反馈混合训练脚本 - 简化调试版本
专门用于调试数据加载和基本训练流程
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
from datetime import datetime
import pandas as pd

# 忽略警告
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加源代码路径
sys.path.append('./源代码备份')
sys.path.append('.')

def load_sample_labels():
    """从Labels.xls加载样本标签信息"""
    try:
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        labels_df = pd.read_excel(labels_path)
        
        # 提取正常样本和故障样本
        normal_samples = labels_df[labels_df['Label'] == 0]['Num'].astype(str).tolist()
        fault_samples = labels_df[labels_df['Label'] == 1]['Num'].astype(str).tolist()
        
        print(f"📊 从Labels.xls加载样本标签:")
        print(f"   正常样本: {len(normal_samples)} 个")
        print(f"   故障样本: {len(fault_samples)} 个")
        print(f"   总样本数: {len(labels_df)} 个")
        
        return normal_samples[:10], fault_samples[:5], labels_df  # 限制数量用于调试
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("🔄 使用默认样本配置")
        # 返回默认配置
        normal_samples = [str(i) for i in range(0, 10)]
        fault_samples = [str(i) for i in range(340, 345)]
        return normal_samples, fault_samples, None

def load_sample_data(sample_id):
    """加载单个样本数据"""
    try:
        base_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS'
        sample_path = f"{base_path}/{sample_id}"
        
        # 检查文件是否存在
        required_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl', 'targets.pkl']
        for file_name in required_files:
            file_path = f"{sample_path}/{file_name}"
            if not os.path.exists(file_path):
                print(f"   ❌ 文件不存在: {file_path}")
                return None
        
        # 加载数据文件
        print(f"   📂 加载样本 {sample_id}...")
        vin_1 = pickle.load(open(f"{sample_path}/vin_1.pkl", 'rb'))
        vin_2 = pickle.load(open(f"{sample_path}/vin_2.pkl", 'rb'))
        vin_3 = pickle.load(open(f"{sample_path}/vin_3.pkl", 'rb'))
        targets = pickle.load(open(f"{sample_path}/targets.pkl", 'rb'))
        
        # 转换为numpy格式
        if hasattr(vin_1, 'detach'):
            vin_1 = vin_1.detach().cpu().numpy()
        if hasattr(vin_2, 'detach'):
            vin_2 = vin_2.detach().cpu().numpy()
        if hasattr(vin_3, 'detach'):
            vin_3 = vin_3.detach().cpu().numpy()
        if hasattr(targets, 'detach'):
            targets = targets.detach().cpu().numpy()
        
        print(f"   ✅ 样本 {sample_id} 加载成功:")
        print(f"      vin_1: {vin_1.shape}")
        print(f"      vin_2: {vin_2.shape}")
        print(f"      vin_3: {vin_3.shape}")
        print(f"      targets: {targets.shape}")
        
        return {
            'vin_1': vin_1,
            'vin_2': vin_2, 
            'vin_3': vin_3,
            'targets': targets,
            'sample_id': sample_id
        }
    except Exception as e:
        print(f"   ❌ 加载样本 {sample_id} 失败: {e}")
        return None

def test_data_loading():
    """测试数据加载功能"""
    print("="*80)
    print("🔍 数据加载测试")
    print("="*80)
    
    # 加载样本标签
    normal_samples, fault_samples, labels_df = load_sample_labels()
    
    print(f"\n📋 测试样本:")
    print(f"   正常样本: {normal_samples}")
    print(f"   故障样本: {fault_samples}")
    
    # 测试加载几个样本
    test_samples = normal_samples[:3] + fault_samples[:2]
    successful_loads = []
    
    for sample_id in test_samples:
        print(f"\n🔍 测试样本 {sample_id}:")
        data = load_sample_data(sample_id)
        if data is not None:
            successful_loads.append(data)
        print("-" * 40)
    
    print(f"\n📊 加载结果:")
    print(f"   测试样本数: {len(test_samples)}")
    print(f"   成功加载: {len(successful_loads)}")
    
    if successful_loads:
        print(f"\n✅ 数据加载测试成功！")
        
        # 分析第一个成功加载的样本
        sample = successful_loads[0]
        print(f"\n🔍 数据结构分析 (样本 {sample['sample_id']}):")
        
        vin_1 = sample['vin_1']
        vin_2 = sample['vin_2']
        vin_3 = sample['vin_3']
        targets = sample['targets']
        
        print(f"   vin_1 统计: min={np.min(vin_1):.4f}, max={np.max(vin_1):.4f}, mean={np.mean(vin_1):.4f}")
        print(f"   vin_2 统计: min={np.min(vin_2):.4f}, max={np.max(vin_2):.4f}, mean={np.mean(vin_2):.4f}")
        print(f"   vin_3 统计: min={np.min(vin_3):.4f}, max={np.max(vin_3):.4f}, mean={np.mean(vin_3):.4f}")
        print(f"   targets 统计: min={np.min(targets):.4f}, max={np.max(targets):.4f}, mean={np.mean(targets):.4f}")
        
        # 测试数据切片
        print(f"\n🔧 测试数据切片:")
        try:
            # 基于原始代码的维度设置
            dim_x, dim_y, dim_z, dim_q = 2, 110, 110, 3
            dim_x2, dim_y2, dim_z2, dim_q2 = 2, 110, 110, 4
            
            print(f"   期望 vin_2 维度: {dim_x + dim_y + dim_z + dim_q} = {2 + 110 + 110 + 3}")
            print(f"   实际 vin_2 维度: {vin_2.shape[1]}")
            
            print(f"   期望 vin_3 维度: {dim_x2 + dim_y2 + dim_z2 + dim_q2} = {2 + 110 + 110 + 4}")
            print(f"   实际 vin_3 维度: {vin_3.shape[1]}")
            
            if vin_2.shape[1] >= dim_x + dim_y:
                x_recovered = vin_2[:, :dim_x]
                y_recovered = vin_2[:, dim_x:dim_x + dim_y]
                print(f"   ✅ vin_2 切片成功: x_recovered {x_recovered.shape}, y_recovered {y_recovered.shape}")
            
            if vin_3.shape[1] >= dim_x2 + dim_y2:
                x_recovered2 = vin_3[:, :dim_x2]
                y_recovered2 = vin_3[:, dim_x2:dim_x2 + dim_y2]
                print(f"   ✅ vin_3 切片成功: x_recovered2 {x_recovered2.shape}, y_recovered2 {y_recovered2.shape}")
            
        except Exception as e:
            print(f"   ❌ 数据切片失败: {e}")
        
        return True
    else:
        print(f"\n❌ 数据加载测试失败！")
        return False

def main():
    """主测试函数"""
    print("🚀 正负反馈混合训练 - 简化调试版本")
    print("="*80)
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备配置: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    # 测试数据加载
    success = test_data_loading()
    
    if success:
        print(f"\n🎉 基础功能测试通过！")
        print(f"   可以继续进行完整训练")
    else:
        print(f"\n❌ 基础功能测试失败！")
        print(f"   请检查数据路径和文件")
    
    print(f"\n📝 测试报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()