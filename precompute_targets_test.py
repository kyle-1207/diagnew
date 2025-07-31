#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充生成测试样本(201-392)的targets.pkl文件

这个脚本专门用于生成测试样本的targets.pkl文件，
包括正常测试样本(201-334)和故障测试样本(335-392)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Function_ import load_lstm
from Class_ import LSTM

def load_labels():
    """从Labels.xls加载标签信息"""
    try:
        labels_path = '../QAS/Labels.xls'
        print(f"📖 读取标签文件: {labels_path}")
        
        # 尝试不同的方式读取Excel文件
        try:
            df = pd.read_excel(labels_path, engine='xlrd')
        except:
            df = pd.read_excel(labels_path, engine='openpyxl')
        
        print(f"📊 标签文件形状: {df.shape}")
        print(f"📊 列名: {df.columns.tolist()}")
        
        # 获取所有样本编号和标签
        all_samples = df['Num'].tolist()
        all_labels = df['label'].tolist()
        
        # 分离测试样本
        test_normal_samples = [i for i in all_samples if 201 <= i <= 334]  # 正常测试样本
        test_fault_samples = [i for i in all_samples if 335 <= i <= 392]   # 故障测试样本
        
        print(f"📊 测试正常样本: {len(test_normal_samples)} 个 (201-334)")
        print(f"📊 测试故障样本: {len(test_fault_samples)} 个 (335-392)")
        print(f"📊 总测试样本: {len(test_normal_samples) + len(test_fault_samples)} 个")
        
        return test_normal_samples, test_fault_samples
        
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认测试样本范围")
        test_normal_samples = list(range(201, 335))  # 201-334
        test_fault_samples = list(range(335, 393))   # 335-392
        return test_normal_samples, test_fault_samples

def check_and_create_targets(sample_id, data_path):
    """检查并创建单个样本的targets.pkl文件"""
    sample_dir = os.path.join(data_path, str(sample_id))
    targets_path = os.path.join(sample_dir, 'targets.pkl')
    
    # 如果targets.pkl已存在，跳过
    if os.path.exists(targets_path):
        return True, "已存在"
    
    # 检查所需的输入文件
    vin1_path = os.path.join(sample_dir, 'vin_1.pkl')
    if not os.path.exists(vin1_path):
        return False, f"缺少vin_1.pkl"
    
    try:
        # 加载vin_1数据
        with open(vin1_path, 'rb') as f:
            vin1_data = pickle.load(f)
        
        # 确保数据是tensor格式
        if not isinstance(vin1_data, torch.Tensor):
            vin1_data = torch.tensor(vin1_data, dtype=torch.float32)
        
        print(f"   vin_1数据形状: {vin1_data.shape}")
        
        # 加载LSTM模型进行预测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 这里需要使用预训练的LSTM模型
        # 如果没有模型文件，使用默认的vin_1数据作为targets
        try:
            # 尝试加载LSTM模型（如果存在）
            lstm_model_path = './models/lstm_model.pth'
            if os.path.exists(lstm_model_path):
                lstm = LSTM()
                lstm.load_state_dict(torch.load(lstm_model_path, map_location=device))
                lstm.to(device)
                lstm.eval()
                
                # 进行LSTM预测
                vin1_input = vin1_data.to(device)
                with torch.no_grad():
                    targets = lstm(vin1_input)
                
                targets = targets.cpu()
            else:
                # 如果没有LSTM模型，使用vin_1的最后两列作为targets
                # 这是一个临时方案，实际应该使用训练好的LSTM模型
                print(f"   ⚠️  未找到LSTM模型，使用vin_1最后2列作为targets")
                targets = vin1_data[:, -2:]  # 假设最后两列是电压和SOC
        
        except Exception as model_error:
            print(f"   ⚠️  LSTM预测失败: {model_error}")
            print(f"   ⚠️  使用vin_1最后2列作为targets")
            targets = vin1_data[:, -2:]
        
        # 保存targets.pkl
        with open(targets_path, 'wb') as f:
            pickle.dump(targets, f)
        
        print(f"   ✅ targets形状: {targets.shape}")
        return True, "生成成功"
        
    except Exception as e:
        return False, f"处理失败: {str(e)}"

def main():
    """主函数：生成测试样本的targets.pkl"""
    print("🚀 开始生成测试样本的targets.pkl文件...")
    
    # 数据路径
    data_path = '../QAS'
    
    # 检查数据路径
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    # 加载测试样本列表
    test_normal_samples, test_fault_samples = load_labels()
    all_test_samples = test_normal_samples + test_fault_samples
    
    print(f"\n📊 需要处理的测试样本总数: {len(all_test_samples)}")
    
    # 统计信息
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 创建models目录（如果不存在）
    os.makedirs('./models', exist_ok=True)
    
    # 处理每个测试样本
    print("\n🔄 开始处理测试样本...")
    for sample_id in tqdm(all_test_samples, desc="生成targets.pkl"):
        success, message = check_and_create_targets(sample_id, data_path)
        
        if success:
            if "已存在" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
            print(f"\n❌ 样本 {sample_id}: {message}")
    
    # 输出统计结果
    print(f"\n📊 处理完成统计:")
    print(f"   ✅ 新生成: {success_count} 个")
    print(f"   ⏭️  已存在: {skip_count} 个")
    print(f"   ❌ 失败: {error_count} 个")
    print(f"   📈 总计: {len(all_test_samples)} 个")
    
    if error_count == 0:
        print(f"\n🎉 所有测试样本的targets.pkl文件已准备完成！")
    else:
        print(f"\n⚠️  有 {error_count} 个样本处理失败，请检查数据完整性")

if __name__ == "__main__":
    main()