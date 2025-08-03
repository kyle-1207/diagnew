#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本1数据维度分析脚本
分析vin_2和vin_3数据中异常值所在的维度
"""

import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_sample_dimensions(sample_id=1):
    """分析指定样本的数据维度"""
    print(f"🔍 分析样本 {sample_id} 的数据维度...")
    
    # 加载数据 - Windows路径格式
    vin2_path = f'F:\\大模型实战\\Batteries over Stochastic\\project\\data\\QAS\\{sample_id}\\vin_2.pkl'
    vin3_path = f'F:\\大模型实战\\Batteries over Stochastic\\project\\data\\QAS\\{sample_id}\\vin_3.pkl'
    
    # 检查文件是否存在
    if not os.path.exists(vin2_path):
        print(f"❌ 文件不存在: {vin2_path}")
        return None
    if not os.path.exists(vin3_path):
        print(f"❌ 文件不存在: {vin3_path}")
        return None
    
    print(f"📁 加载文件:")
    print(f"   vin_2: {vin2_path}")
    print(f"   vin_3: {vin3_path}")
    
    with open(vin2_path, 'rb') as f:
        vin2_data = pickle.load(f)
    with open(vin3_path, 'rb') as f:
        vin3_data = pickle.load(f)
    
    # 转换为numpy数组
    if isinstance(vin2_data, torch.Tensor):
        vin2_np = vin2_data.cpu().numpy()
    else:
        vin2_np = np.array(vin2_data)
        
    if isinstance(vin3_data, torch.Tensor):
        vin3_np = vin3_data.cpu().numpy()
    else:
        vin3_np = np.array(vin3_data)
    
    print(f"📊 数据基本信息:")
    print(f"   vin_2形状: {vin2_np.shape}")
    print(f"   vin_3形状: {vin3_np.shape}")
    
    # 分析vin_2的每个维度
    print(f"\n🔍 vin_2数据维度分析:")
    print(f"{'维度':<6} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'最小2%':<15} {'最大2%':<15} {'异常值':<10}")
    print("-" * 100)
    
    abnormal_dimensions_vin2 = []
    for i in range(vin2_np.shape[1]):
        col_data = vin2_np[:, i]
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        
        # 计算2%分位数
        percentile_2 = np.percentile(col_data, 2)
        percentile_98 = np.percentile(col_data, 98)
        
        # 判断是否为异常维度
        is_abnormal = False
        if abs(min_val) > 1e6 or abs(max_val) > 1e6:
            is_abnormal = True
            abnormal_dimensions_vin2.append(i)
        
        print(f"{i:<6} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f} {percentile_2:<15.6f} {percentile_98:<15.6f} {'异常' if is_abnormal else '正常'}")
    
    # 分析vin_3的每个维度
    print(f"\n🔍 vin_3数据维度分析:")
    print(f"{'维度':<6} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'最小2%':<15} {'最大2%':<15} {'异常值':<10}")
    print("-" * 100)
    
    abnormal_dimensions_vin3 = []
    for i in range(vin3_np.shape[1]):
        col_data = vin3_np[:, i]
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        
        # 计算2%分位数
        percentile_2 = np.percentile(col_data, 2)
        percentile_98 = np.percentile(col_data, 98)
        
        # 判断是否为异常维度
        is_abnormal = False
        if abs(min_val) > 1e6 or abs(max_val) > 1e6:
            is_abnormal = True
            abnormal_dimensions_vin3.append(i)
        
        print(f"{i:<6} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f} {percentile_2:<15.6f} {percentile_98:<15.6f} {'异常' if is_abnormal else '正常'}")
    
    # 详细分析异常维度
    print(f"\n🚨 异常维度详细分析:")
    
    if abnormal_dimensions_vin2:
        print(f"   vin_2异常维度: {abnormal_dimensions_vin2}")
        for dim in abnormal_dimensions_vin2:
            col_data = vin2_np[:, dim]
            print(f"     维度{dim}: 范围[{np.min(col_data):.2e}, {np.max(col_data):.2e}]")
            print(f"            NaN数量: {np.isnan(col_data).sum()}")
            print(f"            Inf数量: {np.isinf(col_data).sum()}")
    else:
        print("   vin_2无异常维度")
    
    if abnormal_dimensions_vin3:
        print(f"   vin_3异常维度: {abnormal_dimensions_vin3}")
        for dim in abnormal_dimensions_vin3:
            col_data = vin3_np[:, dim]
            print(f"     维度{dim}: 范围[{np.min(col_data):.2e}, {np.max(col_data):.2e}]")
            print(f"            NaN数量: {np.isnan(col_data).sum()}")
            print(f"            Inf数量: {np.isinf(col_data).sum()}")
    else:
        print("   vin_3无异常维度")
    
    # 保存分析结果 - Windows路径格式
    output_path = f'sample_{sample_id}_analysis.pkl'
    analysis_result = {
        'sample_id': sample_id,
        'vin2_shape': vin2_np.shape,
        'vin3_shape': vin3_np.shape,
        'abnormal_dimensions_vin2': abnormal_dimensions_vin2,
        'abnormal_dimensions_vin3': abnormal_dimensions_vin3,
        'vin2_data': vin2_np,
        'vin3_data': vin3_np
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(analysis_result, f)
    
    print(f"\n✅ 分析结果已保存到 {output_path}")
    
    return analysis_result

if __name__ == "__main__":
    analyze_sample_dimensions(1)