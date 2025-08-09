#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试targets数据格式的脚本
"""

import os
import pickle
import numpy as np
import pandas as pd

def test_targets_format():
    """测试targets.pkl文件的数据格式"""
    print("🔍 测试targets数据格式")
    print("="*60)
    
    # 加载样本标签
    try:
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        labels_df = pd.read_excel(labels_path)
        
        # 提取前10个正常样本进行测试
        normal_samples = labels_df[labels_df['Label'] == 0]['Num'].astype(str).tolist()[:10]
        print(f"📋 测试样本: {normal_samples}")
        
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        normal_samples = ['0', '1', '2', '3', '4']
    
    # 测试每个样本的targets格式
    for sample_id in normal_samples:
        print(f"\n🔍 测试样本 {sample_id}:")
        
        try:
            base_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS'
            targets_path = f"{base_path}/{sample_id}/targets.pkl"
            
            if not os.path.exists(targets_path):
                print(f"   ❌ 文件不存在: {targets_path}")
                continue
            
            # 加载targets数据
            with open(targets_path, 'rb') as f:
                targets = pickle.load(f)
            
            print(f"   📋 数据类型: {type(targets)}")
            
            if isinstance(targets, dict):
                print(f"   📋 字典键: {list(targets.keys())}")
                
                # 检查是否包含标准键
                if 'terminal_voltages' in targets and 'pack_socs' in targets:
                    print(f"   ✅ 标准targets格式")
                    
                    terminal_voltages = targets['terminal_voltages']
                    pack_socs = targets['pack_socs']
                    
                    print(f"   📊 terminal_voltages:")
                    print(f"      类型: {type(terminal_voltages)}")
                    if hasattr(terminal_voltages, 'shape'):
                        print(f"      形状: {terminal_voltages.shape}")
                    elif isinstance(terminal_voltages, (list, tuple)):
                        print(f"      长度: {len(terminal_voltages)}")
                    
                    print(f"   📊 pack_socs:")
                    print(f"      类型: {type(pack_socs)}")
                    if hasattr(pack_socs, 'shape'):
                        print(f"      形状: {pack_socs.shape}")
                    elif isinstance(pack_socs, (list, tuple)):
                        print(f"      长度: {len(pack_socs)}")
                    
                    # 显示前几个数值
                    try:
                        terminal_voltages_array = np.array(terminal_voltages)
                        pack_socs_array = np.array(pack_socs)
                        
                        print(f"   📊 数值范围:")
                        print(f"      terminal_voltages: [{terminal_voltages_array.min():.4f}, {terminal_voltages_array.max():.4f}]")
                        print(f"      pack_socs: [{pack_socs_array.min():.4f}, {pack_socs_array.max():.4f}]")
                        
                        print(f"   📊 前5个数值:")
                        print(f"      terminal_voltages: {terminal_voltages_array[:5]}")
                        print(f"      pack_socs: {pack_socs_array[:5]}")
                        
                    except Exception as e:
                        print(f"   ⚠️ 数值分析失败: {e}")
                
                else:
                    print(f"   ⚠️ 非标准targets格式")
                    # 显示每个键的详细信息
                    for key, value in targets.items():
                        print(f"      {key}: {type(value)}")
                        if hasattr(value, 'shape'):
                            print(f"         形状: {value.shape}")
                        elif isinstance(value, (list, tuple)):
                            print(f"         长度: {len(value)}")
            
            elif isinstance(targets, (list, tuple)):
                print(f"   📋 列表/元组格式，长度: {len(targets)}")
                try:
                    targets_array = np.array(targets)
                    print(f"   📊 转换为数组: {targets_array.shape}")
                    print(f"   📊 数值范围: [{targets_array.min():.4f}, {targets_array.max():.4f}]")
                except Exception as e:
                    print(f"   ❌ 数组转换失败: {e}")
            
            elif hasattr(targets, 'shape'):
                print(f"   📋 数组格式: {targets.shape}")
                print(f"   📊 数值范围: [{targets.min():.4f}, {targets.max():.4f}]")
            
            else:
                print(f"   ❌ 未知格式: {type(targets)}")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
        
        print("-" * 40)
    
    print("\n✅ targets格式测试完成")

if __name__ == "__main__":
    test_targets_format()