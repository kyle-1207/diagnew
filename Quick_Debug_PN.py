#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PN模型快速调试脚本
检查模型文件和基本功能
"""

import os
import torch
import pickle
import numpy as np

def check_model_files():
    """检查模型文件存在性和大小"""
    print("="*60)
    print("🔍 PN模型文件检查")
    print("="*60)
    
    base_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model"
    
    files_to_check = [
        "transformer_model_pn.pth",
        "net_model_pn.pth", 
        "netx_model_pn.pth",
        "pca_params_pn.pkl"
    ]
    
    for file_name in files_to_check:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_name}: {size:.2f} MB")
            
            # 尝试加载文件
            try:
                if file_name.endswith('.pth'):
                    state_dict = torch.load(file_path, map_location='cpu')
                    print(f"   📊 模型状态字典键数量: {len(state_dict)}")
                elif file_name.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        print(f"   📊 PCA参数键: {list(data.keys())}")
                    else:
                        print(f"   📊 PCA数据类型: {type(data)}")
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
        else:
            print(f"❌ {file_name}: 文件不存在")
    
    print()

def test_transformer_loading():
    """测试Transformer模型加载"""
    print("🔧 测试Transformer模型加载...")
    
    try:
        # 添加路径
        import sys
        sys.path.append('./源代码备份')
        sys.path.append('.')
        
        from Train_Transformer_PN_HybridFeedback_EN import TransformerPredictor
        
        # 创建模型
        model = TransformerPredictor(input_size=7, d_model=128, nhead=8, num_layers=3, output_size=2)
        print(f"   ✅ Transformer模型创建成功")
        
        # 尝试加载权重
        model_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/transformer_model_pn.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"   ✅ Transformer权重加载成功")
            
            # 测试推理
            test_input = torch.randn(1, 7)
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            print(f"   ✅ Transformer推理测试成功，输出形状: {output.shape}")
        else:
            print(f"   ❌ 模型文件不存在")
            
    except Exception as e:
        print(f"   ❌ Transformer测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_mc_ae_loading():
    """测试MC-AE模型加载"""
    print("🔧 测试MC-AE模型加载...")
    
    try:
        from Function_ import *
        from Class_ import *
        
        # 测试 net_model
        net_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/net_model_pn.pth"
        netx_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/netx_model_pn.pth"
        
        if os.path.exists(net_path) and os.path.getsize(net_path) > 0:
            net_model = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                                  activation_fn=custom_activation, use_dx_in_forward=True)
            state_dict = torch.load(net_path, map_location='cpu')
            net_model.load_state_dict(state_dict)
            print(f"   ✅ net_model 加载成功")
        else:
            print(f"   ⚠️ net_model 文件无效或不存在")
            
        if os.path.exists(netx_path) and os.path.getsize(netx_path) > 0:
            netx_model = CombinedAE(input_size=2, encode2_input_size=4, output_size=110,
                                   activation_fn=torch.sigmoid, use_dx_in_forward=True)
            state_dict = torch.load(netx_path, map_location='cpu')
            netx_model.load_state_dict(state_dict)
            print(f"   ✅ netx_model 加载成功")
        else:
            print(f"   ⚠️ netx_model 文件无效或不存在")
            
    except Exception as e:
        print(f"   ❌ MC-AE测试失败: {e}")

def test_sample_loading():
    """测试样本数据加载"""
    print("🔧 测试样本数据加载...")
    
    try:
        # 测试加载第一个样本
        sample_id = "10"  # 使用一个应该存在的正常样本
        base_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}'
        
        if os.path.exists(base_path):
            print(f"   ✅ 样本目录存在: {base_path}")
            
            # 检查必要文件
            required_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl']
            for file_name in required_files:
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"   ✅ {file_name}: {type(data)}, 形状: {np.array(data).shape}")
                else:
                    print(f"   ❌ {file_name}: 不存在")
        else:
            print(f"   ❌ 样本目录不存在: {base_path}")
            
    except Exception as e:
        print(f"   ❌ 样本加载测试失败: {e}")

if __name__ == "__main__":
    check_model_files()
    test_transformer_loading()
    test_mc_ae_loading()
    test_sample_loading()
    
    print("\n" + "="*60)
    print("🎯 调试检查完成")
    print("="*60)
