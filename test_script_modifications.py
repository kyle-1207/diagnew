#!/usr/bin/env python3
"""
测试Test_combine_transonly.py修改是否正确
"""

import sys
import os

def test_paths():
    """测试路径配置"""
    print("🔧 测试路径配置...")
    
    # 测试Labels.xls路径
    labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
    print(f"   Labels.xls路径: {labels_path}")
    print(f"   存在: {os.path.exists(labels_path)}")
    
    # 测试模型文件路径
    model_paths = [
        "/mnt/bz25t/bzhy/datasave/transformer_model_hybrid_feedback.pth",
        "/mnt/bz25t/bzhy/datasave/net_model_hybrid_feedback.pth",
        "/mnt/bz25t/bzhy/datasave/netx_model_hybrid_feedback.pth",
        "/mnt/bz25t/bzhy/datasave/pca_params_hybrid_feedback.pkl"
    ]
    
    print("\n📁 模型文件路径:")
    for path in model_paths:
        exists = os.path.exists(path)
        print(f"   {path}: {'✅' if exists else '❌'}")
        if exists:
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"     大小: {size:.1f}MB")
    
    # 测试测试样本路径
    test_samples = ['10', '11', '335', '336']
    print(f"\n📊 测试样本路径:")
    for sample_id in test_samples:
        sample_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}'
        exists = os.path.exists(sample_path)
        print(f"   样本{sample_id}: {'✅' if exists else '❌'}")
        if exists:
            # 检查数据文件
            data_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl']
            for file in data_files:
                file_path = f'{sample_path}/{file}'
                file_exists = os.path.exists(file_path)
                print(f"     {file}: {'✅' if file_exists else '❌'}")

def test_imports():
    """测试导入"""
    print("\n📦 测试导入...")
    
    try:
        import pandas as pd
        print("   ✅ pandas")
    except ImportError as e:
        print(f"   ❌ pandas: {e}")
    
    try:
        import pickle
        print("   ✅ pickle")
    except ImportError as e:
        print(f"   ❌ pickle: {e}")
    
    try:
        import torch
        print("   ✅ torch")
    except ImportError as e:
        print(f"   ❌ torch: {e}")
    
    try:
        from Train_Transformer_HybridFeedback import TransformerPredictor
        print("   ✅ TransformerPredictor")
    except ImportError as e:
        print(f"   ❌ TransformerPredictor: {e}")

def main():
    """主函数"""
    print("🧪 测试Test_combine_transonly.py修改验证")
    print("=" * 50)
    
    test_paths()
    test_imports()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成!")

if __name__ == "__main__":
    main() 