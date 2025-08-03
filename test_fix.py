#!/usr/bin/env python3
"""
测试Train_Transformer_Integrated.py修复是否有效
"""

import sys
import os

def test_imports():
    """测试关键导入"""
    try:
        print("📦 测试基础导入...")
        import pandas as pd
        import pickle
        import numpy as np
        print("✅ 基础库导入成功")
        
        print("📦 测试自定义模块导入...")
        from Function_ import *
        from Class_ import *
        print("✅ 自定义模块导入成功")
        
        print("📦 测试数据加载器导入...")
        from data_loader_transformer import TransformerBatteryDataset
        print("✅ 数据加载器导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_paths():
    """测试关键路径"""
    test_paths = [
        '/mnt/bz25t/bzhy/zhanglikang/project/QAS',
        '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
    ]
    
    print("📂 测试关键路径...")
    for path in test_paths:
        if os.path.exists(path):
            print(f"✅ 路径存在: {path}")
        else:
            print(f"⚠️  路径不存在: {path}")

def test_load_functions():
    """测试数据加载函数"""
    try:
        print("🔧 测试加载函数...")
        
        # 导入主文件中的函数
        sys.path.append('/mnt/bz25t/bzhy/zhanglikang/project/Linux')
        from Train_Transformer_Integrated import load_train_samples, setup_chinese_fonts
        
        print("📋 测试字体设置...")
        setup_chinese_fonts()
        
        print("📋 测试样本加载...")
        samples = load_train_samples()
        print(f"✅ 加载到 {len(samples)} 个训练样本")
        
        return True
    except Exception as e:
        print(f"❌ 函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🧪 Train_Transformer_Integrated.py 修复验证")
    print("="*60)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试路径
    test_paths()
    
    # 测试函数
    function_success = test_load_functions()
    
    print("\n" + "="*60)
    if import_success and function_success:
        print("🎉 所有测试通过，修复成功！")
        print("✅ 可以尝试运行Train_Transformer_Integrated.py")
    else:
        print("⚠️  部分测试失败，可能需要进一步检查")
    print("="*60)

if __name__ == "__main__":
    main()