#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Transformer-FOR-BACK 模型训练历史加载
验证修复后的路径配置是否正确
"""

import os
import pickle
import sys

def test_transformer_for_back_loading():
    """测试 Transformer-FOR-BACK 模型数据加载"""
    print("🔍 测试 Transformer-FOR-BACK 模型训练历史加载...")
    
    result_base_dir = '/mnt/bz25t/bzhy/datasave'
    
    # 根据Train_Transformer_PN_HybridFeedback_EN.py的实际保存路径配置进行测试
    combined_paths = [
        f"{result_base_dir}/pn_training_history.pkl",  # 主要保存路径（根据PN脚本配置）
        f"{result_base_dir}/Transformer/models/PN_model/pn_training_history.pkl",  # PN模型目录
        f"./pn_training_history.pkl",  # 当前目录备选路径
        f"/tmp/pn_training_history.pkl",  # 临时目录备选路径
        f"{result_base_dir}/hybrid_feedback_training_history.pkl",  # 旧版路径（兼容性）
        f"/tmp/hybrid_feedback_training_history.pkl",  # 旧版备选路径1
        f"./hybrid_feedback_training_history.pkl",  # 旧版备选路径2
        f"{result_base_dir}/Transformer/models/PN_model/training_history.pkl",
        f"{result_base_dir}/Transformer/models/PN_model/combined_training_history.pkl",
        f"{result_base_dir}/Transformer-FOR-BACK/models/combined_training_history.pkl"
    ]
    
    print(f"📁 检查训练历史文件路径:")
    found_files = []
    
    for i, path in enumerate(combined_paths, 1):
        exists = os.path.exists(path)
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"   {i}. {path}")
        print(f"      {status}")
        
        if exists:
            found_files.append(path)
            # 尝试加载文件验证格式
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                print(f"      📊 文件格式正确，包含 {len(data)} 个键: {list(data.keys())}")
            except Exception as e:
                print(f"      ⚠️  文件格式错误: {e}")
        print()
    
    if found_files:
        print(f"✅ 找到 {len(found_files)} 个可用的训练历史文件:")
        for f in found_files:
            print(f"   - {f}")
        
        # 测试使用第一个文件
        test_file = found_files[0]
        print(f"\n🧪 测试加载第一个文件: {test_file}")
        try:
            with open(test_file, 'rb') as f:
                training_data = pickle.load(f)
            
            print("✅ 成功加载训练历史数据")
            print(f"📈 数据结构: {type(training_data)}")
            
            if isinstance(training_data, dict):
                print(f"📊 数据键: {list(training_data.keys())}")
                for key, value in training_data.items():
                    if isinstance(value, list):
                        print(f"   - {key}: 列表，长度 {len(value)}")
                    else:
                        print(f"   - {key}: {type(value)}")
            
            return True, test_file, training_data
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False, test_file, None
    else:
        print("❌ 未找到任何训练历史文件")
        print("💡 建议:")
        print("   1. 检查训练脚本是否已运行完成")
        print("   2. 确认训练历史是否正确保存")
        print("   3. 检查文件路径权限")
        return False, None, None

if __name__ == "__main__":
    success, file_path, data = test_transformer_for_back_loading()
    
    if success:
        print(f"\n🎉 测试成功！Transformer-FOR-BACK 模型数据可以正确加载")
        print(f"📁 使用文件: {file_path}")
    else:
        print(f"\n❌ 测试失败！需要检查文件路径或数据格式")
        sys.exit(1)
