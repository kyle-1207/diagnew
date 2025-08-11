#!/usr/bin/env python3
# 测试可视化配置脚本

import os
import sys
from Run_Complete_Visualization import CompleteVisualizationRunner

def test_configuration():
    """测试可视化配置是否正确"""
    print("🧪 测试可视化配置...")
    print("="*50)
    
    # 创建运行器实例
    runner = CompleteVisualizationRunner()
    
    print("\n📊 测试结果:")
    print("-"*30)
    
    # 测试模型路径
    all_paths_exist = True
    for model_name, path in runner.model_paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {model_name}: {path}")
        if not exists:
            all_paths_exist = False
    
    print(f"\n📁 模型目录存在状态: {'✅ 全部存在' if all_paths_exist else '❌ 部分缺失'}")
    
    # 测试文件映射
    print(f"\n🔍 文件映射测试:")
    print("-"*30)
    
    total_files = 0
    existing_files = 0
    
    for model_name in runner.model_file_patterns.keys():
        print(f"\n📂 {model_name}:")
        for file_type in runner.model_file_patterns[model_name].keys():
            total_files += 1
            file_path = runner.get_model_file_path(model_name, file_type)
            if file_path and os.path.exists(file_path):
                existing_files += 1
                print(f"   ✅ {file_type}: {os.path.basename(file_path)}")
            else:
                print(f"   ❌ {file_type}: {runner.model_file_patterns[model_name][file_type]}")
    
    print(f"\n📈 文件存在率: {existing_files}/{total_files} ({existing_files/total_files*100:.1f}%)")
    
    # 测试辅助方法
    print(f"\n🔧 辅助方法测试:")
    print("-"*30)
    
    test_cases = [
        ('transformer_pn', 'transformer_model'),
        ('transformer_positive', 'net_model'),
        ('bilstm', 'model'),
        ('invalid_model', 'model'),
        ('transformer_pn', 'invalid_file')
    ]
    
    for model_name, file_type in test_cases:
        result = runner.get_model_file_path(model_name, file_type)
        status = "✅" if result else "❌"
        print(f"{status} get_model_file_path('{model_name}', '{file_type}'): {result}")
    
    print(f"\n🎯 配置测试完成!")
    return existing_files, total_files

if __name__ == "__main__":
    existing, total = test_configuration()
    
    if existing == total:
        print(f"\n🎉 所有配置文件都存在！可以运行完整分析。")
        sys.exit(0)
    else:
        print(f"\n⚠️  部分文件缺失 ({existing}/{total})，但脚本仍可运行。")
        sys.exit(1)
