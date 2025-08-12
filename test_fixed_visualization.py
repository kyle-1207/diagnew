#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的三模型可视化脚本
检验是否能正确读取服务器上的模型文件
"""

import os
import glob
import json
import pickle

def test_model_directory_access():
    """测试模型目录访问"""
    
    print("="*60)
    print("🔍 测试模型目录访问和文件读取")
    print("="*60)
    
    # 模型配置
    model_configs = {
        'BILSTM': {
            'name': 'BILSTM',
            'display_name': 'BiLSTM',
            'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/BILSTM/',
            'expected_files': [
                'bilstm_performance_metrics.json',
                'bilstm_detailed_results.pkl',
                'bilstm_test_metadata.json'
            ]
        },
        'TRANSFORMER_POSITIVE': {
            'name': 'TRANSFORMER_POSITIVE', 
            'display_name': 'Transformer (+)',
            'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/transformer_positive/',
            'expected_files': [
                'transformer_performance_metrics.json',
                'transformer_detailed_results.pkl',
                'transformer_test_metadata.json'
            ]
        },
        'TRANSFORMER_PN': {
            'name': 'TRANSFORMER_PN',
            'display_name': 'Transformer (±)',
            'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/transformer_PN/',
            'expected_files': [
                'transformer_performance_metrics.json',
                'transformer_detailed_results.pkl',
                'transformer_test_metadata.json'
            ]
        }
    }
    
    all_models_ready = True
    
    for model_key, config in model_configs.items():
        print(f"\n📁 检查模型: {config['display_name']}")
        print(f"   路径: {config['base_path']}")
        
        # 检查目录是否存在
        if not os.path.exists(config['base_path']):
            print(f"   ❌ 目录不存在")
            all_models_ready = False
            continue
            
        print(f"   ✅ 目录存在")
        
        # 列出目录中的所有文件
        try:
            files_in_dir = os.listdir(config['base_path'])
            print(f"   📋 目录中的文件数量: {len(files_in_dir)}")
            
            # 查找关键文件
            found_files = []
            missing_files = []
            
            for expected_file in config['expected_files']:
                file_path = os.path.join(config['base_path'], expected_file)
                if os.path.exists(file_path):
                    found_files.append(expected_file)
                    print(f"      ✅ {expected_file}")
                else:
                    missing_files.append(expected_file)
                    print(f"      ❌ {expected_file} (缺失)")
            
            # 查找可能的替代文件模式
            print(f"   🔍 查找可能的替代文件:")
            json_files = [f for f in files_in_dir if f.endswith('.json')]
            pkl_files = [f for f in files_in_dir if f.endswith('.pkl')]
            
            print(f"      JSON文件: {json_files}")
            print(f"      PKL文件: {pkl_files}")
            
            if missing_files:
                print(f"   ⚠️  缺失关键文件: {missing_files}")
                all_models_ready = False
            else:
                print(f"   ✅ 所有关键文件都存在")
                
        except Exception as e:
            print(f"   ❌ 读取目录时出错: {e}")
            all_models_ready = False
    
    print(f"\n{'='*60}")
    if all_models_ready:
        print("🎉 所有模型文件都准备就绪！")
        print("📊 可以开始运行三模型比较可视化")
    else:
        print("⚠️  部分模型文件缺失")
        print("💡 建议先运行对应的测试脚本生成缺失的文件")
    
    print(f"{'='*60}")
    
    return all_models_ready

def suggest_missing_file_generation():
    """建议如何生成缺失的文件"""
    
    print("\n📝 生成缺失文件的建议:")
    print("-" * 40)
    
    suggestions = [
        {
            'model': 'BiLSTM',
            'script': 'Test_combine_BILSTMonly.py',
            'description': '运行BiLSTM测试并生成性能指标文件'
        },
        {
            'model': 'Transformer Positive',
            'script': 'Test_combine_HybridFeedback_transonly.py',
            'description': '运行Transformer Positive测试并生成结果文件'
        },
        {
            'model': 'Transformer PN',
            'script': 'Test_combine_transonly.py',
            'description': '运行Transformer PN测试并生成结果文件'
        }
    ]
    
    for suggestion in suggestions:
        print(f"🔧 {suggestion['model']}:")
        print(f"   脚本: {suggestion['script']}")
        print(f"   作用: {suggestion['description']}")
        print()

if __name__ == "__main__":
    # 测试目录访问
    models_ready = test_model_directory_access()
    
    if not models_ready:
        suggest_missing_file_generation()
    else:
        print("\n🚀 准备运行修复后的可视化脚本...")
        print("💡 可以执行: python Three_Model_Comparison_Visualization.py")