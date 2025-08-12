#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Run_Complete_Visualization.py 的配置修改
验证是否正确配置了三个真实模型
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """测试配置修改"""
    print("🧪 测试 Run_Complete_Visualization.py 配置")
    print("="*60)
    
    try:
        from Run_Complete_Visualization import CompleteVisualizationRunner
        
        # 创建运行器实例
        runner = CompleteVisualizationRunner()
        
        print(f"📁 Three_model目录: {runner.three_model_dir}")
        print()
        
        print("📋 模型路径配置:")
        for name, path in runner.model_paths.items():
            print(f"   • {name}: {path}")
        print()
        
        print("🏷️  模型名称映射:")
        for key, name in runner.model_name_mapping.items():
            print(f"   • {key} → {name}")
        print()
        
        print("📊 文件模式配置:")
        for pattern_type, patterns in runner.file_patterns.items():
            print(f"   • {pattern_type}: {patterns}")
        print()
        
        # 验证数量
        model_count = len(runner.model_paths)
        print(f"✅ 支持的模型数量: {model_count}")
        
        if model_count == 3:
            print("🎉 配置正确！只包含三个真实模型")
        else:
            print(f"❌ 配置错误！应该是3个模型，但发现{model_count}个")
            
        # 检查模型名称
        expected_models = {'bilstm', 'transformer_back', 'transformer_for_back'}
        actual_models = set(runner.model_paths.keys())
        
        print(f"\n🔍 模型名称检查:")
        print(f"   期望: {expected_models}")
        print(f"   实际: {actual_models}")
        
        if actual_models == expected_models:
            print("✅ 模型名称配置正确")
        else:
            missing = expected_models - actual_models
            extra = actual_models - expected_models
            if missing:
                print(f"❌ 缺少模型: {missing}")
            if extra:
                print(f"❌ 多余模型: {extra}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_model_scripts():
    """测试对应的训练脚本是否存在"""
    print("\n🔧 检查对应的训练脚本:")
    print("="*60)
    
    expected_scripts = {
        'bilstm': ['Train_BILSTM_Only.py', 'Train_BILSTM.py'],
        'transformer_back': ['Train_Transformer_BackwardFeedback.py'],
        'transformer_for_back': ['Train_Transformer_HybridFeedback.py']
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model, scripts in expected_scripts.items():
        print(f"\n📁 {model} 模型:")
        for script in scripts:
            script_path = os.path.join(current_dir, script)
            if os.path.exists(script_path):
                print(f"   ✅ {script} - 存在")
            else:
                print(f"   ❌ {script} - 不存在")

if __name__ == "__main__":
    print("🚀 开始配置测试...")
    print()
    
    # 测试配置
    config_ok = test_configuration()
    
    # 测试脚本存在性
    test_model_scripts()
    
    print("\n" + "="*60)
    if config_ok:
        print("🎯 测试完成！配置修改验证通过")
    else:
        print("💥 测试失败！需要检查配置")
