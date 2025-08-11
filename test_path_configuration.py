#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径配置测试脚本
验证Run_Complete_Visualization.py的路径配置是否与实际文件结构匹配
"""

import os
import sys

def test_path_configuration():
    """测试路径配置"""
    print("🔧 测试Run_Complete_Visualization.py路径配置")
    print("="*60)
    
    # 基础路径
    base_dir = '/mnt/bz25t/bzhy/datasave'
    three_model_dir = f"{base_dir}/Three_model"
    
    # 根据图片显示的实际文件结构，所有模型都在 Three_model 子目录下
    model_paths = {
        'bilstm': f"{three_model_dir}/BILSTM",  # Train_BILSTM.py 的结果（文件直接在BILSTM目录下）
        'transformer_positive': f"{three_model_dir}/transformer_positive",  # Train_Transformer_HybridFeedback.py 的结果
        'transformer_pn': f"{three_model_dir}/transformer_PN"  # Train_Transformer_PN_HybridFeedback.py 的结果
    }
    
    print(f"📁 基础路径: {base_dir}")
    print(f"   存在状态: {'✅' if os.path.exists(base_dir) else '❌'}")
    print(f"📁 Three_model目录: {three_model_dir}")
    print(f"   存在状态: {'✅' if os.path.exists(three_model_dir) else '❌'}")
    print()
    
    # 测试各模型路径
    print("📋 模型路径测试:")
    for model_name, path in model_paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {model_name:20}: {path}")
        print(f"   {'':20}  状态: {status}")
        
        if exists:
            # 检查关键文件
            key_files = []
            
            if model_name == 'bilstm':
                key_files = [
                    'bilstm_training_history.pkl',
                    'net_model_bilstm_baseline.pth',
                    'netx_model_bilstm_baseline.pth'
                ]
            elif model_name == 'transformer_positive':
                key_files = [
                    'hybrid_feedback_training_history.pkl',
                    'training_history.pkl',
                    'transformer_training_history.pkl'
                ]
            elif model_name == 'transformer_pn':
                key_files = [
                    'pn_training_history.pkl',
                    'training_history.pkl',
                    'hybrid_training_history.pkl'
                ]
            
            for key_file in key_files:
                file_path = os.path.join(path, key_file)
                file_exists = os.path.exists(file_path)
                file_status = "✅" if file_exists else "❌"
                print(f"   {'':20}    {key_file}: {file_status}")
        
        print()
    
    # 测试可视化模块导入
    print("📊 可视化模块测试:")
    try:
        sys.path.append('.')
        from Run_Complete_Visualization import CompleteVisualizationRunner
        
        runner = CompleteVisualizationRunner(base_dir)  # 传入基础路径，内部会自动配置 Three_model
        print("   ✅ Run_Complete_Visualization 导入成功")
        print(f"   📁 报告目录: {runner.report_dir}")
        print(f"   📋 模型路径配置: {len(runner.model_paths)} 个")
        
        for model_name, path in runner.model_paths.items():
            status = "✅" if os.path.exists(path) else "❌"
            print(f"      {model_name}: {status}")
            
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
    
    print()
    
    # 测试可视化子模块
    print("🎨 可视化子模块测试:")
    try:
        from Visualize_Model_Comparison import ModelComparisonVisualizer
        from Visualize_Fault_Detection import FaultDetectionVisualizer
        
        print("   ✅ Visualize_Model_Comparison 导入成功")
        print("   ✅ Visualize_Fault_Detection 导入成功")
        
        # 测试初始化（使用 Three_model 路径）
        model_vis = ModelComparisonVisualizer(three_model_dir)
        fault_vis = FaultDetectionVisualizer(three_model_dir)
        
        print("   ✅ 可视化模块初始化成功")
        
    except ImportError as e:
        print(f"   ❌ 子模块导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 子模块初始化失败: {e}")
    
    print()
    
    # 检查图片中显示的其他关键文件（在 Three_model 目录下）
    print("📄 关键文件检查:")
    key_files_check = [
        (f"{three_model_dir}/BILSTM/bilstm_training_results.png", "BiLSTM训练结果图"),
        (f"{three_model_dir}/transformer_positive/transformer_summary.xlsx", "Transformer Positive汇总"),
        (f"{three_model_dir}/transformer_PN/transformer_summary.xlsx", "Transformer PN汇总")
    ]
    
    for file_path, description in key_files_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"   {description:25}: {status} {file_path}")
    
    print()
    print("🎯 测试总结:")
    print("   1. 路径配置已更新：所有模型都在 Three_model 子目录下")
    print("   2. Three_model/BILSTM -> BiLSTM训练结果（直接在BILSTM目录下）")
    print("   3. Three_model/transformer_positive -> Transformer HybridFeedback")  
    print("   4. Three_model/transformer_PN -> Transformer PN HybridFeedback")
    print("   5. 可视化模块已更新为基于 Three_model 的路径配置")
    print()
    print("✨ 配置完成！现在可以运行 Run_Complete_Visualization.py")

if __name__ == "__main__":
    test_path_configuration()
