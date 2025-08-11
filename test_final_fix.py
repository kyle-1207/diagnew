#!/usr/bin/env python3
"""测试最终修复结果"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_visualizer_import():
    """测试可视化器导入和基本配置"""
    print("🧪 测试可视化器导入和配置...")
    
    try:
        from Visualize_Model_Comparison import ModelComparisonVisualizer
        
        # 创建可视化器实例
        visualizer = ModelComparisonVisualizer()
        
        print("✅ ModelComparisonVisualizer 导入成功")
        print("📋 模型路径配置:", visualizer.model_paths)
        print("🎨 颜色配置:", visualizer.colors)
        print("📍 标记配置:", visualizer.markers)
        
        # 测试加载模型结果
        print("\n🔄 测试加载模型结果...")
        visualizer.load_model_results()
        
        print("📊 已加载的模型数据键名:", list(visualizer.model_data.keys()))
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_runner():
    """测试完整运行器配置"""
    print("\n🧪 测试完整运行器配置...")
    
    try:
        from Run_Complete_Visualization import CompleteVisualizationRunner
        
        # 创建运行器实例
        runner = CompleteVisualizationRunner()
        
        print("✅ CompleteVisualizationRunner 导入成功")
        print("📋 模型路径配置:", runner.model_paths)
        print("🔗 模型名称映射:", runner.model_name_mapping)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始最终修复验证测试...")
    
    success1 = test_visualizer_import()
    success2 = test_complete_runner()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！修复成功！")
    else:
        print("\n❌ 仍有问题需要解决")
