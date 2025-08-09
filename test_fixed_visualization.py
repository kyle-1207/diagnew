#!/usr/bin/env python3
"""
测试修复后的可视化脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Visualize_Model_Comparison import ModelComparisonVisualizer

def test_visualization_fixes():
    """测试修复后的可视化功能"""
    print("🔧 测试修复后的可视化脚本...")
    
    try:
        # 创建可视化器实例
        visualizer = ModelComparisonVisualizer()
        
        print("✅ 可视化器创建成功")
        
        # 检查模型数据
        print(f"📊 加载的模型数据: {list(visualizer.model_data.keys())}")
        
        # 尝试创建综合比较
        print("🎨 创建综合比较...")
        visualizer.create_comprehensive_comparison()
        
        print("✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization_fixes()
