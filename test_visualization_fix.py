#!/usr/bin/env python3
"""
测试可视化修复效果的脚本
"""

import sys
import os
sys.path.append('/mnt/bz25t/bzhy/datasave')

from Visualize_Model_Comparison import ModelComparisonVisualizer

def test_visualization_fixes():
    """测试可视化修复效果"""
    print("🚀 Testing visualization fixes...")
    
    try:
        # 创建可视化器
        visualizer = ModelComparisonVisualizer()
        
        # 加载模型结果
        print("\n📥 Loading model results...")
        success = visualizer.load_model_results()
        
        if success:
            print("✅ Model results loaded successfully")
            
            # 创建综合对比报告
            print("\n📊 Creating comprehensive comparison...")
            output_path = visualizer.create_comprehensive_comparison()
            print(f"✅ Comprehensive comparison saved: {output_path}")
            
            # 创建训练过程分析
            print("\n📈 Creating training process analysis...")
            training_path = visualizer.create_training_process_analysis()
            print(f"✅ Training process analysis saved: {training_path}")
            
            print("\n🎉 All visualizations completed successfully!")
            
        else:
            print("❌ Failed to load model results")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization_fixes()
