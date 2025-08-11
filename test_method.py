#!/usr/bin/env python3
"""测试 _generate_roc_curve_from_auc 方法是否存在"""

import sys
import os
sys.path.append('.')

try:
    from Visualize_Model_Comparison import ModelComparisonVisualizer
    
    # 创建实例
    visualizer = ModelComparisonVisualizer()
    
    # 检查方法是否存在
    if hasattr(visualizer, '_generate_roc_curve_from_auc'):
        print("✅ 方法 _generate_roc_curve_from_auc 存在")
        
        # 尝试调用方法
        import numpy as np
        fpr = np.linspace(0, 1, 100)
        auc_score = 0.8
        
        try:
            tpr = visualizer._generate_roc_curve_from_auc(fpr, auc_score)
            print(f"✅ 方法调用成功，返回长度: {len(tpr)}")
        except Exception as e:
            print(f"❌ 方法调用失败: {e}")
    else:
        print("❌ 方法 _generate_roc_curve_from_auc 不存在")
        print("可用的方法:")
        methods = [method for method in dir(visualizer) if not method.startswith('__')]
        for method in methods:
            print(f"  - {method}")
            
except Exception as e:
    print(f"❌ 导入失败: {e}")
