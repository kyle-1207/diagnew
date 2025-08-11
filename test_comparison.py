#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试三模型对比脚本
"""

from Compare_Three_Models import ThreeModelComparator

def test_comparison():
    """测试对比功能"""
    print("🧪 测试三模型对比脚本...")
    
    # 创建对比器
    comparator = ThreeModelComparator()
    
    # 如果没有数据，创建示例数据
    if comparator.base_path is None:
        print("📦 创建示例数据...")
        comparator.create_sample_data()
        comparator = ThreeModelComparator()  # 重新初始化
    
    # 测试数据加载
    print("\n📊 测试数据加载...")
    success = comparator.load_all_data()
    
    if success:
        print(f"✅ 成功加载 {len(comparator.model_data)} 个模型的数据")
        
        # 显示加载的数据概览
        for model_name, data in comparator.model_data.items():
            performance = data['performance']
            print(f"\n🔸 {model_name}:")
            print(f"   - 准确率: {performance.get('accuracy', 0):.4f}")
            print(f"   - AUC: {performance.get('auc', 0):.4f}")
            print(f"   - F1分数: {performance.get('f1_score', 0):.4f}")
        
        # 测试生成一个简单的对比图
        print("\n🎨 测试生成ROC对比图...")
        try:
            comparator.generate_roc_comparison()
            print("✅ ROC对比图生成成功")
        except Exception as e:
            print(f"❌ ROC对比图生成失败: {e}")
        
    else:
        print("❌ 数据加载失败")

if __name__ == "__main__":
    test_comparison()
