#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的三模型对比脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_script():
    """测试修复后的脚本"""
    print("🧪 测试修复后的三模型对比脚本...")
    
    try:
        # 导入修复后的对比器
        from Compare_Three_Models import ThreeModelComparator
        print("✅ 成功导入ThreeModelComparator")
        
        # 创建对比器实例
        comparator = ThreeModelComparator()
        print("✅ 成功创建对比器实例")
        
        # 检查是否有数据，没有则创建示例数据
        if comparator.base_path is None:
            print("📦 创建示例数据...")
            comparator.create_sample_data()
            print("✅ 示例数据创建完成")
            
            # 重新初始化
            comparator = ThreeModelComparator()
            print("✅ 重新初始化完成")
        
        # 测试数据加载
        print("\n📊 测试数据加载...")
        success = comparator.load_all_data()
        
        if success:
            print(f"✅ 成功加载 {len(comparator.model_data)} 个模型的数据")
            
            # 显示模型信息
            for model_name, data in comparator.model_data.items():
                performance = data['performance']
                print(f"🔸 {model_name}: Accuracy={performance.get('accuracy', 0):.4f}")
            
            print("\n🎨 测试生成一个ROC对比图...")
            try:
                comparator.generate_roc_comparison()
                print("✅ ROC对比图生成成功")
                
                # 检查文件是否存在
                if os.path.exists("Three_model/comparison_roc_curves.png"):
                    print("✅ ROC对比图文件已保存")
                else:
                    print("⚠️ ROC对比图文件未找到")
                    
            except Exception as e:
                print(f"❌ ROC对比图生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("❌ 数据加载失败")
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_script()
