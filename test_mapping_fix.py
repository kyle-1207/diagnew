#!/usr/bin/env python3
"""测试模型映射配置修复"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Run_Complete_Visualization import CompleteVisualizationRunner

def test_mapping_fix():
    """测试映射配置修复"""
    print("🧪 测试模型映射配置修复...")
    
    try:
        # 创建可视化运行器实例
        runner = CompleteVisualizationRunner()
        
        print("\n📋 原始配置:")
        print("model_paths:", runner.model_paths)
        print("model_name_mapping:", runner.model_name_mapping)
        print("model_file_patterns keys:", list(runner.model_file_patterns.keys()))
        
        print("\n🔄 映射后配置:")
        mapped_paths = runner.get_mapped_model_paths()
        mapped_patterns = runner.get_mapped_file_patterns()
        
        print("mapped_model_paths:", mapped_paths)
        print("mapped_file_patterns keys:", list(mapped_patterns.keys()))
        
        print("\n✅ 验证映射关系:")
        expected_display_names = ['BiLSTM', 'Transformer-BACK', 'Transformer-FOR-BACK']
        
        for display_name in expected_display_names:
            if display_name in mapped_paths:
                print(f"   ✅ {display_name}: 路径映射正确")
            else:
                print(f"   ❌ {display_name}: 路径映射缺失")
                
            if display_name in mapped_patterns:
                print(f"   ✅ {display_name}: 文件模式映射正确")
            else:
                print(f"   ❌ {display_name}: 文件模式映射缺失")
        
        print("\n🎯 映射配置修复测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mapping_fix()
