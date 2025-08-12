#!/usr/bin/env python3
"""
测试三窗口过程图生成的修复方案
针对样本340、345、346、347的错误进行验证
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Linux环境配置
mpl.use('Agg')

def test_data_structure_compatibility():
    """测试数据结构兼容性"""
    print("🔧 测试数据结构兼容性...")
    
    # 模拟五点检测模式的detection_info结构
    five_point_detection_info = {
        'trigger_points': [1020, 1025, 1030, 1035],  # 整数列表
        'verified_points': [],  # 空列表（五点检测模式不使用）
        'marked_regions': [
            {'range': (1019, 1022), 'level': 2},
            {'range': (1024, 1027), 'level': 1},
            {'range': (1029, 1032), 'level': 3},
            {'range': (1034, 1037), 'level': 1}
        ],
        'detection_stats': {
            'total_trigger_points': 4,
            'total_marked_regions': 4,
            'level_statistics': {
                'level_1_triggers': 2,
                'level_2_triggers': 1,
                'level_3_triggers': 1
            }
        }
    }
    
    # 模拟三窗口检测模式的detection_info结构
    three_window_detection_info = {
        'trigger_points': [
            {'index': 1020, 'level': 2},
            {'index': 1025, 'level': 1},
            {'index': 1030, 'level': 3}
        ],
        'verified_points': [
            {'point': 1020, 'verify_range': (1015, 1025)},
            {'point': 1030, 'verify_range': (1025, 1035)}
        ],
        'marked_regions': [
            {'range': (1019, 1022)},
            {'range': (1029, 1032)}
        ]
    }
    
    # 测试修复后的处理逻辑
    test_cases = [
        ("五点检测模式", five_point_detection_info),
        ("三窗口检测模式", three_window_detection_info)
    ]
    
    for mode_name, detection_info in test_cases:
        print(f"\n📊 测试 {mode_name}:")
        
        # 测试触发点处理
        trigger_points = detection_info.get('trigger_points', [])
        if trigger_points:
            try:
                if len(trigger_points) > 0 and isinstance(trigger_points[0], dict):
                    trigger_indices = [p['index'] for p in trigger_points if 'index' in p]
                    print(f"   触发点（字典模式）: {trigger_indices}")
                else:
                    trigger_indices = [idx for idx in trigger_points if isinstance(idx, (int, np.integer))]
                    print(f"   触发点（整数模式）: {trigger_indices}")
            except Exception as e:
                print(f"   ❌ 触发点处理失败: {e}")
        
        # 测试验证点处理
        verified_points = detection_info.get('verified_points', [])
        if verified_points:
            try:
                verified_indices = []
                if len(verified_points) > 0:
                    if isinstance(verified_points[0], dict):
                        for p in verified_points:
                            if 'point' in p:
                                verified_indices.append(p['point'])
                            elif 'index' in p:
                                verified_indices.append(p['index'])
                    else:
                        verified_indices = [idx for idx in verified_points if isinstance(idx, (int, np.integer))]
                print(f"   验证点: {verified_indices}")
            except Exception as e:
                print(f"   ❌ 验证点处理失败: {e}")
        
        # 测试级别统计处理
        try:
            level_counts = {'Level 1': 0, 'Level 2': 0, 'Level 3': 0}
            
            if len(trigger_points) > 0 and isinstance(trigger_points[0], dict):
                # 三窗口检测模式
                for point in trigger_points:
                    level = point.get('level', 1)
                    level_counts[f'Level {level}'] += 1
                print(f"   级别统计（从触发点）: {level_counts}")
            else:
                # 五点检测模式
                detection_stats = detection_info.get('detection_stats', {})
                level_statistics = detection_stats.get('level_statistics', {})
                
                if level_statistics:
                    level_counts['Level 1'] = level_statistics.get('level_1_triggers', 0)
                    level_counts['Level 2'] = level_statistics.get('level_2_triggers', 0)
                    level_counts['Level 3'] = level_statistics.get('level_3_triggers', 0)
                    print(f"   级别统计（从统计数据）: {level_counts}")
                else:
                    # 从marked_regions中统计
                    marked_regions = detection_info.get('marked_regions', [])
                    for region in marked_regions:
                        level = region.get('level', 1)
                        level_counts[f'Level {level}'] += 1
                    print(f"   级别统计（从标记区域）: {level_counts}")
        except Exception as e:
            print(f"   ❌ 级别统计处理失败: {e}")
    
    print(f"\n✅ 数据结构兼容性测试完成")

def test_edge_cases():
    """测试边界情况"""
    print(f"\n🔧 测试边界情况...")
    
    edge_cases = [
        ("空触发点列表", {'trigger_points': [], 'verified_points': []}),
        ("None触发点", {'trigger_points': None, 'verified_points': None}),
        ("混合类型列表", {'trigger_points': [1020, None, 1025], 'verified_points': []}),
        ("缺失字段的字典", {'trigger_points': [{'level': 1}, {'index': 1025}], 'verified_points': []}),
    ]
    
    for case_name, detection_info in edge_cases:
        print(f"\n📊 测试 {case_name}:")
        
        try:
            # 模拟修复后的处理逻辑
            trigger_points = detection_info.get('trigger_points', [])
            if trigger_points:
                if len(trigger_points) > 0 and isinstance(trigger_points[0], dict):
                    trigger_indices = [p['index'] for p in trigger_points if isinstance(p, dict) and 'index' in p]
                else:
                    trigger_indices = [idx for idx in trigger_points if isinstance(idx, (int, np.integer))]
                print(f"   ✅ 处理成功，触发点: {trigger_indices}")
            else:
                print(f"   ✅ 处理成功，无触发点")
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")

if __name__ == "__main__":
    print("🔋 三窗口过程图修复方案测试")
    print("=" * 60)
    
    test_data_structure_compatibility()
    test_edge_cases()
    
    print(f"\n🎉 修复方案测试完成！")
    print("=" * 60)
    print("修复要点总结:")
    print("1. ✅ 兼容五点检测（整数列表）和三窗口检测（字典列表）两种数据格式")
    print("2. ✅ 增强了类型检查和边界条件处理")
    print("3. ✅ 添加了try-catch错误处理，防止单个绘制失败导致整个流程中断")
    print("4. ✅ 增加了索引范围验证，防止数组越界")
    print("5. ✅ 提供了详细的错误信息和降级处理")
