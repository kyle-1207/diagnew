#!/usr/bin/env python3
"""
测试Test_combine_transonly.py路径修改是否正确
"""

import os
from datetime import datetime

def test_result_path():
    """测试结果保存路径"""
    print("🔧 测试结果保存路径修改...")
    
    # 模拟脚本中的路径生成逻辑
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/mnt/bz25t/bzhy/datasave/Transformer/transformer_test_results_{timestamp}"
    
    print(f"   生成的结果目录: {result_dir}")
    
    # 检查父目录是否存在
    parent_dir = "/mnt/bz25t/bzhy/datasave/Transformer"
    if os.path.exists(parent_dir):
        print(f"   ✅ 父目录存在: {parent_dir}")
    else:
        print(f"   ❌ 父目录不存在: {parent_dir}")
        print(f"   💡 需要创建目录: mkdir -p {parent_dir}")
    
    # 检查是否有写入权限
    try:
        test_file = f"{parent_dir}/test_write_permission.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"   ✅ 写入权限正常")
    except Exception as e:
        print(f"   ❌ 写入权限问题: {e}")
    
    # 显示目录结构
    print(f"\n📁 预期的目录结构:")
    print(f"   {parent_dir}/")
    print(f"   └── transformer_test_results_{timestamp}/")
    print(f"       ├── visualizations/")
    print(f"       │   ├── transformer_roc_analysis.png")
    print(f"       │   ├── transformer_fault_detection_timeline.png")
    print(f"       │   ├── transformer_performance_radar.png")
    print(f"       │   └── transformer_three_window_process.png")
    print(f"       ├── detailed_results/")
    print(f"       │   ├── transformer_detailed_results.pkl")
    print(f"       │   ├── transformer_test_metadata.json")
    print(f"       │   └── transformer_summary.xlsx")
    print(f"       └── transformer_performance_metrics.json")

def main():
    """主函数"""
    print("🧪 测试Test_combine_transonly.py路径修改验证")
    print("=" * 60)
    
    test_result_path()
    
    print("\n" + "=" * 60)
    print("✅ 路径修改测试完成!")
    print("\n💡 建议:")
    print("   1. 确保 /mnt/bz25t/bzhy/datasave/Transformer 目录存在")
    print("   2. 确保有足够的磁盘空间")
    print("   3. 确保有写入权限")

if __name__ == "__main__":
    main() 