#!/usr/bin/env python3
"""
调试脚本：检查Three_model目录下各子目录的实际文件
"""
import os

def check_directory_files():
    """检查各目录下的文件"""
    base_path = "/mnt/bz25t/bzhy/datasave/Three_model"
    
    print("🔍 检查目录文件结构")
    print("=" * 60)
    print(f"基础路径: {base_path}")
    print(f"基础路径存在: {os.path.exists(base_path)}")
    
    if not os.path.exists(base_path):
        print("❌ 基础路径不存在")
        return
    
    # 检查子目录
    subdirs = ['BiLSTM', 'transformer_PN', 'transformer_positive']
    
    for subdir in subdirs:
        full_path = os.path.join(base_path, subdir)
        print(f"\n📁 检查目录: {subdir}")
        print(f"   路径: {full_path}")
        print(f"   存在: {os.path.exists(full_path)}")
        
        if os.path.exists(full_path):
            try:
                files = os.listdir(full_path)
                print(f"   文件数量: {len(files)}")
                print("   文件列表:")
                for file in sorted(files):
                    file_path = os.path.join(full_path, file)
                    size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                    file_type = "📄" if os.path.isfile(file_path) else "📁"
                    print(f"     {file_type} {file} ({size} bytes)")
                    
                    # 特别检查需要的文件
                    if file == 'performance_metrics.json':
                        print(f"       ✅ 找到性能指标文件")
                    elif file == 'detailed_results.pkl':
                        print(f"       ✅ 找到详细结果文件")
                        
            except Exception as e:
                print(f"   ❌ 读取目录失败: {e}")
        else:
            print("   ❌ 目录不存在")

if __name__ == "__main__":
    check_directory_files()
