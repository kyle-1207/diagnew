#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试保存路径可用性的脚本
"""

import os
import shutil
import tempfile
import numpy as np
import pickle

def test_save_paths():
    """测试各种保存路径的可用性"""
    
    # 测试路径列表
    test_paths = [
        'models/',
        '/tmp/',
        './',
        '/mnt/bz25t/bzhy/zhanglikang/project/',
        '/mnt/bz25t/bzhy/'
    ]
    
    print("🔍 测试保存路径可用性...")
    print("="*60)
    
    # 创建测试数据
    test_data = np.random.rand(100, 100)
    test_dict = {'test': 'data', 'array': test_data}
    
    working_paths = []
    
    for path in test_paths:
        print(f"\n📁 测试路径: {path}")
        
        try:
            # 检查路径是否存在，如果不存在则创建
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"   ✅ 目录创建成功")
            
            # 检查磁盘空间
            try:
                total, used, free = shutil.disk_usage(path)
                print(f"   📊 总空间: {total / (1024**3):.2f} GB")
                print(f"   📊 已用空间: {used / (1024**3):.2f} GB")
                print(f"   📊 可用空间: {free / (1024**3):.2f} GB")
            except Exception as e:
                print(f"   ⚠️ 无法获取磁盘信息: {e}")
            
            # 测试写入权限
            test_file = os.path.join(path, 'test_write.tmp')
            
            # 测试numpy保存
            try:
                np.save(test_file, test_data)
                print(f"   ✅ numpy保存测试通过")
                os.remove(test_file)
            except Exception as e:
                print(f"   ❌ numpy保存测试失败: {e}")
                continue
            
            # 测试pickle保存
            try:
                with open(test_file, 'wb') as f:
                    pickle.dump(test_dict, f)
                print(f"   ✅ pickle保存测试通过")
                os.remove(test_file)
            except Exception as e:
                print(f"   ❌ pickle保存测试失败: {e}")
                continue
            
            working_paths.append(path)
            print(f"   🎉 路径 {path} 完全可用!")
            
        except Exception as e:
            print(f"   ❌ 路径测试失败: {e}")
            print(f"   错误代码: {getattr(e, 'errno', 'N/A')}")
            print(f"   错误信息: {getattr(e, 'strerror', 'N/A')}")
    
    print("\n" + "="*60)
    print("📋 测试结果总结:")
    
    if working_paths:
        print("✅ 可用的保存路径:")
        for path in working_paths:
            print(f"   - {path}")
    else:
        print("❌ 没有找到可用的保存路径!")
        print("💡 建议:")
        print("   1. 检查当前目录权限")
        print("   2. 检查 /tmp 目录权限")
        print("   3. 检查 /mnt/bz25t/bzhy 目录是否存在和可写")
        print("   4. 尝试手动创建目录: mkdir -p /mnt/bz25t/bzhy/zhanglikang/project")
    
    return working_paths

if __name__ == "__main__":
    working_paths = test_save_paths()
    print(f"\n🎯 找到 {len(working_paths)} 个可用保存路径") 