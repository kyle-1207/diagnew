#!/usr/bin/env python3
"""
快速验证三个模型的路径配置是否正确
"""

import os

def verify_model_paths():
    """验证模型路径配置"""
    print("🔍 验证模型路径配置")
    print("=" * 50)
    
    # 基础路径
    base_path = "/mnt/bz25t/bzhy/datasave/Three_model"
    print(f"基础路径: {base_path}")
    print(f"基础路径存在: {os.path.exists(base_path)}")
    
    # 模型配置
    models = {
        'BiLSTM': 'BiLSTM',
        'Transformer-Positive': 'transformer_positive', 
        'Transformer-PN': 'transformer_PN'
    }
    
    print(f"\n📁 检查模型目录:")
    for model_name, folder in models.items():
        full_path = os.path.join(base_path, folder)
        exists = os.path.exists(full_path)
        print(f"  {model_name:20} -> {folder:20} [{full_path}] {'✅' if exists else '❌'}")
        
        if exists:
            # 检查性能指标文件
            perf_file = os.path.join(full_path, 'performance_metrics.json')
            detail_file = os.path.join(full_path, 'detailed_results.pkl')
            print(f"    性能文件: {'✅' if os.path.exists(perf_file) else '❌'} {perf_file}")
            print(f"    详细文件: {'✅' if os.path.exists(detail_file) else '❌'} {detail_file}")
    
    print("=" * 50)

if __name__ == "__main__":
    verify_model_paths()
