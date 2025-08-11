#!/usr/bin/env python3
"""
Debug script to check if model files exist
"""
import os

base_path = "/mnt/bz25t/bzhy/datasave/Three_model/"

model_configs = {
    'BiLSTM': {
        'folder': 'BILSTM',
        'performance_file': 'bilstm_performance_metrics.json',
        'detailed_file': 'bilstm_detailed_results.pkl',
    },
    'Transformer-PN': {
        'folder': 'transformer_PN',
        'performance_file': 'transformer_performance_metrics.json',
        'detailed_file': 'transformer_detailed_results.pkl',
    },
    'Transformer-Positive': {
        'folder': 'transformer_positive',
        'performance_file': 'transformer_performance_metrics.json',
        'detailed_file': 'transformer_detailed_results.pkl',
    }
}

print(f"🔍 Base path: {base_path}")
print(f"📁 Base path exists: {os.path.exists(base_path)}")
print()

for model_name, config in model_configs.items():
    print(f"📂 Checking {model_name}...")
    
    folder_path = os.path.join(base_path, config['folder'])
    performance_path = os.path.join(folder_path, config['performance_file'])
    detailed_path = os.path.join(folder_path, config['detailed_file'])
    
    print(f"   🗂️  Folder: {folder_path}")
    print(f"   📁 Folder exists: {os.path.exists(folder_path)}")
    
    if os.path.exists(folder_path):
        print(f"   📄 Files in folder:")
        try:
            files = os.listdir(folder_path)
            for file in sorted(files):
                print(f"      - {file}")
        except Exception as e:
            print(f"      ❌ Error listing files: {e}")
    
    print(f"   📊 Performance file: {performance_path}")
    print(f"   ✅ Performance exists: {os.path.exists(performance_path)}")
    
    print(f"   📋 Detailed file: {detailed_path}")
    print(f"   ✅ Detailed exists: {os.path.exists(detailed_path)}")
    print()
