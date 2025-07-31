#!/bin/bash

# Linux环境安装脚本
echo "🚀 开始安装Linux环境依赖..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 检测到Python版本: $python_version"

# 检查pip是否可用
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未安装，请先安装pip3"
    exit 1
fi

# 升级pip
echo "📦 升级pip..."
pip3 install --upgrade pip

# 安装依赖包
echo "📦 安装依赖包..."
pip3 install -r requirements.txt

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 检测到NVIDIA GPU，安装CUDA版本的PyTorch..."
    
    # 获取CUDA版本
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "📊 检测到CUDA版本: $cuda_version"
    
    # 根据CUDA版本选择PyTorch
    if [[ "$cuda_version" == "12.3" ]]; then
        echo "🔧 安装CUDA 12.3兼容的PyTorch..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11.8" ]]; then
        echo "🔧 安装CUDA 11.8兼容的PyTorch..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "🔧 安装CUDA 11.8兼容的PyTorch（默认）..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 显示GPU信息
    echo "📊 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    
else
    echo "⚠️  未检测到NVIDIA GPU，安装CPU版本的PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "✅ 环境安装完成！"
echo ""
echo "📋 使用说明："
echo "1. 确保数据文件在 ./data/QAS/ 目录下"
echo "2. 运行预计算: python3 precompute_targets.py"
echo "3. 运行训练: python3 Train_Transformer.py"
echo "4. 运行测试: python3 Test_combine.py" 