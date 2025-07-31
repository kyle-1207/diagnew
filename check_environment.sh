#!/bin/bash

# Linux环境检查脚本
echo "🔍 Linux环境检查脚本"
echo "===================="

# 1. 系统信息
echo "📋 系统信息:"
echo "   操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "   内核版本: $(uname -r)"
echo "   架构: $(uname -m)"

# 2. Python环境
echo ""
echo "🐍 Python环境:"
python3 --version
pip3 --version

# 3. GPU环境
echo ""
echo "🚀 GPU环境:"
if command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA驱动: 已安装"
    nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader,nounits | while IFS=, read -r name driver cuda memory; do
        echo "   GPU: $name"
        echo "   驱动版本: $driver"
        echo "   CUDA版本: $cuda"
        echo "   显存: ${memory}MB"
    done
else
    echo "   ❌ NVIDIA驱动未安装"
fi

# 4. CUDA Toolkit
echo ""
echo "🔧 CUDA Toolkit:"
if command -v nvcc &> /dev/null; then
    echo "   ✅ CUDA Toolkit已安装"
    nvcc --version
else
    echo "   ⚠️  CUDA Toolkit未安装（不影响PyTorch运行）"
fi

# 5. 内存和存储
echo ""
echo "💾 系统资源:"
echo "   内存: $(free -h | grep Mem | awk '{print $2}')"
echo "   可用内存: $(free -h | grep Mem | awk '{print $7}')"
echo "   存储空间: $(df -h . | tail -1 | awk '{print $4}') 可用"

# 6. 检查数据目录
echo ""
echo "📁 数据目录检查:"
if [ -d "./data/QAS" ]; then
    echo "   ✅ QAS数据目录存在"
    qas_count=$(ls -1 ./data/QAS/ | wc -l)
    echo "   QAS样本数量: $qas_count"
else
    echo "   ❌ QAS数据目录不存在"
fi

if [ -d "./data/test" ]; then
    echo "   ✅ test数据目录存在"
    test_count=$(ls -1 ./data/test/ | wc -l)
    echo "   test样本数量: $test_count"
else
    echo "   ❌ test数据目录不存在"
fi

# 7. 检查模型目录
echo ""
echo "🧠 模型目录检查:"
if [ -d "./models" ]; then
    echo "   ✅ models目录存在"
    model_count=$(ls -1 ./models/*.pth 2>/dev/null | wc -l)
    echo "   模型文件数量: $model_count"
else
    echo "   ⚠️  models目录不存在（训练后会创建）"
fi

echo ""
echo "✅ 环境检查完成！" 