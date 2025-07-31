#!/bin/bash

# 锂电池故障检测 - 完整训练流程脚本
echo "🚀 锂电池故障检测 - 完整训练流程"
echo "=================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一张GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 检查环境
echo "🔍 检查环境..."
./check_environment.sh

# 创建必要的目录
echo "📁 创建目录..."
mkdir -p models
mkdir -p logs

# 记录开始时间
start_time=$(date)
echo "⏰ 开始时间: $start_time"

# 步骤1: 生成targets.pkl
echo ""
echo "📊 步骤1: 生成targets.pkl..."
python3 precompute_targets.py > logs/precompute_targets.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ targets.pkl生成成功"
else
    echo "❌ targets.pkl生成失败，请检查日志: logs/precompute_targets.log"
    exit 1
fi

# 步骤2: 训练BiLSTM基准模型
echo ""
echo "🧠 步骤2: 训练BiLSTM基准模型..."
python3 Train_BILSTM.py > logs/train_bilstm.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ BiLSTM训练成功"
else
    echo "❌ BiLSTM训练失败，请检查日志: logs/train_bilstm.log"
    exit 1
fi

# 步骤3: 训练Transformer模型
echo ""
echo "🔧 步骤3: 训练Transformer模型..."
python3 Train_Transformer.py > logs/train_transformer.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Transformer训练成功"
else
    echo "❌ Transformer训练失败，请检查日志: logs/train_transformer.log"
    exit 1
fi

# 步骤4: 运行性能测试
echo ""
echo "🔬 步骤4: 运行性能测试..."
python3 Test_combine.py > logs/test_combine.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 性能测试成功"
else
    echo "❌ 性能测试失败，请检查日志: logs/test_combine.log"
    exit 1
fi

# 记录结束时间
end_time=$(date)
echo ""
echo "⏰ 结束时间: $end_time"
echo "🎉 完整训练流程完成！"

# 显示结果摘要
echo ""
echo "📊 训练结果摘要:"
echo "   模型文件:"
ls -la models/*.pth 2>/dev/null || echo "   无模型文件"
echo ""
echo "   日志文件:"
ls -la logs/ 2>/dev/null || echo "   无日志文件"

echo ""
echo "✅ 所有训练任务完成！" 