#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "🚀 启动分布式训练..."

# 训练Transformer模型
echo "1️⃣ 训练Transformer模型..."
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    Train_Transformer.py

# 等待Transformer训练完成
wait

# 训练BiLSTM模型
echo "2️⃣ 训练BiLSTM模型..."
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    Train_BILSTM.py

echo "✅ 分布式训练完成！"