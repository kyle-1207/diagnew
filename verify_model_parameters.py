#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证BILSTM和Transformer模型参数量匹配
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Class_ import LSTM

class TransformerPredictor(nn.Module):
    """复制Transformer模型结构用于参数量对比"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=3, output_size=2):
        super(TransformerPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model, dtype=torch.float32))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, output_size)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def main():
    print("🔍 验证BILSTM和Transformer模型参数量匹配")
    print("=" * 60)
    
    # 创建BILSTM模型
    print("1. 创建BILSTM模型...")
    bilstm_model = LSTM()
    bilstm_params = bilstm_model.count_parameters()
    
    # 创建Transformer模型
    print("2. 创建Transformer模型...")
    transformer_model = TransformerPredictor(
        input_size=7,
        d_model=128,
        nhead=8,
        num_layers=3,
        output_size=2
    )
    transformer_params = transformer_model.count_parameters()
    
    # 参数量对比
    print("\n📊 模型参数量对比:")
    print("-" * 40)
    print(f"BILSTM模型参数量:      {bilstm_params:,}")
    print(f"Transformer模型参数量: {transformer_params:,}")
    print(f"参数量比例:            {bilstm_params/transformer_params:.3f}")
    
    print(f"\nBILSTM模型规模:        {bilstm_params/1e6:.2f}M 参数")
    print(f"Transformer模型规模:   {transformer_params/1e6:.2f}M 参数")
    
    # 判断是否匹配
    ratio = bilstm_params / transformer_params
    if 0.3 <= ratio <= 1.5:  # 在合理范围内
        print(f"\n✅ 参数量匹配良好！比例 {ratio:.3f} 在合理范围内 (0.3-1.5)")
    else:
        print(f"\n⚠️  参数量差异较大！比例 {ratio:.3f} 超出合理范围")
    
    # 详细结构对比
    print("\n🏗️  模型结构对比:")
    print("-" * 40)
    print("BILSTM:")
    print("  - BiLSTM: input_size=7, hidden_size=128, num_layers=3, bidirectional=True")
    print("  - FC网络: 256 → 128 → 64 → 2")
    print("  - Dropout: 0.1")
    
    print("\nTransformer:")
    print("  - 输入投影: 7 → 128")
    print("  - 位置编码: 1000×128")
    print("  - Transformer编码器: d_model=128, nhead=8, num_layers=3")
    print("  - 输出网络: 128 → 64 → 2")
    print("  - Dropout: 0.1")
    
    print("\n" + "=" * 60)
    print("验证完成！")

if __name__ == "__main__":
    main()
