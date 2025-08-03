#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的实验执行脚本
直接调用现有函数，快速验证想法
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import pickle
from datetime import datetime

# 导入现有模块
try:
    from Function_ import *
    from Class_ import *
    from Comprehensive_calculation import Comprehensive_calculation
    print("✅ 成功导入现有模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

def quick_test():
    """快速测试现有函数"""
    
    print("\n🧪 快速测试现有函数...")
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    try:
        # 尝试加载一个QAS样本
        sample_path = "../QAS/0"
        if os.path.exists(sample_path):
            files = os.listdir(sample_path)
            print(f"   样本0包含文件: {files}")
        else:
            print(f"   ⚠️  样本路径不存在: {sample_path}")
    except Exception as e:
        print(f"   ❌ 数据加载测试失败: {e}")
    
    # 2. 测试PCA函数
    print("2. 测试PCA函数...")
    try:
        # 创建模拟数据测试PCA
        test_data = np.random.rand(100, 7)  # 100个样本，7个特征
        
        result = PCA(test_data, 0.99, 0.99)
        v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = result
        
        print(f"   ✅ PCA成功，主成分数量: {len(k)}，累积贡献率: {v_ratio[k[0]-1]:.3f}")
        
    except Exception as e:
        print(f"   ❌ PCA测试失败: {e}")
    
    # 3. 测试综合诊断计算
    print("3. 测试综合诊断计算...")
    try:
        # 使用PCA结果测试综合诊断
        test_X = np.random.rand(50, 7)  # 50个测试样本
        time_index = np.arange(50)
        
        # 调用综合诊断函数
        diagnosis_result = Comprehensive_calculation(
            X=test_X,
            data_mean=data_mean,
            data_std=data_std,
            v=v,
            p_k=p_k,
            v_I=v_I,
            T_99_limit=T_99_limit,
            SPE_99_limit=SPE_99_limit,
            P=P,
            time=time_index
        )
        
        # 解包结果
        lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean_out, data_std_out = diagnosis_result
        
        print(f"   ✅ 综合诊断成功")
        print(f"      FAI范围: [{fai.min():.4f}, {fai.max():.4f}]")
        print(f"      最大报警等级: {maxlevel}")
        print(f"      故障点数量: {np.sum(level > 0)}")
        
        return diagnosis_result
        
    except Exception as e:
        print(f"   ❌ 综合诊断测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_real_data_sample():
    """尝试加载真实数据样本"""
    
    print("\n📊 尝试加载真实数据...")
    
    # QAS样本路径
    qas_samples = [f"../QAS/{i}" for i in range(10)]
    
    loaded_data = {}
    
    for sample_path in qas_samples:
        sample_id = os.path.basename(sample_path)
        
        try:
            # 检查样本文件
            if not os.path.exists(sample_path):
                continue
                
            files = [f for f in os.listdir(sample_path) if f.endswith('.pkl')]
            
            if len(files) >= 3:  # 至少要有3个pkl文件
                sample_data = {}
                
                for file in files:
                    file_path = os.path.join(sample_path, file)
                    data = pd.read_pickle(file_path)
                    sample_data[file] = data
                    
                loaded_data[sample_id] = sample_data
                print(f"   ✅ 加载样本 {sample_id}，文件数: {len(files)}")
                
                # 显示第一个样本的数据信息
                if sample_id == '0':
                    print(f"      样本0数据详情:")
                    for file, data in sample_data.items():
                        print(f"        {file}: {data.shape if hasattr(data, 'shape') else len(data)}")
                
            else:
                print(f"   ⚠️  样本 {sample_id} 文件不完整，跳过")
                
        except Exception as e:
            print(f"   ❌ 加载样本 {sample_id} 失败: {e}")
    
    print(f"📈 成功加载 {len(loaded_data)} 个样本")
    return loaded_data

def simple_transformer_test(loaded_data):
    """简单的Transformer测试"""
    
    print("\n🤖 简单Transformer测试...")
    
    if len(loaded_data) == 0:
        print("   ⚠️  没有可用数据，跳过Transformer测试")
        return
    
    try:
        # 使用第一个样本构建简单特征
        sample_0 = loaded_data.get('0')
        if not sample_0:
            print("   ❌ 样本0不可用")
            return
        
        # 简单地合并所有数据作为特征（实际应用中需要更仔细的特征工程）
        feature_data = []
        
        for file, data in sample_0.items():
            if hasattr(data, 'values'):
                feature_data.append(data.values.flatten()[:1000])  # 取前1000个数据点
            else:
                feature_data.append(np.array(data).flatten()[:1000])
        
        # 对齐数据长度
        min_length = min(len(fd) for fd in feature_data)
        feature_matrix = np.column_stack([fd[:min_length] for fd in feature_data])
        
        print(f"   📊 构建特征矩阵: {feature_matrix.shape}")
        
        # 创建简单的时序数据
        seq_len = 50
        if feature_matrix.shape[0] < seq_len:
            print(f"   ⚠️  数据长度不足，需要至少{seq_len}个时间步")
            return
        
        # 创建序列
        sequences = []
        targets = []
        
        for i in range(feature_matrix.shape[0] - seq_len):
            sequences.append(feature_matrix[i:i+seq_len])
            # 简单的目标：预测下一步的前两个特征
            targets.append(feature_matrix[i+seq_len, :2])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"   🔄 时序数据: X={X.shape}, y={y.shape}")
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # 创建简单的transformer模型
        from Train_Test_Integrated import TransformerPredictor
        
        model = TransformerPredictor(
            input_size=X.shape[2],
            d_model=64,  # 较小的模型用于快速测试
            nhead=4,
            num_layers=2,
            d_ff=128,
            dropout=0.1,
            output_size=2
        ).to(device)
        
        print(f"   🏗️  创建Transformer模型，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 简单训练测试
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        
        # 只训练几个epoch进行测试
        for epoch in range(5):
            optimizer.zero_grad()
            
            # 随机选择一个batch
            batch_size = min(16, len(X_tensor))
            indices = torch.randperm(len(X_tensor))[:batch_size]
            
            batch_X = X_tensor[indices]
            batch_y = y_tensor[indices]
            
            outputs = model(batch_X)
            
            # 只使用最后一个时间步的输出
            loss = criterion(outputs[:, -1, :], batch_y)
            
            loss.backward()
            optimizer.step()
            
            print(f"     Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        # 测试预测
        model.eval()
        with torch.no_grad():
            test_output = model(X_tensor[:5])  # 测试前5个序列
            print(f"   🎯 测试预测完成，输出形状: {test_output.shape}")
            
        print("   ✅ Transformer简单测试成功！")
        
        return model, X_tensor, y_tensor
        
    except Exception as e:
        print(f"   ❌ Transformer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_test_results(results):
    """保存测试结果"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"test_results_{timestamp}.pkl"
    
    try:
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 测试结果已保存: {result_file}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def main():
    """主函数"""
    
    print("🚀 开始快速功能测试...")
    print("="*60)
    
    # 1. 测试现有函数
    diagnosis_result = quick_test()
    
    # 2. 加载真实数据
    loaded_data = load_real_data_sample()
    
    # 3. 简单Transformer测试
    transformer_result = simple_transformer_test(loaded_data)
    
    # 4. 保存结果
    results = {
        'diagnosis_result': diagnosis_result,
        'loaded_data_info': {k: {file: data.shape if hasattr(data, 'shape') else len(data) 
                                for file, data in v.items()} 
                           for k, v in loaded_data.items()},
        'transformer_test': transformer_result is not None
    }
    
    save_test_results(results)
    
    print("\n🎉 快速测试完成！")
    print("="*60)
    
    if diagnosis_result and loaded_data and transformer_result:
        print("✅ 所有功能测试通过，可以进行完整实验")
        print("\n💡 下一步建议:")
        print("   1. 运行 python Train_Test_Integrated.py 进行完整实验")
        print("   2. 检查 modelsfl_* 目录中的结果")
        print("   3. 查看生成的可视化图表")
    else:
        print("⚠️  部分功能测试失败，请检查:")
        print("   - 数据文件是否存在 (../QAS/0/*.pkl)")
        print("   - 依赖模块是否正确导入")
        print("   - GPU/CPU环境是否配置正确")

if __name__ == "__main__":
    main()