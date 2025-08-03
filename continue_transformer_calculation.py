#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已保存的Transformer模型继续计算剩余结果
避免重新训练，直接加载模型和中间结果
"""

import numpy as np
import pandas as pd
import pickle
import os
import torch
from Function_ import DiagnosisFeature, PCA

def continue_transformer_calculation():
    """从已保存的模型继续计算"""
    print("="*60)
    print("🔄 从已保存的Transformer模型继续计算")
    print("="*60)
    
    # 检查models目录
    if not os.path.exists('models'):
        print("❌ models目录不存在")
        return
    
    # 检查已保存的文件
    model_suffix = "_transformer"
    required_files = [
        f'net_model{model_suffix}.pth',
        f'netx_model{model_suffix}.pth',
        f'transformer_model.pth'
    ]
    
    print("🔍 检查已保存的模型文件...")
    for file in required_files:
        if os.path.exists(f'models/{file}'):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 不存在")
    
    # 检查是否有中间结果文件
    print("\n🔍 检查中间结果文件...")
    intermediate_files = [
        f'ERRORU{model_suffix}.npy',
        f'ERRORX{model_suffix}.npy',
        f'vin2_modified{model_suffix}.npy',
        f'vin3_modified{model_suffix}.npy'
    ]
    
    has_intermediate = False
    for file in intermediate_files:
        if os.path.exists(f'models/{file}'):
            print(f"✅ {file}")
            has_intermediate = True
        else:
            print(f"❌ {file} 不存在")
    
    if not has_intermediate:
        print("\n⚠️  没有找到中间结果文件")
        print("尝试从已保存的模型重新计算中间结果...")
        
        # 检查是否有其他可用的数据文件
        alternative_files = [
            'vin2_modified.npy',
            'vin3_modified.npy',
            'combined_vin2.npy',
            'combined_vin3.npy'
        ]
        
        has_alternative = False
        for file in alternative_files:
            if os.path.exists(f'models/{file}'):
                print(f"✅ 找到替代文件: {file}")
                has_alternative = True
        
        if not has_alternative:
            print("❌ 没有找到任何可用的数据文件")
            print("请先运行Train_Transformer.py到MC-AE训练完成阶段")
            return
        
        # 尝试从替代文件重新计算
        print("🔄 尝试从替代文件重新计算中间结果...")
        try:
            # 这里需要根据实际可用的文件来调整
            # 暂时返回，让用户手动处理
            print("⚠️  需要手动提供ERRORU和ERRORX数据")
            print("建议运行Train_Transformer.py到MC-AE训练完成")
            return
        except Exception as e:
            print(f"❌ 重新计算失败: {e}")
            return
    
    # 加载中间结果
    print("\n📥 加载中间结果...")
    try:
        ERRORU = np.load(f'models/ERRORU{model_suffix}.npy')
        ERRORX = np.load(f'models/ERRORX{model_suffix}.npy')
        print(f"✅ 加载ERRORU: {ERRORU.shape}")
        print(f"✅ 加载ERRORX: {ERRORX.shape}")
    except Exception as e:
        print(f"❌ 加载中间结果失败: {e}")
        return
    
    # 提取诊断特征
    print("\n🔍 提取诊断特征...")
    try:
        df_data = DiagnosisFeature(ERRORU, ERRORX)
        print(f"✅ 诊断特征提取完成: {df_data.shape}")
    except Exception as e:
        print(f"❌ 诊断特征提取失败: {e}")
        return
    
    # 进行PCA分析
    print("\n🔍 进行PCA分析...")
    try:
        v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data, 0.95, 0.95)
        print(f"✅ PCA分析完成:")
        print(f"   主成分数量: {k}")
        print(f"   解释方差比: {v_ratio}")
        print(f"   T²控制限: 95%={T_95_limit:.4f}, 99%={T_99_limit:.4f}")
        print(f"   SPE控制限: 95%={SPE_95_limit:.4f}, 99%={SPE_99_limit:.4f}")
    except Exception as e:
        print(f"❌ PCA分析失败: {e}")
        return
    
    # 保存诊断特征（分块保存，避免文件过大）
    print("\n💾 保存诊断特征...")
    try:
        # 分块保存为CSV（每块50万行，避免内存问题）
        chunk_size = 500000
        total_rows = len(df_data)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"   数据总行数: {total_rows:,}")
        print(f"   分块大小: {chunk_size:,} 行")
        print(f"   分块数量: {num_chunks}")
        
        # 保存分块CSV文件
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df_data.iloc[start_idx:end_idx]
            
            if num_chunks == 1:
                csv_filename = f'models/diagnosis_feature{model_suffix}.csv'
            else:
                csv_filename = f'models/diagnosis_feature{model_suffix}_chunk_{i+1:02d}.csv'
            
            chunk.to_csv(csv_filename, index=False)
            print(f"   保存CSV分块 {i+1}/{num_chunks}: 行 {start_idx+1:,}-{end_idx:,} -> {csv_filename}")
        
        # 分块保存为Excel（每块100万行）
        excel_chunk_size = 1000000
        excel_num_chunks = (total_rows + excel_chunk_size - 1) // excel_chunk_size
        
        with pd.ExcelWriter(f'models/diagnosis_feature{model_suffix}.xlsx', engine='openpyxl') as writer:
            for i in range(excel_num_chunks):
                start_idx = i * excel_chunk_size
                end_idx = min((i + 1) * excel_chunk_size, total_rows)
                chunk = df_data.iloc[start_idx:end_idx]
                sheet_name = f'chunk_{i+1}' if excel_num_chunks > 1 else 'data'
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"   保存Excel分块 {i+1}/{excel_num_chunks}: 行 {start_idx+1:,}-{end_idx:,}")
        
        print(f"✅ 诊断特征已保存:")
        print(f"   - CSV文件: {num_chunks}个分块")
        print(f"   - Excel文件: {excel_num_chunks}个分块")
        if num_chunks > 1:
            print(f"   - 主文件: models/diagnosis_feature{model_suffix}_chunk_01.csv")
        else:
            print(f"   - 主文件: models/diagnosis_feature{model_suffix}.csv")
        
    except Exception as e:
        print(f"❌ 保存诊断特征失败: {e}")
        return
    
    # 保存PCA分析结果
    print("\n💾 保存PCA分析结果...")
    try:
        np.save(f'models/v_I{model_suffix}.npy', v_I)
        np.save(f'models/v{model_suffix}.npy', v)
        np.save(f'models/v_ratio{model_suffix}.npy', v_ratio)
        np.save(f'models/p_k{model_suffix}.npy', p_k)
        np.save(f'models/data_mean{model_suffix}.npy', data_mean)
        np.save(f'models/data_std{model_suffix}.npy', data_std)
        np.save(f'models/T_95_limit{model_suffix}.npy', T_95_limit)
        np.save(f'models/T_99_limit{model_suffix}.npy', T_99_limit)
        np.save(f'models/SPE_95_limit{model_suffix}.npy', SPE_95_limit)
        np.save(f'models/SPE_99_limit{model_suffix}.npy', SPE_99_limit)
        np.save(f'models/P{model_suffix}.npy', P)
        np.save(f'models/k{model_suffix}.npy', k)
        np.save(f'models/P_t{model_suffix}.npy', P_t)
        np.save(f'models/X{model_suffix}.npy', X)
        np.save(f'models/data_nor{model_suffix}.npy', data_nor)
        print(f"✅ PCA分析结果已保存: models/*{model_suffix}.npy")
        
    except Exception as e:
        print(f"❌ 保存PCA分析结果失败: {e}")
        return
    
    # 保存训练历史（如果存在）
    print("\n💾 保存训练历史...")
    try:
        # 尝试加载已有的训练历史
        if os.path.exists(f'models/transformer_mcae_training_history.pkl'):
            with open(f'models/transformer_mcae_training_history.pkl', 'rb') as f:
                mcae_training_history = pickle.load(f)
            print("✅ 已加载现有训练历史")
        else:
            # 创建基本的训练历史
            mcae_training_history = {
                'mcae1_reconstruction_error_mean': np.mean(np.abs(ERRORU)),
                'mcae1_reconstruction_error_std': np.std(np.abs(ERRORU)),
                'mcae2_reconstruction_error_mean': np.mean(np.abs(ERRORX)),
                'mcae2_reconstruction_error_std': np.std(np.abs(ERRORX)),
                'pca_components': k,
                'explained_variance_ratio': v_ratio,
                't2_limits': {'95%': T_95_limit, '99%': T_99_limit},
                'spe_limits': {'95%': SPE_95_limit, '99%': SPE_99_limit}
            }
            print("✅ 创建新的训练历史")
        
        # 更新训练历史
        mcae_training_history.update({
            'diagnosis_feature_shape': df_data.shape,
            'pca_analysis_completed': True,
            'calculation_completed': True
        })
        
        with open(f'models/transformer_mcae_training_history.pkl', 'wb') as f:
            pickle.dump(mcae_training_history, f)
        print(f"✅ 训练历史已保存: models/transformer_mcae_training_history.pkl")
        
    except Exception as e:
        print(f"❌ 保存训练历史失败: {e}")
        return
    
    print("\n🎉 所有计算完成！")
    print("="*60)
    print("📊 计算结果总结:")
    print(f"   诊断特征数据: {df_data.shape}")
    print(f"   PCA主成分数: {k}")
    print(f"   解释方差比: {v_ratio}")
    print(f"   T²控制限: 95%={T_95_limit:.4f}, 99%={T_99_limit:.4f}")
    print(f"   SPE控制限: 95%={SPE_95_limit:.4f}, 99%={SPE_99_limit:.4f}")
    print("")
    print("📁 保存的文件:")
    if num_chunks > 1:
        print(f"   - diagnosis_feature{model_suffix}_chunk_*.csv (分块CSV数据)")
    else:
        print(f"   - diagnosis_feature{model_suffix}.csv (完整CSV数据)")
    print(f"   - diagnosis_feature{model_suffix}.xlsx (分块Excel数据)")
    print(f"   - 所有PCA分析结果 (*{model_suffix}.npy)")
    print(f"   - 训练历史 (transformer_mcae_training_history.pkl)")

if __name__ == "__main__":
    continue_transformer_calculation() 