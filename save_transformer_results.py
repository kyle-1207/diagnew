#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存Transformer训练结果的脚本
用于在训练中断后保存剩余的数据
"""

import numpy as np
import pandas as pd
import pickle
import os

def save_transformer_results():
    """保存Transformer训练结果"""
    print("="*60)
    print("💾 保存Transformer训练结果")
    print("="*60)
    
    # 检查models目录是否存在
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ 创建models目录")
    
    # 检查数据是否在内存中
    try:
        # 这些变量应该在训练脚本的内存中
        print("🔍 检查数据状态...")
        
        # 检查关键变量是否存在
        if 'ERRORU' in globals():
            print(f"✅ ERRORU shape: {ERRORU.shape}")
        else:
            print("❌ ERRORU 不在内存中")
            return
            
        if 'ERRORX' in globals():
            print(f"✅ ERRORX shape: {ERRORX.shape}")
        else:
            print("❌ ERRORX 不在内存中")
            return
            
        if 'df_data' in globals():
            print(f"✅ df_data shape: {df_data.shape}")
        else:
            print("❌ df_data 不在内存中")
            return
        
        # 保存诊断特征（分块保存，避免Excel文件过大）
        model_suffix = "_transformer"
        print(f"\n💾 保存诊断特征（数据量: {df_data.shape}）...")
        
        # CSV文件保存（无大小限制）
        csv_path = f'models/diagnosis_feature{model_suffix}.csv'
        df_data.to_csv(csv_path, index=False)
        print(f"✅ 诊断特征CSV已保存: {csv_path}")
        
        # Excel文件分块保存（避免超过Excel行数限制）
        excel_path = f'models/diagnosis_feature{model_suffix}.xlsx'
        max_rows_per_sheet = 1000000  # Excel限制约104万行，留些余量
        
        if len(df_data) > max_rows_per_sheet:
            print(f"⚠️  数据量过大({len(df_data)}行)，进行分块保存...")
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 计算需要多少个工作表
                num_sheets = (len(df_data) + max_rows_per_sheet - 1) // max_rows_per_sheet
                
                for i in range(num_sheets):
                    start_idx = i * max_rows_per_sheet
                    end_idx = min((i + 1) * max_rows_per_sheet, len(df_data))
                    chunk = df_data.iloc[start_idx:end_idx]
                    
                    sheet_name = f'Sheet_{i+1}' if i > 0 else 'Sheet_1'
                    chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   工作表 {i+1}/{num_sheets}: {start_idx+1}-{end_idx} 行")
            
            print(f"✅ 诊断特征Excel已分块保存: {excel_path} ({num_sheets}个工作表)")
        else:
            # 数据量不大，直接保存
            df_data.to_excel(excel_path, index=False)
            print(f"✅ 诊断特征Excel已保存: {excel_path}")
        
        # 保存PCA分析结果
        print(f"\n💾 保存PCA分析结果...")
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
        
        # 保存MC-AE训练历史
        print(f"\n💾 保存MC-AE训练历史...")
        mcae_training_history = {
            'mcae1_losses': train_losses_mcae1,
            'mcae2_losses': train_losses_mcae2,
            'final_mcae1_loss': train_losses_mcae1[-1],
            'final_mcae2_loss': train_losses_mcae2[-1],
            'mcae1_reconstruction_error_mean': np.mean(np.abs(ERRORU)),
            'mcae1_reconstruction_error_std': np.std(np.abs(ERRORU)),
            'mcae2_reconstruction_error_mean': np.mean(np.abs(ERRORX)),
            'mcae2_reconstruction_error_std': np.std(np.abs(ERRORX)),
            'training_samples': len(train_samples),
            'epochs': EPOCH_MCAE,
            'learning_rate': LR_MCAE,
            'batch_size': BATCHSIZE_MCAE
        }
        
        with open(f'models/transformer_mcae_training_history.pkl', 'wb') as f:
            pickle.dump(mcae_training_history, f)
        print(f"✅ MC-AE训练历史已保存: models/transformer_mcae_training_history.pkl")
        
        print("\n🎉 所有数据保存完成！")
        
    except Exception as e:
        print(f"❌ 保存过程中出现错误: {e}")
        print("请确保在Train_Transformer.py运行环境中执行此脚本")

if __name__ == "__main__":
    save_transformer_results() 