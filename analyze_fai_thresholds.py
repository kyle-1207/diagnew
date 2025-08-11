# FAI阈值分析脚本
# 专门用于分析故障样本和正常样本的FAI值分布情况

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import os
import warnings
from Function_ import *
from Class_ import *
from Comprehensive_calculation import Comprehensive_calculation

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_sample_data(sample_id):
    """加载样本数据"""
    base_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}'
    
    with open(f'{base_path}/vin_1.pkl', 'rb') as f:
        vin1_data = pickle.load(f)
    with open(f'{base_path}/vin_2.pkl', 'rb') as f:
        vin2_data = pickle.load(f) 
    with open(f'{base_path}/vin_3.pkl', 'rb') as f:
        vin3_data = pickle.load(f)
        
    return vin1_data, vin2_data, vin3_data

def load_models():
    """加载模型和PCA参数"""
    models = {}
    
    # 加载MC-AE模型
    models['net'] = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                              activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    models['netx'] = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                               activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
    
    # 加载模型权重
    net_path = "/mnt/bz25t/bzhy/datasave/Three_model/BILSTM/net_model_bilstm_baseline.pth"
    netx_path = "/mnt/bz25t/bzhy/datasave/Three_model/BILSTM/netx_model_bilstm_baseline.pth"
    
    models['net'].load_state_dict(torch.load(net_path, map_location=device), strict=False)
    models['netx'].load_state_dict(torch.load(netx_path, map_location=device), strict=False)
    
    # 加载PCA参数
    pca_params_path = "/mnt/bz25t/bzhy/datasave/Three_model/BILSTM/pca_params_bilstm_baseline.pkl"
    with open(pca_params_path, 'rb') as f:
        models['pca_params'] = pickle.load(f)
    
    return models

def compute_fai_for_sample(sample_id, models):
    """计算样本的FAI值"""
    print(f"\n🔬 处理样本 {sample_id}...")
    
    # 加载数据
    vin1_data, vin2_data, vin3_data = load_sample_data(sample_id)
    
    # 数据预处理
    if len(vin1_data.shape) == 2:
        vin1_data = vin1_data.unsqueeze(1)
    vin1_data = vin1_data.to(torch.float32).to(device)

    # 定义维度
    dim_x, dim_y, dim_z, dim_q = 2, 110, 110, 3
    dim_x2, dim_y2, dim_z2, dim_q2 = 2, 110, 110, 4
    
    # 分离数据
    x_recovered = vin2_data[:, :dim_x]
    y_recovered = vin2_data[:, dim_x:dim_x + dim_y]
    z_recovered = vin2_data[:, dim_x + dim_y: dim_x + dim_y + dim_z]
    q_recovered = vin2_data[:, dim_x + dim_y + dim_z:]
    
    x_recovered2 = vin3_data[:, :dim_x2]
    y_recovered2 = vin3_data[:, dim_x2:dim_x2 + dim_y2]
    z_recovered2 = vin3_data[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
    q_recovered2 = vin3_data[:, dim_x2 + dim_y2 + dim_z2:]
    
    # MC-AE推理
    models['net'].eval()
    models['netx'].eval()
    
    with torch.no_grad():
        x_recovered = x_recovered.float()
        z_recovered = z_recovered.float()
        q_recovered = q_recovered.float()
        x_recovered2 = x_recovered2.float()
        z_recovered2 = z_recovered2.float()
        q_recovered2 = q_recovered2.float()
        
        recon_imtest = models['net'](x_recovered, z_recovered, q_recovered)
        reconx_imtest = models['netx'](x_recovered2, z_recovered2, q_recovered2)
    
    # 计算重构误差
    AA = recon_imtest[0].cpu().detach().numpy()
    yTrainU = y_recovered.cpu().detach().numpy()
    ERRORU = AA - yTrainU

    BB = reconx_imtest[0].cpu().detach().numpy()
    yTrainX = y_recovered2.cpu().detach().numpy()
    ERRORX = BB - yTrainX

    # 诊断特征提取
    df_data = DiagnosisFeature(ERRORU, ERRORX)
    
    # 综合计算获取FAI
    pca_params = models['pca_params']
    time = np.arange(df_data.shape[0])
    
    result = Comprehensive_calculation(
        df_data.values, 
        pca_params['data_mean'], 
        pca_params['data_std'], 
        pca_params['v'].reshape(len(pca_params['v']), 1), 
        pca_params['p_k'], 
        pca_params['v_I'], 
        pca_params['T_99_limit'], 
        pca_params['SPE_99_limit'], 
        pca_params['X'], 
        time
    )
    
    fai = result[9]  # FAI是第10个返回值
    
    print(f"   数据长度: {len(fai)} 个时间点")
    print(f"   FAI统计: 均值={np.mean(fai):.6f}, 标准差={np.std(fai):.6f}")
    print(f"   FAI范围: [{np.min(fai):.6f}, {np.max(fai):.6f}]")
    
    return fai

def analyze_thresholds(fai, sample_id, sample_type):
    """分析阈值情况"""
    print(f"\n📊 样本{sample_id} ({sample_type}) 阈值分析:")
    
    # 按照原版方式计算阈值
    nm = 3000
    mm = len(fai)
    
    if mm > nm:
        # 使用后半段数据计算阈值
        baseline_fai = fai[nm:mm]
        mean_fai = np.mean(baseline_fai)
        std_fai = np.std(baseline_fai)
        print(f"   使用后半段数据({nm}:{mm})计算阈值")
        print(f"   基线统计: 均值={mean_fai:.6f}, 标准差={std_fai:.6f}")
    else:
        # 数据太短，使用全部数据
        mean_fai = np.mean(fai)
        std_fai = np.std(fai)
        print(f"   数据长度不足{nm}，使用全部数据计算阈值")
        print(f"   全量统计: 均值={mean_fai:.6f}, 标准差={std_fai:.6f}")
    
    # 计算各级阈值
    threshold_configs = {
        '3σ (原版)': {'multipliers': [3, 4.5, 6]},
        '2.5σ (宽松)': {'multipliers': [2.5, 3.5, 4.5]}, 
        '2σ (很宽松)': {'multipliers': [2, 3, 4]},
        '1.5σ (极宽松)': {'multipliers': [1.5, 2.5, 3.5]}
    }
    
    print(f"\n   各阈值配置下的检测结果:")
    print(f"   {'配置':<12} {'T1':<10} {'T2':<10} {'T3':<10} {'>T1':<8} {'>T2':<8} {'>T3':<8} {'>T1%':<8}")
    print(f"   {'-'*80}")
    
    best_config = None
    target_ratio = 0.05 if sample_type == "故障" else 0.01  # 故障样本期望5%异常率，正常样本1%
    
    for config_name, config in threshold_configs.items():
        t1 = mean_fai + config['multipliers'][0] * std_fai
        t2 = mean_fai + config['multipliers'][1] * std_fai
        t3 = mean_fai + config['multipliers'][2] * std_fai
        
        above_t1 = np.sum(fai > t1)
        above_t2 = np.sum(fai > t2)
        above_t3 = np.sum(fai > t3)
        ratio_t1 = above_t1 / len(fai)
        
        print(f"   {config_name:<12} {t1:<10.4f} {t2:<10.4f} {t3:<10.4f} {above_t1:<8} {above_t2:<8} {above_t3:<8} {ratio_t1*100:<7.2f}%")
        
        # 为故障样本选择合适的配置
        if sample_type == "故障" and best_config is None and above_t1 > 0:
            best_config = config_name
    
    if sample_type == "故障":
        if best_config:
            print(f"\n   💡 推荐配置: {best_config} (首个有检测结果的配置)")
        else:
            print(f"\n   ⚠️  所有配置都无法检测到异常点！数据可能存在问题")
    
    return threshold_configs

def analyze_continuity(fai, sample_id, sample_type):
    """分析异常点的连续性"""
    print(f"\n🔗 样本{sample_id} ({sample_type}) 连续性分析:")
    
    # 使用2σ阈值进行连续性分析（相对宽松）
    nm = 3000
    mm = len(fai)
    
    if mm > nm:
        baseline_fai = fai[nm:mm]
        mean_fai = np.mean(baseline_fai)
        std_fai = np.std(baseline_fai)
    else:
        mean_fai = np.mean(fai)
        std_fai = np.std(fai)
    
    threshold = mean_fai + 2 * std_fai  # 使用2σ阈值
    above_indices = np.where(fai > threshold)[0]
    
    if len(above_indices) == 0:
        print(f"   没有点超过2σ阈值({threshold:.6f})")
        return
    
    print(f"   使用2σ阈值({threshold:.6f})进行连续性分析")
    print(f"   超过阈值的点数: {len(above_indices)} ({len(above_indices)/len(fai)*100:.2f}%)")
    
    # 分析连续段
    continuous_segments = []
    current_segment = [above_indices[0]]
    
    for i in range(1, len(above_indices)):
        gap = above_indices[i] - above_indices[i-1]
        if gap <= 5:  # 允许4个点的间隔
            current_segment.append(above_indices[i])
        else:
            if len(current_segment) >= 3:  # 至少3个点才算连续段
                continuous_segments.append(current_segment)
            current_segment = [above_indices[i]]
    
    # 添加最后一段
    if len(current_segment) >= 3:
        continuous_segments.append(current_segment)
    
    print(f"   连续异常段数: {len(continuous_segments)}")
    
    if len(continuous_segments) > 0:
        print(f"   连续段详情:")
        for j, segment in enumerate(continuous_segments[:5]):  # 显示前5段
            start_pos = segment[0]
            end_pos = segment[-1]
            length = len(segment)
            max_fai = np.max(fai[segment])
            avg_fai = np.mean(fai[segment])
            print(f"     段{j+1}: 位置[{start_pos:4d}-{end_pos:4d}] 长度={length:3d} 峰值={max_fai:.6f} 均值={avg_fai:.6f}")
        
        # 分析间隔
        if len(continuous_segments) > 1:
            gaps = []
            for i in range(1, len(continuous_segments)):
                gap = continuous_segments[i][0] - continuous_segments[i-1][-1]
                gaps.append(gap)
            print(f"   段间间隔: 平均={np.mean(gaps):.1f}, 最小={np.min(gaps)}, 最大={np.max(gaps)}")
    else:
        print(f"   没有发现连续异常段（长度>=3）")
        # 显示孤立异常点
        print(f"   孤立异常点分布:")
        for i in range(min(10, len(above_indices))):
            idx = above_indices[i]
            print(f"     位置{idx:4d}: FAI={fai[idx]:.6f} (超出阈值: {fai[idx]-threshold:.6f})")

def main():
    """主函数"""
    print("="*80)
    print("🔬 FAI阈值和连续性分析工具")
    print("="*80)
    
    # 测试样本配置
    test_samples = {
        '10': '正常',   # 正常样本
        '335': '故障'   # 故障样本
    }
    
    # 加载模型
    print("🔧 加载模型和参数...")
    models = load_models()
    print("✅ 模型加载完成")
    
    # 分析每个样本
    for sample_id, sample_type in test_samples.items():
        print(f"\n{'='*60}")
        print(f"分析样本 {sample_id} ({sample_type})")
        print(f"{'='*60}")
        
        try:
            # 计算FAI
            fai = compute_fai_for_sample(sample_id, models)
            
            # 阈值分析
            analyze_thresholds(fai, sample_id, sample_type)
            
            # 连续性分析
            analyze_continuity(fai, sample_id, sample_type)
            
        except Exception as e:
            print(f"❌ 处理样本{sample_id}失败: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("🎉 分析完成！")
    print("💡 根据分析结果调整阈值配置:")
    print("   1. 如果故障样本没有异常点，降低阈值倍数")
    print("   2. 如果正常样本异常点过多，提高阈值倍数") 
    print("   3. 观察连续性，设计合适的5点检测策略")
    print("="*80)

if __name__ == "__main__":
    main()