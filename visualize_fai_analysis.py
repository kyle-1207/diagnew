# FAI可视化分析脚本
# 生成FAI时序图和阈值分析图

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os
import warnings
from Function_ import *
from Class_ import *
from Comprehensive_calculation import Comprehensive_calculation

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
    return fai

def plot_fai_analysis(fai_normal, fai_fault, sample_id_normal, sample_id_fault):
    """绘制FAI分析图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 计算阈值
    def calc_thresholds(fai, sigma_mult):
        nm = 3000
        mm = len(fai)
        if mm > nm:
            baseline = fai[nm:mm]
        else:
            baseline = fai
        mean_val = np.mean(baseline)
        std_val = np.std(baseline)
        return mean_val + sigma_mult * std_val
    
    # 子图1: 正常样本FAI时序
    ax1 = axes[0, 0]
    time_normal = np.arange(len(fai_normal))
    ax1.plot(time_normal, fai_normal, 'b-', linewidth=1, alpha=0.8, label=f'样本{sample_id_normal} (正常)')
    
    # 添加不同阈值线
    thresholds_normal = {
        '3σ': calc_thresholds(fai_normal, 3),
        '2σ': calc_thresholds(fai_normal, 2),
        '1.5σ': calc_thresholds(fai_normal, 1.5)
    }
    
    colors = ['red', 'orange', 'green']
    for i, (name, thresh) in enumerate(thresholds_normal.items()):
        ax1.axhline(y=thresh, color=colors[i], linestyle='--', alpha=0.7, label=f'{name}阈值')
        above_count = np.sum(fai_normal > thresh)
        ax1.text(0.02, 0.98-i*0.05, f'{name}: {above_count}点 ({above_count/len(fai_normal)*100:.2f}%)', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    
    ax1.set_title(f'正常样本{sample_id_normal} - FAI时序分析')
    ax1.set_xlabel('时间步长')
    ax1.set_ylabel('FAI值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 故障样本FAI时序
    ax2 = axes[0, 1]
    time_fault = np.arange(len(fai_fault))
    ax2.plot(time_fault, fai_fault, 'r-', linewidth=1, alpha=0.8, label=f'样本{sample_id_fault} (故障)')
    
    # 添加不同阈值线
    thresholds_fault = {
        '3σ': calc_thresholds(fai_fault, 3),
        '2σ': calc_thresholds(fai_fault, 2),
        '1.5σ': calc_thresholds(fai_fault, 1.5)
    }
    
    for i, (name, thresh) in enumerate(thresholds_fault.items()):
        ax2.axhline(y=thresh, color=colors[i], linestyle='--', alpha=0.7, label=f'{name}阈值')
        above_count = np.sum(fai_fault > thresh)
        ax2.text(0.02, 0.98-i*0.05, f'{name}: {above_count}点 ({above_count/len(fai_fault)*100:.2f}%)', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    
    ax2.set_title(f'故障样本{sample_id_fault} - FAI时序分析')
    ax2.set_xlabel('时间步长')
    ax2.set_ylabel('FAI值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: FAI分布直方图对比
    ax3 = axes[1, 0]
    ax3.hist(fai_normal, bins=50, alpha=0.6, color='blue', label=f'正常样本{sample_id_normal}', density=True)
    ax3.hist(fai_fault, bins=50, alpha=0.6, color='red', label=f'故障样本{sample_id_fault}', density=True)
    
    # 添加统计信息
    ax3.axvline(np.mean(fai_normal), color='blue', linestyle='-', alpha=0.8, label=f'正常均值: {np.mean(fai_normal):.4f}')
    ax3.axvline(np.mean(fai_fault), color='red', linestyle='-', alpha=0.8, label=f'故障均值: {np.mean(fai_fault):.4f}')
    
    ax3.set_title('FAI值分布对比')
    ax3.set_xlabel('FAI值')
    ax3.set_ylabel('密度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 阈值敏感性分析
    ax4 = axes[1, 1]
    
    sigma_range = np.arange(1.0, 4.1, 0.1)
    ratios_normal = []
    ratios_fault = []
    
    for sigma in sigma_range:
        thresh_normal = calc_thresholds(fai_normal, sigma)
        thresh_fault = calc_thresholds(fai_fault, sigma)
        
        ratio_normal = np.sum(fai_normal > thresh_normal) / len(fai_normal)
        ratio_fault = np.sum(fai_fault > thresh_fault) / len(fai_fault)
        
        ratios_normal.append(ratio_normal)
        ratios_fault.append(ratio_fault)
    
    ax4.plot(sigma_range, ratios_normal, 'b-', linewidth=2, label=f'正常样本{sample_id_normal}')
    ax4.plot(sigma_range, ratios_fault, 'r-', linewidth=2, label=f'故障样本{sample_id_fault}')
    
    # 标记关键点
    ax4.axvline(3.0, color='gray', linestyle='--', alpha=0.7, label='原版3σ')
    ax4.axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='宽松2σ')
    ax4.axhline(0.05, color='green', linestyle=':', alpha=0.7, label='目标5%异常率')
    
    ax4.set_title('阈值敏感性分析')
    ax4.set_xlabel('阈值倍数 (σ)')
    ax4.set_ylabel('超过阈值的比例')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 0.2)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = "Linux/fai_analysis_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 分析图保存至: {save_path}")
    plt.show()

def main():
    """主函数"""
    print("="*80)
    print("🎨 FAI可视化分析工具")
    print("="*80)
    
    # 加载模型
    print("🔧 加载模型和参数...")
    models = load_models()
    print("✅ 模型加载完成")
    
    # 计算FAI
    print("\n🔬 计算样本FAI值...")
    fai_normal = compute_fai_for_sample('10', models)
    fai_fault = compute_fai_for_sample('335', models)
    
    print(f"   正常样本10: 长度={len(fai_normal)}, 均值={np.mean(fai_normal):.6f}, 标准差={np.std(fai_normal):.6f}")
    print(f"   故障样本335: 长度={len(fai_fault)}, 均值={np.mean(fai_fault):.6f}, 标准差={np.std(fai_fault):.6f}")
    
    # 生成可视化
    print("\n🎨 生成可视化分析...")
    plot_fai_analysis(fai_normal, fai_fault, '10', '335')
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()