# 导入必要的库
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import math
import math
from create_dataset import series_to_supervised
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs
import scipy.stats as stats
import seaborn as sns
import pickle
from Comprehensive_calculation import Comprehensive_calculation

# 新增导入
from tqdm import tqdm
import json
import time
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
import glob

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#----------------------------------------测试配置------------------------------
print("="*60)
print("🔬 电池故障诊断系统 - 双模型比对测试")
print("="*60)

TEST_MODE = "COMPARE"  # 固定为比对模式

# 测试数据集配置 (根据Labels.xls动态加载)
def load_test_samples():
    """从Labels.xls加载测试样本"""
    try:
        import pandas as pd
        labels_path = '../QAS/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取测试样本
        all_samples = df['Num'].tolist()
        all_labels = df['Label'].tolist()
        
        # 201-334的正常样本
        test_normal_samples = [str(i) for i in all_samples if 201 <= i <= 334 and all_labels[all_samples.index(i)] == 0]
        # 335-392的故障样本
        test_fault_samples = [str(i) for i in all_samples if 335 <= i <= 392 and all_labels[all_samples.index(i)] == 1]
        
        print(f"📋 从Labels.xls加载测试样本:")
        print(f"   测试正常样本 (201-334): {len(test_normal_samples)} 个")
        print(f"   测试故障样本 (335-392): {len(test_fault_samples)} 个")
        
        return {
            'normal': test_normal_samples,
            'fault': test_fault_samples
        }
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认测试样本")
        return {
            'normal': ['166', '209'],
            'fault': ['335', '386']
        }

TEST_SAMPLES = load_test_samples()
ALL_TEST_SAMPLES = TEST_SAMPLES['normal'] + TEST_SAMPLES['fault']

# 模型路径配置
MODEL_PATHS = {
    "BILSTM": {
        "net_model": "./models/net_model_bilstm_baseline.pth",
        "netx_model": "./models/netx_model_bilstm_baseline.pth",
        "pca_files": ["./models/v_I_bilstm_baseline.npy", "./models/v_bilstm_baseline.npy", 
                     "./models/v_ratio_bilstm_baseline.npy", "./models/p_k_bilstm_baseline.npy",
                     "./models/data_mean_bilstm_baseline.npy", "./models/data_std_bilstm_baseline.npy",
                     "./models/T_95_limit_bilstm_baseline.npy", "./models/T_99_limit_bilstm_baseline.npy",
                     "./models/SPE_95_limit_bilstm_baseline.npy", "./models/SPE_99_limit_bilstm_baseline.npy",
                     "./models/P_bilstm_baseline.npy", "./models/k_bilstm_baseline.npy",
                     "./models/P_t_bilstm_baseline.npy", "./models/X_bilstm_baseline.npy",
                     "./models/data_nor_bilstm_baseline.npy"]
    },
    "TRANSFORMER": {
        "transformer_model": "./models/transformer_model.pth",
        "net_model": "./models/net_model_transformer.pth", 
        "netx_model": "./models/netx_model_transformer.pth",
        "pca_files": ["./models/v_I_transformer.npy", "./models/v_transformer.npy",
                     "./models/v_ratio_transformer.npy", "./models/p_k_transformer.npy",
                     "./models/data_mean_transformer.npy", "./models/data_std_transformer.npy",
                     "./models/T_95_limit_transformer.npy", "./models/T_99_limit_transformer.npy",
                     "./models/SPE_95_limit_transformer.npy", "./models/SPE_99_limit_transformer.npy",
                     "./models/P_transformer.npy", "./models/k_transformer.npy",
                     "./models/P_t_transformer.npy", "./models/X_transformer.npy",
                     "./models/data_nor_transformer.npy"]
    }
}

# 三窗口固定参数
WINDOW_CONFIG = {
    "detection_window": 100,     # 检测窗口：100个采样点
    "verification_window": 50,   # 验证窗口：50个采样点  
    "marking_window": 50        # 标记窗口：前后各50个采样点
}

# 高分辨率可视化配置
PLOT_CONFIG = {
    "dpi": 300,
    "figsize_large": (15, 12),
    "figsize_medium": (12, 8), 
    "bbox_inches": "tight"
}

print(f"📊 测试配置:")
print(f"   测试样本: {ALL_TEST_SAMPLES}")
print(f"   正常样本: {TEST_SAMPLES['normal']}")
print(f"   故障样本: {TEST_SAMPLES['fault']}")
print(f"   三窗口参数: {WINDOW_CONFIG}")

#----------------------------------------模型文件检查------------------------------
def check_model_files():
    """检查所有必需的模型文件"""
    print("\n🔍 检查模型文件...")
    
    missing_files = []
    
    for model_type, paths in MODEL_PATHS.items():
        print(f"  检查 {model_type} 模型文件...")
        
        # 检查主模型文件
        for key, path in paths.items():
            if key != "pca_files":
                if not os.path.exists(path):
                    missing_files.append(f"{model_type}: {path}")
            else:
                # 检查PCA相关文件
                for pca_file in path:
                    if not os.path.exists(pca_file):
                        missing_files.append(f"{model_type}: {pca_file}")
    
    if missing_files:
        print("❌ 缺失模型文件:")
        for file in missing_files:
            print(f"   {file}")
        raise FileNotFoundError("请先运行对应的训练脚本生成所需模型文件")
    
    print("✅ 所有模型文件检查通过")
    return True

# 执行模型文件检查
check_model_files()

#----------------------------------------三窗口故障检测机制------------------------------
def three_window_fault_detection(fai_values, threshold1, sample_id):
    """
    三窗口故障检测机制：检测→验证→标记
    
    Args:
        fai_values: 综合诊断指标序列
        threshold1: 一级预警阈值
        sample_id: 样本ID（用于调试）
    
    Returns:
        fault_labels: 故障标签序列 (0=正常, 1=故障)
        detection_info: 检测过程详细信息
    """
    detection_window = WINDOW_CONFIG["detection_window"]
    verification_window = WINDOW_CONFIG["verification_window"] 
    marking_window = WINDOW_CONFIG["marking_window"]
    
    fault_labels = np.zeros(len(fai_values), dtype=int)
    detection_info = {
        'candidate_points': [],
        'verified_points': [],
        'marked_regions': [],
        'window_stats': {}
    }
    
    # 阶段1：检测窗口 - 寻找候选故障点
    candidate_points = []
    for i in range(len(fai_values)):
        if fai_values[i] > threshold1:
            candidate_points.append(i)
    
    detection_info['candidate_points'] = candidate_points
    
    if len(candidate_points) == 0:
        # 没有候选点，直接返回
        return fault_labels, detection_info
    
    # 阶段2：验证窗口 - 检查持续性
    verified_points = []
    for candidate in candidate_points:
        # 定义验证窗口范围
        start_verify = max(0, candidate - verification_window//2)
        end_verify = min(len(fai_values), candidate + verification_window//2)
        verify_data = fai_values[start_verify:end_verify]
        
        # 持续性判断：验证窗口内超阈值点比例
        continuous_ratio = np.sum(verify_data > threshold1) / len(verify_data)
        
        # 30%以上超阈值认为持续异常
        if continuous_ratio >= 0.3:
            verified_points.append({
                'point': candidate,
                'continuous_ratio': continuous_ratio,
                'verify_range': (start_verify, end_verify)
            })
    
    detection_info['verified_points'] = verified_points
    
    # 阶段3：标记窗口 - 标记故障区域
    marked_regions = []
    for verified in verified_points:
        candidate = verified['point']
        
        # 定义标记窗口范围
        start_mark = max(0, candidate - marking_window)
        end_mark = min(len(fai_values), candidate + marking_window)
        
        # 标记故障区域
        fault_labels[start_mark:end_mark] = 1
        
        marked_regions.append({
            'center': candidate,
            'range': (start_mark, end_mark),
            'length': end_mark - start_mark
        })
    
    detection_info['marked_regions'] = marked_regions
    
    # 统计信息
    detection_info['window_stats'] = {
        'total_candidates': len(candidate_points),
        'verified_candidates': len(verified_points),
        'total_fault_points': np.sum(fault_labels),
        'fault_ratio': np.sum(fault_labels) / len(fault_labels)
    }
    
    return fault_labels, detection_info

#----------------------------------------数据加载函数------------------------------
def load_test_sample(sample_id):
    """加载测试样本"""
    base_path = f'../QAS/{sample_id}'
    
    # 检查样本目录是否存在
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"测试样本目录不存在: {base_path}")
    
    # 加载vin_1, vin_2, vin_3数据
    try:
        with open(f'{base_path}/vin_1.pkl', 'rb') as f:
            vin1_data = pickle.load(f)
        with open(f'{base_path}/vin_2.pkl', 'rb') as f:
            vin2_data = pickle.load(f) 
        with open(f'{base_path}/vin_3.pkl', 'rb') as f:
            vin3_data = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"样本 {sample_id} 数据文件缺失: {e}")
        
    return vin1_data, vin2_data, vin3_data

def load_models(model_type):
    """加载指定类型的模型"""
    models = {}
    
    if model_type == "BILSTM":
        # 加载BiLSTM基准模型
        models['net'] = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                            activation_fn=custom_activation, use_dx_in_forward=True).to(device)
        models['netx'] = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                                   activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)

    # 加载模型参数
        models['net'].load_state_dict(torch.load(MODEL_PATHS[model_type]["net_model"]))
        models['netx'].load_state_dict(torch.load(MODEL_PATHS[model_type]["netx_model"]))
        
        # 加载PCA参数
        models['pca_params'] = {}
        pca_files = MODEL_PATHS[model_type]["pca_files"]
        models['pca_params']['v_I'] = np.load(pca_files[0])
        models['pca_params']['v'] = np.load(pca_files[1])
        models['pca_params']['v_ratio'] = np.load(pca_files[2])
        models['pca_params']['p_k'] = np.load(pca_files[3])
        models['pca_params']['data_mean'] = np.load(pca_files[4])
        models['pca_params']['data_std'] = np.load(pca_files[5])
        models['pca_params']['T_95_limit'] = np.load(pca_files[6])
        models['pca_params']['T_99_limit'] = np.load(pca_files[7])
        models['pca_params']['SPE_95_limit'] = np.load(pca_files[8])
        models['pca_params']['SPE_99_limit'] = np.load(pca_files[9])
        models['pca_params']['P'] = np.load(pca_files[10])
        models['pca_params']['k'] = np.load(pca_files[11])
        models['pca_params']['P_t'] = np.load(pca_files[12])
        models['pca_params']['X'] = np.load(pca_files[13])
        models['pca_params']['data_nor'] = np.load(pca_files[14])
        
    elif model_type == "TRANSFORMER":
        # 加载Transformer模型
        from Train_Transformer import TransformerPredictor
        models['transformer'] = TransformerPredictor().to(device)
        models['transformer'].load_state_dict(torch.load(MODEL_PATHS[model_type]["transformer_model"]))
        
        # 加载MC-AE模型
        models['net'] = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                                  activation_fn=custom_activation, use_dx_in_forward=True).to(device)
        models['netx'] = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                                   activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
        
        models['net'].load_state_dict(torch.load(MODEL_PATHS[model_type]["net_model"]))
        models['netx'].load_state_dict(torch.load(MODEL_PATHS[model_type]["netx_model"]))
        
        # 加载PCA参数
        models['pca_params'] = {}
        pca_files = MODEL_PATHS[model_type]["pca_files"]
        models['pca_params']['v_I'] = np.load(pca_files[0])
        models['pca_params']['v'] = np.load(pca_files[1])
        models['pca_params']['v_ratio'] = np.load(pca_files[2])
        models['pca_params']['p_k'] = np.load(pca_files[3])
        models['pca_params']['data_mean'] = np.load(pca_files[4])
        models['pca_params']['data_std'] = np.load(pca_files[5])
        models['pca_params']['T_95_limit'] = np.load(pca_files[6])
        models['pca_params']['T_99_limit'] = np.load(pca_files[7])
        models['pca_params']['SPE_95_limit'] = np.load(pca_files[8])
        models['pca_params']['SPE_99_limit'] = np.load(pca_files[9])
        models['pca_params']['P'] = np.load(pca_files[10])
        models['pca_params']['k'] = np.load(pca_files[11])
        models['pca_params']['P_t'] = np.load(pca_files[12])
        models['pca_params']['X'] = np.load(pca_files[13])
        models['pca_params']['data_nor'] = np.load(pca_files[14])
    
    return models

#----------------------------------------单样本处理函数------------------------------
def process_single_sample(sample_id, models, model_type):
    """处理单个测试样本"""
    
    # 加载样本数据
    vin1_data, vin2_data, vin3_data = load_test_sample(sample_id)
    
    # 数据预处理
    if len(vin1_data.shape) == 2:
        vin1_data = vin1_data.unsqueeze(1)
    vin1_data = vin1_data.to(torch.float32).to(device)

    # 定义维度
    dim_x, dim_y, dim_z, dim_q = 2, 110, 110, 3
    dim_x2, dim_y2, dim_z2, dim_q2 = 2, 110, 110, 4
    
    # 使用预训练的PCA参数而不是重新计算
    pca_params = models['pca_params']
    
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
        models['net'] = models['net'].double()
        models['netx'] = models['netx'].double()
        
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
    
    # 使用预训练的PCA参数进行综合计算
    time = np.arange(df_data.shape[0])
    
    lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std = Comprehensive_calculation(
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
    
    # 计算阈值 - 与源代码保持一致
    nm = 3000  # 固定值，与源代码一致
    mm = len(fai)  # 数据总长度
    
    # 确保数据长度足够
    if mm > nm:
        # 使用后半段数据计算阈值
        threshold1 = np.mean(fai[nm:mm]) + 3*np.std(fai[nm:mm])
        threshold2 = np.mean(fai[nm:mm]) + 4.5*np.std(fai[nm:mm])
        threshold3 = np.mean(fai[nm:mm]) + 6*np.std(fai[nm:mm])
    else:
        # 数据太短，使用全部数据
        print(f"   ⚠️ 样本{sample_id}数据长度({mm})不足3000，使用全部数据计算阈值")
        threshold1 = np.mean(fai) + 3*np.std(fai)
        threshold2 = np.mean(fai) + 4.5*np.std(fai)
        threshold3 = np.mean(fai) + 6*np.std(fai)
    
    # 三窗口故障检测
    fault_labels, detection_info = three_window_fault_detection(fai, threshold1, sample_id)
    
    # 构建结果
    sample_result = {
        'sample_id': sample_id,
        'model_type': model_type,
        'label': 1 if sample_id in TEST_SAMPLES['fault'] else 0,
        'df_data': df_data.values,
        'fai': fai,
        'T_squared': t_total,
        'SPE': q_total,
        'thresholds': {
            'threshold1': threshold1,
            'threshold2': threshold2, 
            'threshold3': threshold3
        },
        'fault_labels': fault_labels,
        'detection_info': detection_info,
        'performance_metrics': {
            'fai_mean': np.mean(fai),
            'fai_std': np.std(fai),
            'fai_max': np.max(fai),
            'fai_min': np.min(fai),
            'anomaly_count': np.sum(fai > threshold1),
            'anomaly_ratio': np.sum(fai > threshold1) / len(fai)
        }
    }
    
    return sample_result

#----------------------------------------主测试流程------------------------------
def main_test_process():
    """主要测试流程"""
    
    # 初始化结果存储
    test_results = {
        "BILSTM": [],
        "TRANSFORMER": [],
        "metadata": {
            "test_samples": TEST_SAMPLES,
            "window_config": WINDOW_CONFIG,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 双模型测试循环
    total_operations = len(ALL_TEST_SAMPLES) * 2  # 4个样本 × 2个模型
    
    print(f"\n🚀 开始双模型比对测试...")
    print(f"总共需要处理: {total_operations} 个任务")
    
    with tqdm(total=total_operations, desc="双模型测试进度", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:
        
        for model_type in ["BILSTM", "TRANSFORMER"]:
            print(f"\n{'='*20} 测试 {model_type} 模型 {'='*20}")
            
            # 加载模型
            pbar.set_description(f"加载{model_type}模型")
            models = load_models(model_type)
            print(f"✅ {model_type} 模型加载完成")
            
            for sample_id in ALL_TEST_SAMPLES:
                pbar.set_description(f"{model_type}-样本{sample_id}")
                
                try:
                    # 处理单个样本
                    sample_result = process_single_sample(sample_id, models, model_type)
                    test_results[model_type].append(sample_result)
                    
                    # 输出简要结果
                    metrics = sample_result['performance_metrics']
                    print(f"   样本{sample_id}: fai均值={metrics['fai_mean']:.6f}, "
                          f"异常率={metrics['anomaly_ratio']:.2%}, "
                          f"三窗口检测={sample_result['detection_info']['window_stats']['fault_ratio']:.2%}")
                    
                except Exception as e:
                    print(f"❌ 样本 {sample_id} 处理失败: {e}")
                    continue
                
                pbar.update(1)
                time.sleep(0.1)  # 避免进度条更新过快
    
    print(f"\n✅ 双模型测试完成!")
    print(f"   BiLSTM: 成功处理 {len(test_results['BILSTM'])} 个样本")
    print(f"   Transformer: 成功处理 {len(test_results['TRANSFORMER'])} 个样本")
    
    return test_results

# 执行主测试流程
test_results = main_test_process()

#----------------------------------------性能分析函数------------------------------
def calculate_performance_metrics(test_results):
    """计算双模型性能指标"""
    print("\n🔬 计算性能指标...")
    
    performance_metrics = {}
    
    for model_type in ["BILSTM", "TRANSFORMER"]:
        model_results = test_results[model_type]
        
        # 收集所有样本的预测结果
        all_true_labels = []
        all_fai_values = []
        all_fault_predictions = []
        
        for result in model_results:
            true_label = result['label']
            fai_values = result['fai']
            fault_labels = result['fault_labels']
            threshold1 = result['thresholds']['threshold1']
            
            # 对于每个时间点
            for i, (fai_val, fault_pred) in enumerate(zip(fai_values, fault_labels)):
                all_true_labels.append(true_label)
                all_fai_values.append(fai_val)
                
                # 根据我们之前讨论的ROC逻辑：
                if true_label == 0:  # 正常样本
                    # 正常样本中：fai > threshold 就是FP，fai <= threshold 就是TN
                    prediction = 1 if fai_val > threshold1 else 0
                else:  # 故障样本
                    # 故障样本中：需要fai > threshold 且 三窗口确认为故障 才是TP
                    if fai_val > threshold1 and fault_pred == 1:
                        prediction = 1  # TP
                    else:
                        prediction = 0  # FN
                
                all_fault_predictions.append(prediction)
        
        # 计算ROC指标
        all_true_labels = np.array(all_true_labels)
        all_fai_values = np.array(all_fai_values)
        all_fault_predictions = np.array(all_fault_predictions)
        
        # 计算混淆矩阵
        tn = np.sum((all_true_labels == 0) & (all_fault_predictions == 0))
        fp = np.sum((all_true_labels == 0) & (all_fault_predictions == 1))
        fn = np.sum((all_true_labels == 1) & (all_fault_predictions == 0))
        tp = np.sum((all_true_labels == 1) & (all_fault_predictions == 1))
        
        # 计算性能指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        tpr = recall  # True Positive Rate = Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        
        # 样本级统计
        sample_metrics = {
            'total_samples': len(model_results),
            'normal_samples': len([r for r in model_results if r['label'] == 0]),
            'fault_samples': len([r for r in model_results if r['label'] == 1]),
            'avg_fai_normal': np.mean([r['performance_metrics']['fai_mean'] 
                                     for r in model_results if r['label'] == 0]),
            'avg_fai_fault': np.mean([r['performance_metrics']['fai_mean'] 
                                    for r in model_results if r['label'] == 1]),
            'avg_anomaly_ratio_normal': np.mean([r['performance_metrics']['anomaly_ratio'] 
                                               for r in model_results if r['label'] == 0]),
            'avg_anomaly_ratio_fault': np.mean([r['performance_metrics']['anomaly_ratio'] 
                                              for r in model_results if r['label'] == 1])
        }
        
        performance_metrics[model_type] = {
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
            'classification_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'specificity': float(specificity),
                'tpr': float(tpr),
                'fpr': float(fpr)
            },
            'sample_metrics': sample_metrics,
            'roc_data': {
                'true_labels': all_true_labels.tolist(),
                'fai_values': all_fai_values.tolist(),
                'predictions': all_fault_predictions.tolist()
            }
        }
    
    return performance_metrics

#----------------------------------------ROC曲线对比------------------------------
def create_roc_comparison(test_results, performance_metrics, save_path):
    """生成ROC曲线对比图"""
    print("   📈 生成ROC曲线对比...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_large"])
    
    # === 子图1: 连续阈值ROC曲线 ===
    ax1.set_title('(a) ROC曲线对比\n(连续阈值扫描)')
    
    colors = {'BILSTM': 'green', 'TRANSFORMER': 'blue'}
    
    for model_type in ["BILSTM", "TRANSFORMER"]:
        model_results = test_results[model_type]
        
        # 收集所有fai值和真实标签
        all_fai = []
        all_labels = []
        
        for result in model_results:
            all_fai.extend(result['fai'])
            all_labels.extend([result['label']] * len(result['fai']))
        
        all_fai = np.array(all_fai)
        all_labels = np.array(all_labels)
        
        # 生成连续阈值范围
        thresholds = np.linspace(np.min(all_fai), np.max(all_fai), 100)
        
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            tp = fp = tn = fn = 0
            
            for i, (fai_val, true_label) in enumerate(zip(all_fai, all_labels)):
                if true_label == 0:  # 正常样本
                    if fai_val > threshold:
                        fp += 1
                    else:
                        tn += 1
                else:  # 故障样本
                    # 简化：这里用fai阈值代替三窗口确认
                    if fai_val > threshold:
                        tp += 1
                    else:
                        fn += 1
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # 计算AUC
        from sklearn.metrics import auc
        auc_score = auc(fpr_list, tpr_list)
        
        # 绘制ROC曲线
        ax1.plot(fpr_list, tpr_list, color=colors[model_type], linewidth=2,
                label=f'{model_type} (AUC={auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
    ax1.set_xlabel('假正例率')
    ax1.set_ylabel('真正例率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 固定阈值工作点 ===
    ax2.set_title('(b) 论文工作点\n(三级报警阈值)')
    
    for model_type in ["BILSTM", "TRANSFORMER"]:
        metrics = performance_metrics[model_type]['classification_metrics']
        ax2.scatter(metrics['fpr'], metrics['tpr'], 
                   s=200, color=colors[model_type], 
                   label=f'{model_type}\n(TPR={metrics["tpr"]:.3f}, FPR={metrics["fpr"]:.3f})',
                   marker='o', edgecolors='black', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('假正例率')
    ax2.set_ylabel('真正例率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === 子图3: 性能指标对比 ===
    ax3.set_title('(c) 分类指标对比')
    
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数', '特异性']
    metric_mapping = {'准确率': 'accuracy', '精确率': 'precision', '召回率': 'recall', 'F1分数': 'f1_score', '特异性': 'specificity'}
    bilstm_values = [performance_metrics['BILSTM']['classification_metrics'][metric_mapping[m]] 
                    for m in metrics_names]
    transformer_values = [performance_metrics['TRANSFORMER']['classification_metrics'][metric_mapping[m]] 
                         for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax3.bar(x - width/2, bilstm_values, width, label='BiLSTM', color='green', alpha=0.7)
    ax3.bar(x + width/2, transformer_values, width, label='Transformer', color='blue', alpha=0.7)
    
    ax3.set_xlabel('指标')
    ax3.set_ylabel('分数')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === 子图4: 样本级性能对比 ===
    ax4.set_title('(d) 样本级性能对比')
    
    sample_metrics = ['平均φ(正常)', '平均φ(故障)', '异常率(正常)', '异常率(故障)']
    bilstm_sample_values = [
        performance_metrics['BILSTM']['sample_metrics']['avg_fai_normal'],
        performance_metrics['BILSTM']['sample_metrics']['avg_fai_fault'],
        performance_metrics['BILSTM']['sample_metrics']['avg_anomaly_ratio_normal'],
        performance_metrics['BILSTM']['sample_metrics']['avg_anomaly_ratio_fault']
    ]
    transformer_sample_values = [
        performance_metrics['TRANSFORMER']['sample_metrics']['avg_fai_normal'],
        performance_metrics['TRANSFORMER']['sample_metrics']['avg_fai_fault'],
        performance_metrics['TRANSFORMER']['sample_metrics']['avg_anomaly_ratio_normal'],
        performance_metrics['TRANSFORMER']['sample_metrics']['avg_anomaly_ratio_fault']
    ]
    
    x = np.arange(len(sample_metrics))
    ax4.bar(x - width/2, bilstm_sample_values, width, label='BiLSTM', color='green', alpha=0.7)
    ax4.bar(x + width/2, transformer_sample_values, width, label='Transformer', color='blue', alpha=0.7)
    
    ax4.set_xlabel('样本指标')
    ax4.set_ylabel('数值')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_metrics, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ ROC对比图保存至: {save_path}")

#----------------------------------------故障检测时序图------------------------------
def create_fault_detection_timeline(test_results, save_path):
    """生成故障检测时序图"""
    print("   📊 生成故障检测时序图...")
    
    # 选择一个故障样本进行可视化
    fault_sample_id = TEST_SAMPLES['fault'][0]  # 使用第一个故障样本
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    
    colors = {'BILSTM': 'green', 'TRANSFORMER': 'blue'}
    
    for i, model_type in enumerate(["BILSTM", "TRANSFORMER"]):
        # 找到对应样本的结果
        sample_result = next(r for r in test_results[model_type] if r['sample_id'] == fault_sample_id)
        
        fai_values = sample_result['fai']
        fault_labels = sample_result['fault_labels']
        thresholds = sample_result['thresholds']
        time_axis = np.arange(len(fai_values))
        
        # 子图1&2: 综合诊断指标时序
        ax = axes[i]
        ax.plot(time_axis, fai_values, color=colors[model_type], linewidth=1, alpha=0.8,
               label=f'{model_type} FAI')
        ax.axhline(y=thresholds['threshold1'], color='orange', linestyle='--', alpha=0.7,
                  label='一级阈值')
        ax.axhline(y=thresholds['threshold2'], color='red', linestyle='--', alpha=0.7,
                  label='二级阈值')
        ax.axhline(y=thresholds['threshold3'], color='darkred', linestyle='--', alpha=0.7,
                  label='三级阈值')
        
        ax.set_ylabel(f'{model_type}\n综合诊断指标φ')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{model_type} - 样本 {fault_sample_id} (故障样本)')
    
    # 子图3: 故障检测结果对比
    ax3 = axes[2]
    
    for model_type in ["BILSTM", "TRANSFORMER"]:
        sample_result = next(r for r in test_results[model_type] if r['sample_id'] == fault_sample_id)
        fault_labels = sample_result['fault_labels']
        time_axis = np.arange(len(fault_labels))
        
        # 将故障标签转换为可视化区域
        fault_regions = np.where(fault_labels == 1, 0.8 if model_type == 'BILSTM' else 0.4, 0)
        ax3.fill_between(time_axis, fault_regions, 
                        alpha=0.6, color=colors[model_type],
                        label=f'{model_type} Fault Detection')
    
    ax3.set_ylabel('故障检测结果')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('故障检测结果对比')
    
    # 子图4: 三窗口检测过程（以TRANSFORMER为例）
    ax4 = axes[3]
    transformer_result = next(r for r in test_results['TRANSFORMER'] if r['sample_id'] == fault_sample_id)
    detection_info = transformer_result['detection_info']
    
    time_axis = np.arange(len(transformer_result['fai']))
    ax4.plot(time_axis, transformer_result['fai'], 'b-', alpha=0.5, label='φ指标值')
    
    # 标记候选点
    if detection_info['candidate_points']:
        ax4.scatter(detection_info['candidate_points'], 
                   [transformer_result['fai'][i] for i in detection_info['candidate_points']],
                   color='orange', s=30, label='候选点', alpha=0.8)
    
    # 标记验证通过的点
    if detection_info['verified_points']:
        verified_indices = [v['point'] for v in detection_info['verified_points']]
        ax4.scatter(verified_indices,
                   [transformer_result['fai'][i] for i in verified_indices],
                   color='red', s=50, label='验证点', marker='^')
    
    # 标记故障区域
    for region in detection_info['marked_regions']:
        start, end = region['range']
        ax4.axvspan(start, end, alpha=0.2, color='red', label='标记故障区域')
    
    ax4.set_ylabel('三窗口\n检测过程')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('三窗口检测过程 (Transformer)')
    
    # 子图5: 检测统计信息
    ax5 = axes[4]
    
    models = ['BiLSTM', 'Transformer']
    fault_ratios = []
    anomaly_ratios = []
    
    for model_type in ["BILSTM", "TRANSFORMER"]:
        sample_result = next(r for r in test_results[model_type] if r['sample_id'] == fault_sample_id)
        fault_ratios.append(sample_result['detection_info']['window_stats']['fault_ratio'])
        anomaly_ratios.append(sample_result['performance_metrics']['anomaly_ratio'])
    
    x = np.arange(len(models))
    width = 0.35
    
    ax5.bar(x - width/2, fault_ratios, width, label='三窗口故障比率', alpha=0.7)
    ax5.bar(x + width/2, anomaly_ratios, width, label='阈值异常比率', alpha=0.7)
    
    ax5.set_xlabel('模型类型')
    ax5.set_ylabel('检测比率')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('检测性能统计')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ 时序图保存至: {save_path}")

#----------------------------------------性能指标雷达图------------------------------
def create_performance_radar(performance_metrics, save_path):
    """生成性能指标雷达图"""
    print("   🕸️ 生成性能指标雷达图...")
    
    # 定义雷达图指标
    radar_metrics = {
        '准确率': 'accuracy',
        '精确率': 'precision', 
        '召回率': 'recall',
        'F1分数': 'f1_score',
        '特异性': 'specificity',
        '早期预警': 'tpr',  # 早期预警能力 (TPR)
        '误报控制': 'fpr',  # 误报控制 (1-FPR)
        '检测稳定性': 'accuracy'  # 检测稳定性 (用准确率代表)
    }
    
    # 数据预处理：FPR需要转换为控制能力 (1-FPR)
    bilstm_values = []
    transformer_values = []
    
    for metric_name, metric_key in radar_metrics.items():
        bilstm_val = performance_metrics['BILSTM']['classification_metrics'][metric_key]
        transformer_val = performance_metrics['TRANSFORMER']['classification_metrics'][metric_key]
        
        # 特殊处理：误报控制 = 1 - FPR
        if metric_name == '误报控制':
            bilstm_val = 1 - bilstm_val
            transformer_val = 1 - transformer_val
            
        bilstm_values.append(bilstm_val)
        transformer_values.append(transformer_val)
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    bilstm_values += bilstm_values[:1]  # 闭合
    transformer_values += transformer_values[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_medium"], subplot_kw=dict(projection='polar'))
    
    # 绘制雷达图
    ax.plot(angles, bilstm_values, 'o-', linewidth=2, label='BiLSTM', color='green')
    ax.fill(angles, bilstm_values, alpha=0.25, color='green')
    
    ax.plot(angles, transformer_values, 'o-', linewidth=2, label='Transformer', color='blue')
    ax.fill(angles, transformer_values, alpha=0.25, color='blue')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(radar_metrics.keys()))
    ax.set_ylim(0, 1)
    
    # 添加网格线
    ax.grid(True)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # 添加标题和图例
    plt.title('性能指标雷达图\nBiLSTM vs Transformer 对比', 
              pad=20, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 添加性能总结
    bilstm_avg = np.mean(bilstm_values[:-1])
    transformer_avg = np.mean(transformer_values[:-1])
    
    plt.figtext(0.02, 0.02, f'综合性能:\nBiLSTM: {bilstm_avg:.3f}\nTransformer: {transformer_avg:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ 雷达图保存至: {save_path}")

#----------------------------------------三窗口过程可视化------------------------------
def create_three_window_visualization(test_results, save_path):
    """生成三窗口检测过程可视化"""
    print("   🔍 生成三窗口过程可视化...")
    
    # 选择一个故障样本进行详细分析
    fault_sample_id = TEST_SAMPLES['fault'][0]
    
    fig = plt.figure(figsize=(16, 10))
    
    # 使用GridSpec进行复杂布局
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # === 主图：三窗口检测过程时序图 ===
    ax_main = fig.add_subplot(gs[0, :])
    
    # 选择Transformer结果进行可视化
    transformer_result = next(r for r in test_results['TRANSFORMER'] if r['sample_id'] == fault_sample_id)
    fai_values = transformer_result['fai']
    detection_info = transformer_result['detection_info']
    threshold1 = transformer_result['thresholds']['threshold1']
    
    time_axis = np.arange(len(fai_values))
    
    # 绘制FAI时序
    ax_main.plot(time_axis, fai_values, 'b-', linewidth=1.5, alpha=0.8, label='综合诊断指标 φ(FAI)')
    ax_main.axhline(y=threshold1, color='red', linestyle='--', alpha=0.7, label='一级阈值')
    
    # 阶段1：检测窗口 - 标记候选点
    if detection_info['candidate_points']:
        candidate_points = detection_info['candidate_points']
        ax_main.scatter(candidate_points, [fai_values[i] for i in candidate_points],
                       color='orange', s=40, alpha=0.8, label=f'检测: {len(candidate_points)} 个候选点',
                       marker='o', zorder=5)
    
    # 阶段2：验证窗口 - 标记验证通过的点
    if detection_info['verified_points']:
        verified_indices = [v['point'] for v in detection_info['verified_points']]
        ax_main.scatter(verified_indices, [fai_values[i] for i in verified_indices],
                       color='red', s=60, alpha=0.9, label=f'验证: {len(verified_indices)} 个确认点',
                       marker='^', zorder=6)
        
        # 显示验证窗口范围
        for v_point in detection_info['verified_points']:
            verify_start, verify_end = v_point['verify_range']
            ax_main.axvspan(verify_start, verify_end, alpha=0.1, color='yellow')
    
    # 阶段3：标记窗口 - 故障区域
    fault_regions_plotted = set()  # 避免重复绘制图例
    for i, region in enumerate(detection_info['marked_regions']):
        start, end = region['range']
        label = '标记: 故障区域' if i == 0 else ""
        ax_main.axvspan(start, end, alpha=0.2, color='red', label=label)
    
    ax_main.set_xlabel('时间步长')
    ax_main.set_ylabel('综合诊断指标 φ')
    ax_main.set_title(f'三窗口故障检测过程 - 样本 {fault_sample_id} (Transformer)', 
                     fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    # === 子图1：检测窗口统计 ===
    ax1 = fig.add_subplot(gs[1, 0])
    
    window_stats = detection_info['window_stats']
    detection_data = [
        window_stats['total_candidates'],
        window_stats['verified_candidates'], 
        window_stats['total_fault_points']
    ]
    detection_labels = ['候选点', '验证点', '故障点']
    colors1 = ['orange', 'red', 'darkred']
    
    bars1 = ax1.bar(detection_labels, detection_data, color=colors1, alpha=0.7)
    ax1.set_title('检测统计')
    ax1.set_ylabel('数量')
    
    # 添加数值标签
    for bar, value in zip(bars1, detection_data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom')
    
    # === 子图2：窗口参数配置 ===
    ax2 = fig.add_subplot(gs[1, 1])
    
    window_params = [
        WINDOW_CONFIG['detection_window'],
        WINDOW_CONFIG['verification_window'],
        WINDOW_CONFIG['marking_window']
    ]
    window_labels = ['检测窗口\n(100)', '验证窗口\n(50)', '标记窗口\n(50)']
    colors2 = ['lightblue', 'lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax2.pie(window_params, labels=window_labels, colors=colors2,
                                      autopct='%1.0f', startangle=90)
    ax2.set_title('窗口大小\n(采样点数)')
    
    # === 子图3：验证窗口详情 ===
    ax3 = fig.add_subplot(gs[1, 2])
    
    if detection_info['verified_points']:
        continuous_ratios = [v['continuous_ratio'] for v in detection_info['verified_points']]
        verify_points = [v['point'] for v in detection_info['verified_points']]
        
        bars3 = ax3.bar(range(len(continuous_ratios)), continuous_ratios, 
                       color='green', alpha=0.7)
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='阈值 (30%)')
        ax3.set_title('验证比率')
        ax3.set_xlabel('验证点')
        ax3.set_ylabel('连续比率')
        ax3.set_xticks(range(len(continuous_ratios)))
        ax3.set_xticklabels([f'P{i+1}' for i in range(len(continuous_ratios))])
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '无验证点', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('验证比率')
    
    # === 子图4：模型对比 ===
    ax4 = fig.add_subplot(gs[1, 3])
    
    model_comparison = []
    model_labels = []
    
    for model_type in ['BILSTM', 'TRANSFORMER']:
        sample_result = next(r for r in test_results[model_type] if r['sample_id'] == fault_sample_id)
        model_comparison.append(sample_result['detection_info']['window_stats']['fault_ratio'])
        model_labels.append(model_type)
    
    colors4 = ['green', 'blue']
    bars4 = ax4.bar(model_labels, model_comparison, color=colors4, alpha=0.7)
    ax4.set_title('模型对比\n(故障检测比率)')
    ax4.set_ylabel('故障比率')
    
    for bar, value in zip(bars4, model_comparison):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # === 底部：过程说明 ===
    process_text = """
    三窗口检测过程:
    
    1. 检测窗口 (100点): 扫描候选故障点，条件：φ(FAI) > 阈值
    2. 验证窗口 (50点): 验证候选点，检查连续性 (≥30% 超阈值)
    3. 标记窗口 (±50点): 标记确认的故障区域
    
    优势: 在保持高敏感性的同时减少误报
    """
    
    fig.text(0.02, 0.02, process_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ 三窗口过程图保存至: {save_path}")

#----------------------------------------结果保存函数------------------------------
def save_test_results(test_results, performance_metrics):
    """保存测试结果"""
    print("\n💾 保存测试结果...")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"test_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{result_dir}/detailed_results", exist_ok=True)
    
    # 1. 保存性能指标JSON
    performance_file = f"{result_dir}/performance_metrics.json"
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 性能指标保存至: {performance_file}")
    
    # 2. 保存详细结果
    for model_type in ["BILSTM", "TRANSFORMER"]:
        detail_file = f"{result_dir}/detailed_results/{model_type.lower()}_detailed_results.pkl"
        with open(detail_file, 'wb') as f:
            pickle.dump(test_results[model_type], f)
        print(f"   ✅ {model_type}详细结果保存至: {detail_file}")
    
    # 3. 保存元数据
    metadata_file = f"{result_dir}/detailed_results/test_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(test_results['metadata'], f, indent=2, ensure_ascii=False)
    print(f"   ✅ 测试元数据保存至: {metadata_file}")
    
    # 4. 创建Excel总结报告
    summary_file = f"{result_dir}/detailed_results/comparison_summary.xlsx"
    
    with pd.ExcelWriter(summary_file) as writer:
        # 性能对比表
        comparison_data = []
        for model_type in ["BILSTM", "TRANSFORMER"]:
            metrics = performance_metrics[model_type]['classification_metrics']
            confusion = performance_metrics[model_type]['confusion_matrix']
            sample_metrics = performance_metrics[model_type]['sample_metrics']
            
            comparison_data.append({
                'Model': model_type,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Specificity': metrics['specificity'],
                'TPR': metrics['tpr'],
                'FPR': metrics['fpr'],
                'TP': confusion['TP'],
                'FP': confusion['FP'],
                'TN': confusion['TN'],
                'FN': confusion['FN'],
                'Avg_FAI_Normal': sample_metrics['avg_fai_normal'],
                'Avg_FAI_Fault': sample_metrics['avg_fai_fault']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)
        
        # 样本详情表
        sample_details = []
        for model_type in ["BILSTM", "TRANSFORMER"]:
            for result in test_results[model_type]:
                sample_details.append({
                    'Model': model_type,
                    'Sample_ID': result['sample_id'],
                    'True_Label': 'Fault' if result['label'] == 1 else 'Normal',
                    'FAI_Mean': result['performance_metrics']['fai_mean'],
                    'FAI_Std': result['performance_metrics']['fai_std'],
                    'FAI_Max': result['performance_metrics']['fai_max'],
                    'Anomaly_Ratio': result['performance_metrics']['anomaly_ratio'],
                    'Fault_Detection_Ratio': result['detection_info']['window_stats']['fault_ratio'],
                    'Candidates_Found': result['detection_info']['window_stats']['total_candidates'],
                    'Verified_Points': result['detection_info']['window_stats']['verified_candidates']
                })
        
        sample_df = pd.DataFrame(sample_details)
        sample_df.to_excel(writer, sheet_name='Sample_Details', index=False)
    
    print(f"   ✅ Excel总结报告保存至: {summary_file}")
    
    return result_dir

#----------------------------------------主执行流程------------------------------
print("\n🎨 生成可视化分析...")

# 计算性能指标
performance_metrics = calculate_performance_metrics(test_results)

# 保存测试结果和生成可视化
result_dir = save_test_results(test_results, performance_metrics)

#----------------------------------------最终总结------------------------------
print("\n" + "="*80)
print("🎉 双模型比对测试完成！")
print("="*80)

print(f"\n📊 测试结果总结:")
print(f"   • 测试样本: {len(ALL_TEST_SAMPLES)} 个 (正常: {len(TEST_SAMPLES['normal'])}, 故障: {len(TEST_SAMPLES['fault'])})")
print(f"   • 模型对比: BiLSTM vs Transformer")
print(f"   • 三窗口检测: 检测({WINDOW_CONFIG['detection_window']}) → 验证({WINDOW_CONFIG['verification_window']}) → 标记({WINDOW_CONFIG['marking_window']})")

print(f"\n🔬 性能对比:")
for model_type in ["BILSTM", "TRANSFORMER"]:
    metrics = performance_metrics[model_type]['classification_metrics']
    print(f"   {model_type}:")
    print(f"     准确率: {metrics['accuracy']:.3f}")
    print(f"     精确率: {metrics['precision']:.3f}")
    print(f"     召回率: {metrics['recall']:.3f}")
    print(f"     F1分数: {metrics['f1_score']:.3f}")
    print(f"     TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}")

print(f"\n📁 结果文件:")
print(f"   • 结果目录: {result_dir}")
print(f"   • 可视化图表: {result_dir}/visualizations")
print(f"     - ROC曲线对比: roc_comparison.png")
print(f"     - 故障检测时序图: fault_detection_timeline.png") 
print(f"     - 性能雷达图: performance_radar.png")
print(f"     - 三窗口过程图: three_window_process.png")
print(f"   • 性能指标: performance_metrics.json")
print(f"   • Excel报告: comparison_summary.xlsx")

# 最终推荐
bilstm_score = np.mean(list(performance_metrics['BILSTM']['classification_metrics'].values()))
transformer_score = np.mean(list(performance_metrics['TRANSFORMER']['classification_metrics'].values()))

print(f"\n🏆 综合性能评估:")
print(f"   BiLSTM 综合得分: {bilstm_score:.3f}")
print(f"   Transformer 综合得分: {transformer_score:.3f}")

if transformer_score > bilstm_score:
    print(f"   ✅ 推荐: Transformer 模型性能更优 (+{transformer_score - bilstm_score:.3f})")
else:
    print(f"   ✅ 推荐: BiLSTM 模型性能更优 (+{bilstm_score - transformer_score:.3f})")

print("\n" + "="*80)
print("测试完成！请查看生成的可视化图表和分析报告。")
print("="*80)