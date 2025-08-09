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
# 确保从当前目录导入更新后的Comprehensive_calculation函数
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from Comprehensive_calculation import Comprehensive_calculation

# 新增导入
from tqdm import tqdm
import json
import time
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
import glob

# 添加模型加载辅助函数
def remove_module_prefix(state_dict):
    """移除state_dict中的module.前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除'module.'前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def safe_get_nested(dictionary, keys, default=None):
    """安全获取嵌套字典的值"""
    try:
        for key in keys:
            dictionary = dictionary[key]
        return dictionary
    except (KeyError, TypeError, IndexError):
        return default

def safe_load_model(model, model_path, model_name):
    """安全加载模型，处理DataParallel前缀问题"""
    try:
        print(f"   正在加载{model_name}模型: {model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"   ❌ 模型文件不存在: {model_path}")
            return False
        
        state_dict = torch.load(model_path, map_location=device)
        
        # 检查是否需要移除module前缀
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            print(f"   检测到DataParallel前缀，正在移除...")
            state_dict = remove_module_prefix(state_dict)
        
        # 检查模型结构匹配
        model_state_dict = model.state_dict()
        missing_keys = []
        unexpected_keys = []
        
        for key in model_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        for key in state_dict.keys():
            if key not in model_state_dict:
                unexpected_keys.append(key)
        
        if missing_keys:
            print(f"   ⚠️  缺失键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"   ⚠️  缺失键: {missing_keys}")
        
        if unexpected_keys:
            print(f"   ⚠️  多余键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"   ⚠️  多余键: {unexpected_keys}")
        
        # 尝试加载
        model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ {model_name}模型加载成功")
        return True
        
    except Exception as e:
        print(f"   ❌ {model_name}模型加载失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return False

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU配置检查
print("🖥️ GPU配置检查:")
print(f"   CUDA可用: {torch.cuda.is_available()}")
print(f"   GPU数量: {torch.cuda.device_count()}")
print(f"   当前设备: {device}")

if torch.cuda.is_available():
    print("   GPU详细信息:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"     GPU {i}: {props.name}")
        print(f"       显存: {props.total_memory/1024**3:.1f}GB")
        print(f"       计算能力: {props.major}.{props.minor}")
else:
    print("   ⚠️ 使用CPU模式")

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体显示
import matplotlib.font_manager as fm
from matplotlib import rcParams
import platform

def setup_chinese_fonts_strict():
    """Linux服务器环境中文字体配置（增强版）"""
    import subprocess
    import os
    
    # 1. 尝试安装中文字体包（仅Linux）
    if platform.system() == "Linux":
        try:
            # 检查是否有管理员权限安装字体
            result = subprocess.run(['which', 'apt-get'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("🔧 正在尝试安装中文字体包...")
                subprocess.run(['sudo', 'apt-get', 'update'], capture_output=True, timeout=30)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-noto-cjk', 'fonts-wqy-microhei', 'fonts-arphic-ukai'], capture_output=True, timeout=60)
        except Exception as e:
            print(f"⚠️ 字体安装失败（可能需要管理员权限）: {e}")
    
    # 2. 扩展候选字体列表
    candidates = [
        # Linux优先字体
        'Noto Sans CJK SC Regular',
        'Noto Sans CJK SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Source Han Sans CN',
        'Source Han Sans SC',
        'AR PL UKai CN',
        'AR PL UMing CN',
        # 通用字体
        'Droid Sans Fallback',
        'Liberation Sans',
        # Windows兜底
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        # 最终兜底
        'DejaVu Sans',
        'Liberation Sans'
    ]

    chosen = None
    for name in candidates:
        try:
            font_path = fm.findfont(name, fallback_to_default=False)
            if font_path and 'DejaVu' not in font_path and os.path.exists(font_path):
                chosen = name
                print(f"🔍 找到字体: {name} -> {font_path}")
                break
        except Exception:
            continue

    # 3. 如果没找到合适字体，尝试系统字体扫描
    if chosen is None:
        print("🔍 进行系统字体扫描...")
        all_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = [f for f in all_fonts if any(keyword in f.lower() for keyword in ['cjk', 'han', 'hei', 'kai', 'ming', 'noto', 'wenquanyi'])]
        if chinese_fonts:
            chosen = chinese_fonts[0]
            print(f"🔍 通过扫描找到中文字体: {chosen}")
        else:
            chosen = 'DejaVu Sans'
            print("⚠️ 未找到中文字体，使用DejaVu Sans")

    # 4. 增强的全局渲染参数
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans', 'Liberation Sans']
    rcParams['axes.unicode_minus'] = False
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['savefig.dpi'] = 300
    rcParams['figure.dpi'] = 100  # 降低以提高兼容性
    rcParams['figure.autolayout'] = False
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    
    # 5. 强制字体缓存重建
    try:
        fm._rebuild()
        # 额外清理缓存
        cache_dir = os.path.expanduser('~/.cache/matplotlib')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠️ 字体缓存清理失败: {e}")

    print(f"✅ 最终使用字体: {chosen}")
    
    # 6. 测试中文显示
    try:
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, 'Font Test', fontsize=10)
        plt.close()
        print("✅ 中文字体测试通过")
    except Exception as e:
        print(f"⚠️ 中文字体测试失败: {e}")
        # 降级到安全模式
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("🔄 已切换到安全模式（英文标签）")

# 执行字体配置（更稳健）
setup_chinese_fonts_strict()

#----------------------------------------测试配置------------------------------
print("="*60)
print("🔬 电池故障诊断系统 - BiLSTM模型测试")
print("="*60)

TEST_MODE = "BILSTM_ONLY"  # 固定为BiLSTM单模型测试

# 测试数据集配置 (根据Labels.xls动态加载)
def load_test_samples():
    """从Labels.xls加载测试样本"""
    try:
        import pandas as pd
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取测试样本
        all_samples = df['Num'].tolist()
        all_labels = df['Label'].tolist()
        
        # 指定测试样本：正常样本10-20 和故障样本340-350
        test_normal_samples = [str(i) for i in range(10, 21)]  # 正常样本：10-20
        test_fault_samples = [str(i) for i in range(340, 351)]  # 故障样本：340-350
        
        print(f"📋 从Labels.xls加载测试样本:")
        print(f"   测试正常样本: {test_normal_samples}")
        print(f"   测试故障样本: {test_fault_samples}")
        
        return {
            'normal': test_normal_samples,
            'fault': test_fault_samples
        }
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认测试样本")
        return {
            'normal': [str(i) for i in range(10, 21)],  # 正常样本：10-20
            'fault': [str(i) for i in range(340, 351)]  # 故障样本：340-350
        }

TEST_SAMPLES = load_test_samples()
ALL_TEST_SAMPLES = TEST_SAMPLES['normal'] + TEST_SAMPLES['fault']

# 模型路径配置 (从BiLSTM训练结果目录加载)
MODEL_PATHS = {
    "BILSTM": {
        "net_model": "/mnt/bz25t/bzhy/datasave/BILSTM/models/net_model_bilstm_baseline.pth",
        "netx_model": "/mnt/bz25t/bzhy/datasave/BILSTM/models/netx_model_bilstm_baseline.pth"
    }
}

# 检测模式配置
DETECTION_MODES = {
    "three_point": {
        "name": "3点检测模式（原版）", 
        "description": "对于故障样本，如果某点高于阈值且前后相邻点也高于阈值，则标记该点及前后1个点（共3个点）",
        "function": "three_point_fault_detection"
    },
    "three_point_improved": {
        "name": "3点检测模式（改进版）",
        "description": "改进的3点检测：严格的触发条件 + 分级标记范围 + 有效降噪机制",
        "function": "three_point_fault_detection"
    }
}

# 当前使用的检测模式
CURRENT_DETECTION_MODE = "three_point_improved"  # 使用改进的3点检测模式
# 备选：如果改进版仍然过严格，可以切换回 "three_point" 原版

# 3点检测配置
# 设计原理：
# 1. 触发条件：中心点+前后相邻点共3个点满足阈值条件
# 2. 标记范围：标记触发点前后各1个点（共3个点）
# 3. 分级检测：根据阈值严格程度分为3个等级
THREE_POINT_CONFIG = {
    "marking_range": 1,          # 标记范围：前后各1个点
    "neighbor_check": True,      # 是否检查邻居点
    "multi_level": True,         # 是否启用多级检测
    "startup_period": 3000       # 启动期（跳过前3000个点）
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
print(f"   检测模式: {DETECTION_MODES[CURRENT_DETECTION_MODE]['name']}")
print(f"   3点检测参数: {THREE_POINT_CONFIG}")
print(f"   3点检测模式: 当前点+前后相邻点高于阈值时，标记3点区域")

#----------------------------------------模型文件检查------------------------------
def check_model_files():
    """检查BiLSTM模型文件"""
    print("\n🔍 检查BiLSTM模型文件...")
    
    missing_files = []
    paths = MODEL_PATHS["BILSTM"]
    
    # 检查主模型文件
    for key, path in paths.items():
        if not os.path.exists(path):
            missing_files.append(f"BILSTM: {path}")
            print(f"   ❌ 缺失: {path}")
        else:
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   ✅ 存在: {path} ({file_size:.1f}MB)")
    
    # 检查PCA参数文件 (从BiLSTM训练结果加载)
    pca_params_path = "/mnt/bz25t/bzhy/datasave/BILSTM/models/pca_params_bilstm_baseline.pkl"
    if not os.path.exists(pca_params_path):
        # 尝试从npy文件重建PCA参数
        print(f"   ⚠️  PCA参数文件不存在，尝试从npy文件重建...")
        try:
            # 加载PCA相关参数
            data_mean = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/data_mean_bilstm_baseline.npy")
            data_std = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/data_std_bilstm_baseline.npy")
            v = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/v_bilstm_baseline.npy")
            p_k = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/p_k_bilstm_baseline.npy")
            v_I = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/v_I_bilstm_baseline.npy")
            T_99_limit = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/T_99_limit_bilstm_baseline.npy")
            SPE_99_limit = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/SPE_99_limit_bilstm_baseline.npy")
            X = np.load("/mnt/bz25t/bzhy/datasave/BILSTM/models/X_bilstm_baseline.npy")
            
            # 重建PCA参数字典
            pca_params = {
                'data_mean': data_mean,
                'data_std': data_std,
                'v': v,
                'p_k': p_k,
                'v_I': v_I,
                'T_99_limit': T_99_limit,
                'SPE_99_limit': SPE_99_limit,
                'X': X
            }
            
            # 保存重建的PCA参数
            with open(pca_params_path, 'wb') as f:
                pickle.dump(pca_params, f)
            print(f"   ✅ PCA参数重建并保存: {pca_params_path}")
            
        except Exception as e:
            missing_files.append(f"PCA_PARAMS: {pca_params_path}")
            print(f"   ❌ PCA参数重建失败: {e}")
    else:
        file_size = os.path.getsize(pca_params_path) / (1024 * 1024)  # MB
        print(f"   ✅ 存在: {pca_params_path} ({file_size:.1f}MB)")
    
    if missing_files:
        print(f"\n❌ 缺失 {len(missing_files)} 个模型文件:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 解决方案:")
        print("   1. 确保已运行BiLSTM训练脚本")
        print("   2. 检查模型文件路径是否正确")
        print("   3. 检查文件权限")
        raise FileNotFoundError("请先运行BiLSTM训练脚本生成所需模型文件")
    
    print("✅ BiLSTM模型文件检查通过")
    return True

# 执行模型文件检查
check_model_files()

#----------------------------------------3点故障检测机制------------------------------
def three_point_fault_detection(fai_values, threshold1, sample_id, config=None):
    """
    改进的3点故障检测机制：增强连续性检测和降噪能力
    
    设计原理：
    1. 严格的触发条件：要求中心点及其邻域满足更严格的一致性
    2. 合理的标记范围：根据故障级别标记3个点的区域
    3. 有效的降噪机制：过滤孤立异常点，关注持续性故障
    
    Args:
        fai_values: 综合诊断指标序列
        threshold1: 一级预警阈值
        sample_id: 样本ID（用于调试）
        config: 配置参数（兼容性参数，可包含threshold2, threshold3）
    
    Returns:
        fault_labels: 故障标签序列 (0=正常, 1=轻微故障, 2=中等故障, 3=严重故障)
        detection_info: 检测过程详细信息
    """
    # 与Transformer保持一致：跳过启动期并首选外部阈值
    STARTUP_PERIOD = 3000
    fault_labels = np.zeros(len(fai_values), dtype=int)
    detection_info = {
        'trigger_points': [],
        'marked_regions': [],
        'detection_stats': {},
        'fai_stats': {
            'mean': np.mean(fai_values),
            'std': np.std(fai_values),
            'max': np.max(fai_values),
            'min': np.min(fai_values)
        },
        'startup_period': STARTUP_PERIOD
    }

    # 样本类型判定（字符串对齐）
    sample_id_str = str(sample_id)
    is_fault_sample = sample_id_str in TEST_SAMPLES['fault']
    is_normal_sample = sample_id_str in TEST_SAMPLES['normal']

    if not is_fault_sample and not is_normal_sample:
        print(f"   ⚠️ 样本{sample_id}未出现在两类列表中，默认按故障样本处理")
        is_fault_sample = True

    # 正常样本：不标注任何故障，输出假阳性统计（与Transformer一致）
    if not is_fault_sample:
        startup_fai = fai_values[:STARTUP_PERIOD] if len(fai_values) > STARTUP_PERIOD else fai_values
        stable_fai = fai_values[STARTUP_PERIOD:] if len(fai_values) > STARTUP_PERIOD else []
        startup_fp = np.sum(startup_fai > threshold1) if len(startup_fai) > 0 else 0
        stable_fp = np.sum(stable_fai > threshold1) if len(stable_fai) > 0 else 0
        total_fp = startup_fp + stable_fp
        detection_info['detection_stats'] = {
            'total_trigger_points': 0,
            'total_marked_regions': 0,
            'total_fault_points': 0,
            'fault_ratio': 0.0,
            'detection_mode': 'normal_sample',
            'startup_false_positives': int(startup_fp),
            'stable_false_positives': int(stable_fp),
            'total_false_positives': int(total_fp),
            'startup_fp_ratio': float(startup_fp/len(startup_fai)) if len(startup_fai) > 0 else 0.0,
            'stable_fp_ratio': float(stable_fp/len(stable_fai)) if len(stable_fai) > 0 else 0.0,
            'total_fp_ratio': float(total_fp/len(fai_values)) if len(fai_values) > 0 else 0.0
        }
        fault_labels.fill(0)
        return fault_labels, detection_info

    # 获取/计算多级阈值（优先外部配置）
    if config and 'threshold2' in config and 'threshold3' in config:
        threshold2 = config['threshold2']
        threshold3 = config['threshold3']
        print(f"   ✅ 使用外部阈值: T1={threshold1:.4f}, T2={threshold2:.4f}, T3={threshold3:.4f}")
    else:
        nm = STARTUP_PERIOD
        mm = len(fai_values)
        if mm > nm:
            baseline_fai = fai_values[nm:mm]
            mean_fai = np.mean(baseline_fai)
            std_fai = np.std(baseline_fai)
        else:
            mean_fai = np.mean(fai_values)
            std_fai = np.std(fai_values)
        threshold2 = mean_fai + 4.5 * std_fai
        threshold3 = mean_fai + 6.0 * std_fai
        print(f"   ℹ️ 内部阈值计算: T2={threshold2:.4f}, T3={threshold3:.4f}")
    
    # 故障样本：实施改进的多级3点检测（与Transformer一致，跳过启动期）
    trigger_points = []
    marked_regions = []
    
    # 策略4.0：三级分级检测策略
    print(f"   🔧 策略4.0: 三级分级检测策略...")
    print(f"   分析结果: 故障样本有{np.sum(fai_values > threshold1)}个异常点({np.sum(fai_values > threshold1)/len(fai_values)*100:.2f}%)")
    print(f"   阈值分布: >6σ({np.sum(fai_values > threshold3)}个), >4.5σ({np.sum(fai_values > threshold2)}个), >3σ({np.sum(fai_values > threshold1)}个)")
    
    detection_config = {
        'mode': 'hierarchical_v2',
        'level_3': {
            'center_threshold': threshold3,      # 6σ
            'neighbor_threshold': None,          # 无邻域要求
            'min_neighbors': 0,
            'marking_range': [-1, 0, 1],        # 标记i-1, i, i+1
            'condition': 'level3_high_confidence'
        },
        'level_2': {
            'center_threshold': threshold2,      # 4.5σ  
            'neighbor_threshold': threshold1,    # 3σ
            'min_neighbors': 1,
            'marking_range': [-1, 0, 1],        # 标记i-1, i, i+1
            'condition': 'level2_medium_confidence'
        },
        'level_1': {
            'center_threshold': threshold1,      # 3σ
            'neighbor_threshold': threshold1 * 0.67,  # 2σ
            'min_neighbors': 1,
            'marking_range': [-1, 0, 1],        # 标记i-1, i, i+1 (3个点)
            'condition': 'level1_basic_confidence'
        }
    }
    
    print(f"   检测参数:")
    print(f"   Level 3 (6σ): 中心阈值={threshold3:.4f}, 无邻域要求, 标记3点")
    print(f"   Level 2 (4.5σ): 中心阈值={threshold2:.4f}, 邻域阈值={threshold1:.4f}, 最少邻居=1个, 标记3点")
    print(f"   Level 1 (3σ): 中心阈值={threshold1:.4f}, 邻域阈值={threshold1*0.67:.4f}, 最少邻居=1个, 标记3点")
    
    # 三级分级检测实现：从稳定期开始
    triggers = []
    detection_start = max(STARTUP_PERIOD + 2, 2)
    detection_end = len(fai_values) - 2
    for i in range(detection_start, detection_end):
        neighborhood = fai_values[i-2:i+3]  # 5个点的邻域
        neighbors = [fai_values[i-2], fai_values[i-1], fai_values[i+1], fai_values[i+2]]  # 4个邻居
        center = fai_values[i]
        
        triggered = False
        trigger_level = None
        trigger_condition = None
        detection_details = {}
        
        # Level 3: 最严格阈值，最宽松条件 (6σ)
        if center > detection_config['level_3']['center_threshold']:
            triggered = True
            trigger_level = 3
            trigger_condition = detection_config['level_3']['condition']
            marking_range = detection_config['level_3']['marking_range']
            detection_details = {
                'center_value': center,
                'center_threshold': detection_config['level_3']['center_threshold'],
                'neighbors_above_threshold': 'N/A (no requirement)',
                'required_neighbors': 0,
                'neighborhood_values': neighborhood.tolist(),
                'trigger_reason': '6σ high confidence detection'
            }
            
        # Level 2: 中等阈值，中等条件 (4.5σ) 
        elif center > detection_config['level_2']['center_threshold']:
            neighbors_above_t1 = np.sum(np.array(neighbors) > detection_config['level_2']['neighbor_threshold'])
            if neighbors_above_t1 >= detection_config['level_2']['min_neighbors']:
                triggered = True
                trigger_level = 2
                trigger_condition = detection_config['level_2']['condition']
                marking_range = detection_config['level_2']['marking_range']
                detection_details = {
                    'center_value': center,
                    'center_threshold': detection_config['level_2']['center_threshold'],
                    'neighbors_above_threshold': neighbors_above_t1,
                    'required_neighbors': detection_config['level_2']['min_neighbors'],
                    'neighborhood_values': neighborhood.tolist(),
                    'trigger_reason': '4.5σ medium confidence detection'
                }
                
        # Level 1: 最低阈值，相对严格条件 (3σ)
        elif center > detection_config['level_1']['center_threshold']:
            neighbors_above_2sigma = np.sum(np.array(neighbors) > detection_config['level_1']['neighbor_threshold'])
            if neighbors_above_2sigma >= detection_config['level_1']['min_neighbors']:
                triggered = True
                trigger_level = 1
                trigger_condition = detection_config['level_1']['condition']
                marking_range = detection_config['level_1']['marking_range']
                detection_details = {
                    'center_value': center,
                    'center_threshold': detection_config['level_1']['center_threshold'],
                    'neighbors_above_threshold': neighbors_above_2sigma,
                    'required_neighbors': detection_config['level_1']['min_neighbors'],
                    'neighborhood_values': neighborhood.tolist(),
                    'trigger_reason': '3σ basic confidence detection'
                }
        
        if triggered:
            # 计算标记范围
            start_mark = max(0, i + min(marking_range))
            end_mark = min(len(fai_values), i + max(marking_range) + 1)
            
            triggers.append({
                'center': i,
                'level': trigger_level,
                'range': (start_mark, end_mark),
                'trigger_condition': trigger_condition,
                'detection_details': detection_details
            })
    
    # 第二轮：处理所有触发点（分级处理）
    processed_triggers = []
    level_counts = {1: 0, 2: 0, 3: 0}
    
    for trigger in triggers:
        start, end = trigger['range']
        center = trigger['center']
        level = trigger['level']
        
        # 统计各级别触发次数
        level_counts[level] += 1
        
        # 标记故障区域
        fault_labels[start:end] = level  # 使用级别作为标记值
        trigger_points.append(center)
        
        # 记录区域信息
        region_data = fai_values[start:end]
        region_stats = {
            'mean_fai': np.mean(region_data),
            'max_fai': np.max(region_data),
            'min_fai': np.min(region_data),
            'std_fai': np.std(region_data),
            'length': end - start
        }
        
        marked_regions.append({
            'trigger_point': center,
            'level': level,  # 分级标记
            'range': (start, end),
            'length': end - start,
            'region_stats': region_stats,
            'trigger_condition': trigger['trigger_condition'],
            'trigger_values': {
                'center': fai_values[center],
                'detection_level': f"Level {level}",
                'trigger_reason': trigger['detection_details']['trigger_reason']
            }
        })
        
        processed_triggers.append(trigger)
    
    print(f"   触发统计: Level 3({level_counts[3]}次), Level 2({level_counts[2]}次), Level 1({level_counts[1]}次)")
    
    detection_info['trigger_points'] = trigger_points
    detection_info['marked_regions'] = marked_regions
    detection_info['processed_triggers'] = processed_triggers
    
    # 为兼容旧版检测模式的可视化代码，添加空的兼容字段
    detection_info['candidate_points'] = []  # 3点检测模式中不使用，但为兼容性保留
    detection_info['verified_points'] = []   # 3点检测模式中不使用，但为兼容性保留
    
    # 统计信息（分级检测）
    fault_count = np.sum(fault_labels > 0)  # 全序列
    effective_labels = fault_labels[STARTUP_PERIOD:] if len(fault_labels) > STARTUP_PERIOD else fault_labels
    effective_fault_count = np.sum(effective_labels > 0) if len(effective_labels) > 0 else 0
    level1_count = np.sum(fault_labels == 1)
    level2_count = np.sum(fault_labels == 2)
    level3_count = np.sum(fault_labels == 3)

    detection_info['detection_stats'] = {
        'total_trigger_points': len(trigger_points),
        'total_marked_regions': len(marked_regions),
        'total_fault_points': int(fault_count),
        'effective_fault_points': int(effective_fault_count),
        'fault_ratio': float(fault_count / len(fault_labels)) if len(fault_labels) > 0 else 0.0,
        'effective_fault_ratio': float(effective_fault_count / len(effective_labels)) if len(effective_labels) > 0 else 0.0,
        'detection_mode': 'hierarchical_three_level_with_startup_skip',
        'startup_period': STARTUP_PERIOD,
        'effective_length': len(effective_labels) if len(effective_labels) > 0 else 0,
        'level_statistics': {
            'level_1_points': int(level1_count),
            'level_2_points': int(level2_count),
            'level_3_points': int(level3_count),
            'level_1_triggers': int(level_counts[1]),
            'level_2_triggers': int(level_counts[2]),
            'level_3_triggers': int(level_counts[3])
        },
        'mean_region_length': float(np.mean([m['length'] for m in marked_regions])) if marked_regions else 0.0,
        'mean_trigger_fai': float(np.mean([m['trigger_values']['center'] for m in marked_regions])) if marked_regions else 0.0,
        'strategy_used': 'strategy_4_hierarchical_detection_startup_aware',
        'parameters': detection_config
    }
    
    print(f"   → 策略4.0检测结果: 检测到故障点={fault_count}个 ({fault_count/len(fault_labels)*100:.2f}%)")
    print(f"   → 分级统计: L1={level1_count}点, L2={level2_count}点, L3={level3_count}点")
    print(f"   → 触发点数: {len(triggers)}个, 标记区域: {len(marked_regions)}个")
    
    # 添加改进效果对比
    original_anomaly_count = np.sum(fai_values > threshold1)
    detected_fault_count = np.sum(fault_labels > 0)
    noise_reduction_ratio = 1 - (detected_fault_count / original_anomaly_count) if original_anomaly_count > 0 else 0
    
    print(f"   → 降噪效果: 原始异常点={original_anomaly_count}, 检测故障点={detected_fault_count}, 降噪率={noise_reduction_ratio:.2%}")
    
    # 如果策略1没有检测到故障，自动切换到策略2
    if detected_fault_count == 0 and is_fault_sample:
        print(f"   ⚠️  策略1未检测到故障点，自动切换到策略2...")
        print(f"   🔧 策略2: 进一步放宽邻域要求（邻域阈值=0.6×3σ, 无邻居要求）")
        
        # 重置标签和列表
        fault_labels.fill(0)
        trigger_points.clear()
        marked_regions.clear()
        
        # 策略2参数
        strategy2_config = {
            'center_threshold': threshold1,           # 保持3σ阈值
            'neighbor_threshold': threshold1 * 0.6,  # 进一步降低邻域要求到60%
            'min_neighbors': 0,                      # 不要求邻居（纯中心点检测）
            'marking_range': 1,                      # 标记±1个点（3点总共）
            'condition': 'strategy2_relaxed'
        }
        
        # 策略2检测
        strategy2_triggers = []
        for i in range(2, len(fai_values) - 2):
            center = fai_values[i]
            
            # 策略2条件：只检查中心点
            if center > strategy2_config['center_threshold']:
                start_mark = max(0, i - strategy2_config['marking_range'])
                end_mark = min(len(fai_values), i + strategy2_config['marking_range'] + 1)
                
                fault_labels[start_mark:end_mark] = 1
                trigger_points.append(i)
                
                region_data = fai_values[start_mark:end_mark]
                marked_regions.append({
                    'trigger_point': i,
                    'level': 1,
                    'range': (start_mark, end_mark),
                    'length': end_mark - start_mark,
                    'region_stats': {
                        'mean_fai': np.mean(region_data),
                        'max_fai': np.max(region_data),
                        'std_fai': np.std(region_data),
                        'length': end_mark - start_mark
                    },
                    'trigger_condition': strategy2_config['condition'],
                    'trigger_values': {
                        'center': center
                    }
                })
        
        # 重新计算统计信息
        detected_fault_count = np.sum(fault_labels > 0)
        
        # 更新detection_info
        detection_info['trigger_points'] = trigger_points
        detection_info['marked_regions'] = marked_regions
        detection_info['detection_stats']['total_trigger_points'] = len(trigger_points)
        detection_info['detection_stats']['total_marked_regions'] = len(marked_regions)
        detection_info['detection_stats']['total_fault_points'] = detected_fault_count
        detection_info['detection_stats']['fault_ratio'] = detected_fault_count / len(fault_labels)
        detection_info['detection_stats']['strategy_used'] = 'strategy_2_center_only'
        detection_info['detection_stats']['parameters'] = strategy2_config
        
        noise_reduction_ratio = 1 - (detected_fault_count / original_anomaly_count) if original_anomaly_count > 0 else 0
        
        print(f"   → 策略2检测结果: 检测到故障点={detected_fault_count}个 ({detected_fault_count/len(fault_labels)*100:.2f}%)")
        print(f"   → 触发点数: {len(trigger_points)}个, 标记区域: {len(marked_regions)}个")
        print(f"   → 策略2降噪效果: 原始异常点={original_anomaly_count}, 检测故障点={detected_fault_count}, 降噪率={noise_reduction_ratio:.2%}")
    
    elif detected_fault_count == 0:
        print(f"   ⚠️  正常样本未检测到故障点，符合预期")
        print(f"   → 检测逻辑工作正常")
    
    return fault_labels, detection_info


#----------------------------------------数据加载函数------------------------------
def load_test_sample(sample_id):
    """加载测试样本"""
    base_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}'
    
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

def load_models():
    """加载BiLSTM模型"""
    models = {}
    
    print("🔧 开始加载BiLSTM模型...")
    
    # 加载MC-AE模型 (BiLSTM训练脚本使用的是CombinedAE)
    models['net'] = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                              activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    models['netx'] = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                               activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
    
    # 使用安全加载函数
    if not safe_load_model(models['net'], 
                          MODEL_PATHS["BILSTM"]["net_model"], 
                          "MC-AE1"):
        raise RuntimeError("MC-AE1模型加载失败")
    
    if not safe_load_model(models['netx'], 
                          MODEL_PATHS["BILSTM"]["netx_model"], 
                          "MC-AE2"):
        raise RuntimeError("MC-AE2模型加载失败")
    
    # 加载PCA参数 (从pickle文件加载)
    pca_params_path = "/mnt/bz25t/bzhy/datasave/BILSTM/models/pca_params_bilstm_baseline.pkl"
    try:
        with open(pca_params_path, 'rb') as f:
            models['pca_params'] = pickle.load(f)
        print(f"✅ PCA参数已加载: {pca_params_path}")
    except Exception as e:
        print(f"❌ 加载PCA参数失败: {e}")
        raise RuntimeError("PCA参数加载失败")
    
    return models

#----------------------------------------单样本处理函数------------------------------
def process_single_sample(sample_id, models):
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
        # 确保数据类型一致
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
    
    # 使用预训练的PCA参数进行综合计算
    time = np.arange(df_data.shape[0])
    
    # 先进行初步计算获取fai值，用于阈值计算
    print("   进行初步FAI计算...")
    temp_result = Comprehensive_calculation(
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
    temp_fai = temp_result[9]  # fai是第10个返回值（索引9）
    
    # 按照源代码方式计算阈值（与源代码保持一致）
    print("   计算报警阈值...")
    nm = 3000  # 固定值，与源代码一致
    mm = len(temp_fai)  # 数据总长度
    
    # 确保数据长度足够
    if mm > nm:
        # 使用后半段数据计算阈值
        threshold1 = np.mean(temp_fai[nm:mm]) + 3*np.std(temp_fai[nm:mm])
        threshold2 = np.mean(temp_fai[nm:mm]) + 4.5*np.std(temp_fai[nm:mm])
        threshold3 = np.mean(temp_fai[nm:mm]) + 6*np.std(temp_fai[nm:mm])
    else:
        # 数据太短，使用全部数据
        print(f"   ⚠️ 样本{sample_id}数据长度({mm})不足3000，使用全部数据计算阈值")
        threshold1 = np.mean(temp_fai) + 3*np.std(temp_fai)
        threshold2 = np.mean(temp_fai) + 4.5*np.std(temp_fai)
        threshold3 = np.mean(temp_fai) + 6*np.std(temp_fai)
    
    print(f"   外部计算阈值: L1={threshold1:.4f}, L2={threshold2:.4f}, L3={threshold3:.4f}")
    
    # 使用外部计算的阈值重新计算报警等级（不使用Comprehensive_calculation内部的报警等级）
    print("   使用外部阈值重新计算报警等级...")
    lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, old_level, old_maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std = temp_result
    
    # 使用外部计算的阈值重新计算报警等级
    level = np.zeros_like(fai, dtype=int)
    level[fai > threshold1] = 1
    level[fai > threshold2] = 2
    level[fai > threshold3] = 3
    maxlevel = np.max(level)
    
    # 显示修正后的报警统计
    print(f"   外部阈值报警点数: L1={np.sum(level==1)}, L2={np.sum(level==2)}, L3={np.sum(level==3)}")
    print(f"   最大报警等级: {maxlevel}")
    print(f"   (内部计算的报警点数被忽略，使用外部阈值重新计算)")
    
    # 根据检测模式选择检测函数
    threshold_config = {
        'threshold1': threshold1,
        'threshold2': threshold2,
        'threshold3': threshold3
    }
    if CURRENT_DETECTION_MODE in ("three_point", "three_point_improved"):
        fault_labels, detection_info = three_point_fault_detection(fai, threshold1, sample_id, threshold_config)
    else:
        # 默认使用3点检测
        fault_labels, detection_info = three_point_fault_detection(fai, threshold1, sample_id, threshold_config)
    
    # 构建结果
    sample_result = {
        'sample_id': sample_id,
        'model_type': 'BILSTM',
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
            'anomaly_count': np.sum(fai > threshold1),  # L1阈值异常点数
            'anomaly_ratio': np.sum(fai > threshold1) / len(fai),
            'alarm_counts': {
                'L1': int(np.sum(level==1)),
                'L2': int(np.sum(level==2)), 
                'L3': int(np.sum(level==3))
            },
            'external_thresholds_used': True  # 标记使用了外部阈值
        }
    }
    
    return sample_result

#----------------------------------------主测试流程------------------------------
def main_test_process():
    """主要测试流程"""
    
    # 初始化结果存储
    test_results = {
        "BILSTM": [],
        "metadata": {
            "test_samples": TEST_SAMPLES,
            "three_point_config": THREE_POINT_CONFIG,
            "detection_modes": DETECTION_MODES,
            "current_mode": CURRENT_DETECTION_MODE,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # BiLSTM单模型测试
    total_operations = len(ALL_TEST_SAMPLES)  # 22个样本 (11正常+11故障)
    
    print(f"\n🚀 开始BiLSTM模型测试...")
    print(f"检测模式: {DETECTION_MODES[CURRENT_DETECTION_MODE]['name']}")
    print(f"检测描述: {DETECTION_MODES[CURRENT_DETECTION_MODE]['description']}")
    print(f"总共需要处理: {total_operations} 个样本")
    
    with tqdm(total=total_operations, desc="BiLSTM测试进度", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:
        
        print(f"\n{'='*20} 测试 BiLSTM 模型 {'='*20}")
        
        # 加载模型
        pbar.set_description(f"加载BiLSTM模型")
        models = load_models()
        print(f"✅ BiLSTM 模型加载完成")
        
        for sample_id in ALL_TEST_SAMPLES:
            pbar.set_description(f"BiLSTM-样本{sample_id}")
            
            try:
                # 处理单个样本
                sample_result = process_single_sample(sample_id, models)
                test_results["BILSTM"].append(sample_result)
                
                # 输出简要结果
                metrics = sample_result.get('performance_metrics', {})
                detection_info = sample_result.get('detection_info', {})
                
                # 3点检测模式 - 安全获取检测统计
                detection_stats = detection_info.get('detection_stats', {})
                detection_ratio = detection_stats.get('fault_ratio', 0.0)
                
                print(f"   样本{sample_id}: fai均值={metrics.get('fai_mean', 0.0):.6f}, "
                      f"异常率={metrics.get('anomaly_ratio', 0.0):.2%}, "
                      f"检测率={detection_ratio:.2%}")
                
            except Exception as e:
                print(f"❌ 样本 {sample_id} 处理失败: {e}")
                continue
            
            pbar.update(1)
            time.sleep(0.1)  # 避免进度条更新过快
    
    print(f"\n✅ BiLSTM测试完成!")
    print(f"   BiLSTM: 成功处理 {len(test_results['BILSTM'])} 个样本")
    
    return test_results

# 执行主测试流程
test_results = main_test_process()

#----------------------------------------性能分析函数------------------------------
def calculate_performance_metrics(test_results):
    """计算BiLSTM性能指标"""
    print("\n🔬 计算BiLSTM性能指标...")
    
    performance_metrics = {}
    model_results = test_results["BILSTM"]
    
    # 收集所有样本的预测结果
    all_true_labels = []
    all_fai_values = []
    all_fault_predictions = []
    
    for result in model_results:
        true_label = result.get('label', 0)
        fai_values = result.get('fai', [])
        fault_labels = result.get('fault_labels', [])
        thresholds = result.get('thresholds', {})
        threshold1 = thresholds.get('threshold1', 0.0)
        
        # 对于每个时间点
        for i, (fai_val, fault_pred) in enumerate(zip(fai_values, fault_labels)):
            # 正确的ROC逻辑：使用点级别的真实标签
            if true_label == 0:  # 正常样本
                point_true_label = 0  # 正常样本的所有点都是正常的
            else:  # 故障样本
                point_true_label = fault_pred  # 故障样本使用3点检测生成的伪标签
            
            all_true_labels.append(point_true_label)  # 使用点级别标签
            all_fai_values.append(fai_val)
            
            # 预测逻辑：基于fai阈值判断
            prediction = 1 if fai_val > threshold1 else 0
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
    
    performance_metrics["BILSTM"] = {
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
def create_roc_analysis(test_results, performance_metrics, save_path):
    """生成BiLSTM ROC曲线分析"""
    print("   📈 生成BiLSTM ROC曲线分析...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_large"], constrained_layout=True)
    
    # === 子图1: 连续阈值ROC曲线 ===
    ax1.set_title('(a) BiLSTM ROC Curve\n(Continuous Threshold Scan)')
    
    model_results = test_results["BILSTM"]
    
    # 收集所有fai值和真实标签，用于连续阈值ROC
    all_fai = []
    all_labels = []
    all_fault_labels = []
    
    for result in model_results:
        all_fai.extend(result['fai'])
        all_labels.extend([result['label']] * len(result['fai']))
        all_fault_labels.extend(result['fault_labels'])
    
    all_fai = np.array(all_fai)
    all_labels = np.array(all_labels)
    all_fault_labels = np.array(all_fault_labels)
    
    # 检查数据是否为空
    if len(all_fai) == 0:
        print("   ⚠️ 警告: 没有可用的FAI数据，跳过ROC曲线生成")
        # 创建一个空的ROC图
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('(a) BiLSTM ROC Curve\n(No Data)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.grid(True, alpha=0.3)
        return
    
    # 🔧 添加数据统计分析
    print(f"   📊 BiLSTM ROC曲线数据统计:")
    print(f"      总数据点: {len(all_fai)}")
    print(f"      正常样本点: {np.sum(all_labels == 0)}")
    print(f"      故障样本点: {np.sum(all_labels == 1)}")
    print(f"      故障标记点: {np.sum(all_fault_labels == 1)}")
    print(f"      FAI范围: [{np.min(all_fai):.6f}, {np.max(all_fai):.6f}]")
    
    # 🔧 改进阈值扫描策略：使用更合理的范围
    # 方法1：使用分位数范围避免极端值影响
    fai_p1 = np.percentile(all_fai, 1)   # 1%分位数
    fai_p99 = np.percentile(all_fai, 99) # 99%分位数
    fai_median = np.median(all_fai)
    fai_mean = np.mean(all_fai)
    
    print(f"   📊 FAI分布分析:")
    print(f"      1%分位数: {fai_p1:.6f}")
    print(f"      50%分位数(中位数): {fai_median:.6f}")
    print(f"      均值: {fai_mean:.6f}")
    print(f"      99%分位数: {fai_p99:.6f}")
    print(f"      最大值: {np.max(all_fai):.6f}")
    
    # 使用更智能的阈值范围：从最小值到99%分位数，避免极端值
    threshold_min = np.min(all_fai)
    threshold_max = fai_p99  # 使用99%分位数而不是最大值
    
    # 🔧 使用混合扫描策略：线性+对数尺度
    if threshold_max > threshold_min * 10:  # 如果范围较大，使用对数尺度
        # 方法1：对数尺度扫描（处理大范围的数据）
        log_min = np.log10(max(threshold_min, 1e-10))  # 避免log(0)
        log_max = np.log10(threshold_max)
        log_thresholds = np.logspace(log_min, log_max, 50)
        
        # 方法2：线性扫描（处理小范围的数据）
        linear_thresholds = np.linspace(threshold_min, min(threshold_max, fai_median*3), 50)
        
        # 合并并去重
        thresholds = np.unique(np.concatenate([linear_thresholds, log_thresholds]))
        thresholds = np.sort(thresholds)
        
        print(f"   📊 使用混合扫描策略 (线性+对数): {len(thresholds)}个阈值点")
    else:
        # 范围较小时使用线性扫描
        thresholds = np.linspace(threshold_min, threshold_max, 100)
        print(f"   📊 使用线性扫描策略: {len(thresholds)}个阈值点")
    
    print(f"   📊 阈值扫描范围: [{threshold_min:.6f}, {threshold_max:.6f}]")
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        
        for i, (fai_val, sample_label, fault_pred) in enumerate(zip(all_fai, all_labels, all_fault_labels)):
            # 🔧 修复ROC曲线逻辑：使用点级别的真实标签
            if sample_label == 0:  # 正常样本的所有点都是正常的
                point_true_label = 0
            else:  # 故障样本：使用故障检测算法的结果作为点级别真实标签
                point_true_label = fault_pred
            
            # 预测标签：简单基于FAI阈值
            predicted_label = 1 if fai_val > threshold else 0
            
            # 统计混淆矩阵
            if point_true_label == 0 and predicted_label == 0:
                tn += 1
            elif point_true_label == 0 and predicted_label == 1:
                fp += 1
            elif point_true_label == 1 and predicted_label == 0:
                fn += 1
            elif point_true_label == 1 and predicted_label == 1:
                tp += 1
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # 计算AUC - 需要确保fpr_list是单调递增的
    from sklearn.metrics import auc
    
    # 🔧 修复AUC计算：确保FPR单调递增
    combined = list(zip(fpr_list, tpr_list))
    combined.sort(key=lambda x: x[0])  # 按FPR排序
    fpr_sorted, tpr_sorted = zip(*combined)
    
    auc_score = auc(fpr_sorted, tpr_sorted)
    
    print(f"   📊 BiLSTM ROC曲线计算结果:")
    print(f"      阈值数量: {len(thresholds)}")
    print(f"      FPR范围: [{min(fpr_sorted):.3f}, {max(fpr_sorted):.3f}]")
    print(f"      TPR范围: [{min(tpr_sorted):.3f}, {max(tpr_sorted):.3f}]")
    print(f"      AUC得分: {auc_score:.6f}")
    
    # 绘制ROC曲线 - 使用排序后的数据
    ax1.plot(fpr_sorted, tpr_sorted, color='blue', linewidth=2,
            label=f'BiLSTM (AUC={auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 固定阈值工作点 ===
    ax2.set_title('(b) Working Point\n(Three-Level Alarm Threshold)')
    
    metrics = performance_metrics["BILSTM"]['classification_metrics']
    ax2.scatter(metrics['fpr'], metrics['tpr'], 
               s=200, color='blue', 
               label=f'BiLSTM\n(TPR={metrics["tpr"]:.3f}, FPR={metrics["fpr"]:.3f})',
               marker='o', edgecolors='black', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === 子图3: 性能指标展示 ===
    ax3.set_title('(c) BiLSTM Classification Metrics')
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_mapping = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall', 'F1-Score': 'f1_score', 'Specificity': 'specificity'}
    bilstm_values = [performance_metrics["BILSTM"]['classification_metrics'][metric_mapping[m]] 
                     for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.6
    
    bars = ax3.bar(x, bilstm_values, width, color='blue', alpha=0.7)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, bilstm_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # === 子图4: 样本级性能展示 ===
    ax4.set_title('(d) BiLSTM Sample-Level Performance')
    
    sample_metrics = ['Avg φ(Normal)', 'Avg φ(Fault)', 'Anomaly Rate(Normal)', 'Anomaly Rate(Fault)']
    bilstm_sample_values = [
        performance_metrics["BILSTM"]['sample_metrics']['avg_fai_normal'],
        performance_metrics["BILSTM"]['sample_metrics']['avg_fai_fault'],
        performance_metrics["BILSTM"]['sample_metrics']['avg_anomaly_ratio_normal'],
        performance_metrics["BILSTM"]['sample_metrics']['avg_anomaly_ratio_fault']
    ]
    
    x = np.arange(len(sample_metrics))
    bars = ax4.bar(x, bilstm_sample_values, width, color='blue', alpha=0.7)
    
    ax4.set_xlabel('Sample Metrics')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_metrics, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, bilstm_sample_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ BiLSTM ROC分析图保存至: {save_path}")

#----------------------------------------故障检测时序图------------------------------
def create_fault_detection_timeline(test_results, save_path):
    """生成BiLSTM故障检测时序图"""
    print("   📊 生成BiLSTM故障检测时序图...")
    
    # 选择一个故障样本进行可视化
    fault_sample_id = TEST_SAMPLES['fault'][0] if TEST_SAMPLES['fault'] else '335'  # 使用第一个故障样本
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    
    # 找到对应样本的结果
    sample_result = next((r for r in test_results["BILSTM"] if r.get('sample_id') == fault_sample_id), None)
    
    if sample_result is None:
        print(f"   ⚠️ 未找到样本 {fault_sample_id} 的结果，使用第一个可用结果")
        sample_result = test_results["BILSTM"][0] if test_results["BILSTM"] else None
    
    if sample_result is None:
        print("   ❌ 没有可用的测试结果")
        return
    
    fai_values = sample_result.get('fai', [])
    fault_labels = sample_result.get('fault_labels', [])
    thresholds = sample_result.get('thresholds', {})
    time_axis = np.arange(len(fai_values))
    
    # 子图1: 综合诊断指标时序
    ax1 = axes[0]
    ax1.plot(time_axis, fai_values, color='blue', linewidth=1, alpha=0.8,
           label='BiLSTM FAI')
    ax1.axhline(y=thresholds['threshold1'], color='orange', linestyle='--', alpha=0.7,
              label='Level 1 Threshold')
    ax1.axhline(y=thresholds['threshold2'], color='red', linestyle='--', alpha=0.7,
              label='Level 2 Threshold')
    ax1.axhline(y=thresholds['threshold3'], color='darkred', linestyle='--', alpha=0.7,
              label='Level 3 Threshold')
    
    ax1.set_ylabel('BiLSTM\nComprehensive Diagnostic Index φ')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'BiLSTM - Sample {fault_sample_id} (Fault Sample)')
    
    # 子图2: 故障检测结果
    ax2 = axes[1]
    
    # 将故障标签转换为可视化区域
    fault_regions = np.where(fault_labels == 1, 0.8, 0)
    ax2.fill_between(time_axis, fault_regions, 
                    alpha=0.6, color='blue',
                    label='BiLSTM Fault Detection')
    
    ax2.set_ylabel('Fault Detection Result')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('BiLSTM Fault Detection Result')
    
    # 子图3: 检测过程（兼容两种模式）
    ax3 = axes[2]
    detection_info = sample_result['detection_info']
    
    ax3.plot(time_axis, fai_values, 'b-', alpha=0.5, label='φ Index Value')
    
    # 3点检测模式
    # 标记触发点
    if detection_info.get('trigger_points'):
        ax3.scatter(detection_info['trigger_points'], 
                   [fai_values[i] for i in detection_info['trigger_points']],
                   color='orange', s=30, label='Trigger Points', alpha=0.8)
    
    ax3.set_ylabel('3-Point Detection\nProcess')
    ax3.set_title('3-Point Detection Process (BiLSTM)')
    
    # 标记故障区域
    if detection_info.get('marked_regions'):
        for i, region in enumerate(detection_info['marked_regions']):
            start, end = region['range']
            label = 'Marked Fault Region' if i == 0 else ""
            ax3.axvspan(start, end, alpha=0.2, color='red', label=label)
    
    ax3.set_xlabel('Time Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ BiLSTM时序图保存至: {save_path}")

#----------------------------------------性能指标雷达图------------------------------
def create_performance_radar(performance_metrics, save_path):
    """生成BiLSTM性能指标雷达图"""
    print("   🕸️ 生成BiLSTM性能指标雷达图...")
    
    # 定义雷达图指标
    radar_metrics = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-Score': 'f1_score',
        'Specificity': 'specificity',
        'Early Warning': 'tpr',  # 早期预警能力 (TPR)
        'False Alarm Control': 'fpr',  # 误报控制 (1-FPR)
        'Detection Stability': 'accuracy'  # 检测稳定性 (用准确率代表)
    }
    
    # 数据预处理：FPR需要转换为控制能力 (1-FPR)
    bilstm_values = []
    
    for metric_name, metric_key in radar_metrics.items():
        bilstm_val = performance_metrics['BILSTM']['classification_metrics'][metric_key]
        
        # 特殊处理：误报控制 = 1 - FPR
        if metric_name == 'False Alarm Control':
            bilstm_val = 1 - bilstm_val
            
        bilstm_values.append(bilstm_val)
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    bilstm_values += bilstm_values[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_medium"], subplot_kw=dict(projection='polar'), constrained_layout=True)
    
    # 绘制雷达图
    ax.plot(angles, bilstm_values, 'o-', linewidth=2, label='BiLSTM', color='blue')
    ax.fill(angles, bilstm_values, alpha=0.25, color='blue')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(radar_metrics.keys()))
    ax.set_ylim(0, 1)
    
    # 添加网格线
    ax.grid(True)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # 添加标题和图例
    plt.title('BiLSTM Performance Metrics Radar Chart', 
              pad=20, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 添加性能总结
    bilstm_avg = np.mean(bilstm_values[:-1])
    
    plt.figtext(0.02, 0.02, f'BiLSTM Overall Performance: {bilstm_avg:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ BiLSTM雷达图保存至: {save_path}")

#----------------------------------------3点检测过程可视化------------------------------
def create_three_point_visualization(test_results, save_path):
    """生成BiLSTM 3点检测过程可视化"""
    print("   🔍 生成BiLSTM 3点检测过程可视化...")
    
    # 选择一个故障样本进行详细分析
    fault_sample_id = TEST_SAMPLES['fault'][0] if TEST_SAMPLES['fault'] else '335'
    
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    
    # 使用GridSpec进行复杂布局
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # === 主图：3点检测过程时序图 ===
    ax_main = fig.add_subplot(gs[0, :])
    
    # 选择BiLSTM结果进行可视化
    bilstm_result = next((r for r in test_results['BILSTM'] if r.get('sample_id') == fault_sample_id), None)
    
    if bilstm_result is None:
        print(f"   ⚠️ 未找到样本 {fault_sample_id} 的结果，使用第一个可用结果")
        bilstm_result = test_results['BILSTM'][0] if test_results['BILSTM'] else None
    
    if bilstm_result is None:
        print("   ❌ 没有可用的测试结果")
        return
    
    fai_values = bilstm_result.get('fai', [])
    detection_info = bilstm_result.get('detection_info', {})
    thresholds = bilstm_result.get('thresholds', {})
    threshold1 = thresholds.get('threshold1', 0.0)
    
    time_axis = np.arange(len(fai_values))
    
    # 绘制FAI时序
    ax_main.plot(time_axis, fai_values, 'b-', linewidth=1.5, alpha=0.8, label='Comprehensive Diagnostic Index φ(FAI)')
    ax_main.axhline(y=threshold1, color='red', linestyle='--', alpha=0.7, label='Level 1 Threshold')
    
    # 3点检测模式
    # 标记触发点
    if detection_info.get('trigger_points'):
        trigger_points = detection_info['trigger_points']
        ax_main.scatter(trigger_points, [fai_values[i] for i in trigger_points],
                       color='orange', s=40, alpha=0.8, label=f'Trigger: {len(trigger_points)} Points',
                       marker='o', zorder=5)
    
    # 标记故障区域
    for i, region in enumerate(detection_info['marked_regions']):
        start, end = region['range']
        label = 'Marked: Fault Region' if i == 0 else ""
        ax_main.axvspan(start, end, alpha=0.2, color='red', label=label)
    
    ax_main.set_xlabel('Time Step')
    ax_main.set_ylabel('Comprehensive Diagnostic Index φ')
    
    # 设置标题
    title = f'BiLSTM Three-Point Fault Detection Process - Sample {fault_sample_id}'
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    # === 子图1：检测窗口统计 ===
    ax1 = fig.add_subplot(gs[1, 0])
    
    detection_stats = detection_info.get('detection_stats', {})
    detection_data = [
        detection_stats.get('total_trigger_points', 0),
        detection_stats.get('total_marked_regions', 0), 
        detection_stats.get('total_fault_points', 0)
    ]
    detection_labels = ['Trigger Points', 'Marked Regions', 'Fault Points']
    colors1 = ['orange', 'red', 'darkred']
    
    bars1 = ax1.bar(detection_labels, detection_data, color=colors1, alpha=0.7)
    ax1.set_title('Detection Statistics')
    ax1.set_ylabel('Count')
    
    # 添加数值标签
    for bar, value in zip(bars1, detection_data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom')
    
    # === 子图2：检测模式配置 ===
    ax2 = fig.add_subplot(gs[1, 1])
    
    # 3点检测模式
    mode_params = [3, 2, 1]  # 3点区域, 2个邻居, 1个中心点
    mode_labels = ['Marked Region\n(3 Points)', 'Neighbor Check\n(2 Points)', 'Center Point\n(1 Point)']
    colors2 = ['lightblue', 'lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax2.pie(mode_params, labels=mode_labels, colors=colors2,
                                      autopct='%1.0f', startangle=90)
    ax2.set_title('3-Point Detection\nParameter Config')
    
    # === 子图3：检测详情 ===
    ax3 = fig.add_subplot(gs[1, 2])
    
    # 3点检测模式：显示触发点的FAI值分布
    if detection_info.get('trigger_points'):
        trigger_points = detection_info['trigger_points']
        trigger_fai_values = [fai_values[i] for i in trigger_points]
        
        bars3 = ax3.bar(range(len(trigger_fai_values)), trigger_fai_values, 
                       color='orange', alpha=0.7)
        ax3.axhline(y=threshold1, color='red', linestyle='--', 
                   alpha=0.7, label='Level 1 Threshold')
        ax3.set_title('Trigger Point FAI Values')
        ax3.set_xlabel('Trigger Point')
        ax3.set_ylabel('FAI Value')
        ax3.set_xticks(range(len(trigger_fai_values)))
        ax3.set_xticklabels([f'T{i+1}' for i in range(len(trigger_fai_values))])
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Trigger Points', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Trigger Point FAI Values')
    
    # === 子图4：BiLSTM性能 ===
    ax4 = fig.add_subplot(gs[1, 3])
    
    sample_result = next((r for r in test_results['BILSTM'] if r.get('sample_id') == fault_sample_id), None)
    if sample_result is None:
        sample_result = test_results['BILSTM'][0] if test_results['BILSTM'] else None
    
    if sample_result is None:
        fault_ratio = 0.0
    else:
        detection_info = sample_result.get('detection_info', {})
        detection_stats = detection_info.get('detection_stats', {})
        fault_ratio = detection_stats.get('fault_ratio', 0.0)
    
    bars4 = ax4.bar(['BiLSTM'], [fault_ratio], color='blue', alpha=0.7)
    ax4.set_title('BiLSTM\n(Fault Detection Ratio)')
    ax4.set_ylabel('Fault Ratio')
    
    for bar, value in zip(bars4, [fault_ratio]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # === 底部：过程说明 ===
    process_text = """
    BiLSTM Three-Point Detection Process:
    
    1. Time Series Scan: Check φ(FAI) values point by point against threshold
    2. Trigger Condition: Triggered when current point and neighbors exceed threshold
    3. Region Marking: Mark current point and ±1 neighboring points (3 total) as fault
    
    Advantage: Precise detection with minimal false positives and efficient computation
    """
    
    fig.text(0.02, 0.02, process_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ BiLSTM 3点检测过程图保存至: {save_path}")

#----------------------------------------结果保存函数------------------------------
def save_test_results(test_results, performance_metrics):
    """保存BiLSTM测试结果"""
    print("\n💾 保存BiLSTM测试结果...")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/mnt/bz25t/bzhy/datasave/BILSTM/test_results/bilstm_test_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{result_dir}/detailed_results", exist_ok=True)
    
    # 1. 保存性能指标JSON
    performance_file = f"{result_dir}/bilstm_performance_metrics.json"
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ BiLSTM性能指标保存至: {performance_file}")
    
    # 2. 保存详细结果
    detail_file = f"{result_dir}/detailed_results/bilstm_detailed_results.pkl"
    with open(detail_file, 'wb') as f:
        pickle.dump(test_results["BILSTM"], f)
    print(f"   ✅ BiLSTM详细结果保存至: {detail_file}")
    
    # 3. 保存元数据
    metadata_file = f"{result_dir}/detailed_results/bilstm_test_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(test_results['metadata'], f, indent=2, ensure_ascii=False)
    print(f"   ✅ BiLSTM测试元数据保存至: {metadata_file}")
    
    # 4. 创建Excel总结报告
    summary_file = f"{result_dir}/detailed_results/bilstm_summary.xlsx"
    
    with pd.ExcelWriter(summary_file) as writer:
        # BiLSTM性能表
        metrics = performance_metrics["BILSTM"]['classification_metrics']
        confusion = performance_metrics["BILSTM"]['confusion_matrix']
        sample_metrics = performance_metrics["BILSTM"]['sample_metrics']
        
        performance_data = [{
            'Model': 'BILSTM',
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
        }]
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(writer, sheet_name='BILSTM_Performance', index=False)
        
        # 样本详情表
        sample_details = []
        for result in test_results["BILSTM"]:
            # 安全获取detection_info和detection_stats
            detection_info = result.get('detection_info', {})
            detection_stats = detection_info.get('detection_stats', {})
            performance_metrics = result.get('performance_metrics', {})
            
            sample_details.append({
                'Sample_ID': result.get('sample_id', 'Unknown'),
                'True_Label': 'Fault' if result.get('label', 0) == 1 else 'Normal',
                'FAI_Mean': performance_metrics.get('fai_mean', 0.0),
                'FAI_Std': performance_metrics.get('fai_std', 0.0),
                'FAI_Max': performance_metrics.get('fai_max', 0.0),
                'Anomaly_Ratio': performance_metrics.get('anomaly_ratio', 0.0),
                'Fault_Detection_Ratio': detection_stats.get('fault_ratio', 0.0),
                'Trigger_Points_Found': detection_stats.get('total_trigger_points', 0),
                'Marked_Regions': detection_stats.get('total_marked_regions', 0)
            })
        
        sample_df = pd.DataFrame(sample_details)
        sample_df.to_excel(writer, sheet_name='Sample_Details', index=False)
    
    print(f"   ✅ BiLSTM Excel总结报告保存至: {summary_file}")
    
    return result_dir

#----------------------------------------主执行流程------------------------------
print("\n🎨 生成可视化分析...")

# 计算性能指标
performance_metrics = calculate_performance_metrics(test_results)

# 保存测试结果和生成可视化
result_dir = save_test_results(test_results, performance_metrics)

# 生成可视化图表
print("\n🎨 生成BiLSTM可视化分析...")

# 生成ROC分析图
create_roc_analysis(test_results, performance_metrics, f"{result_dir}/visualizations/bilstm_roc_analysis.png")

# 生成故障检测时序图
create_fault_detection_timeline(test_results, f"{result_dir}/visualizations/bilstm_fault_detection_timeline.png")

# 生成性能雷达图
create_performance_radar(performance_metrics, f"{result_dir}/visualizations/bilstm_performance_radar.png")

# 生成3点检测过程图
create_three_point_visualization(test_results, f"{result_dir}/visualizations/bilstm_three_point_process.png")

#----------------------------------------最终总结------------------------------
print("\n" + "="*80)
print("🎉 BiLSTM模型测试完成！")
print("="*80)

print(f"\n📊 测试结果总结:")
print(f"   • 测试样本: {len(ALL_TEST_SAMPLES)} 个 (正常: {len(TEST_SAMPLES['normal'])}, 故障: {len(TEST_SAMPLES['fault'])})")
print(f"   • 模型类型: BiLSTM")
print(f"   • 检测模式: {DETECTION_MODES[CURRENT_DETECTION_MODE]['name']}")
print(f"   • 3点检测参数: {THREE_POINT_CONFIG}")
print(f"   • 3点检测模式: 当前点+前后相邻点高于阈值时，标记3点区域")

print(f"\n🔬 BiLSTM性能:")
metrics = performance_metrics["BILSTM"]['classification_metrics']
print(f"   准确率: {metrics['accuracy']:.3f}")
print(f"   精确率: {metrics['precision']:.3f}")
print(f"   召回率: {metrics['recall']:.3f}")
print(f"   F1分数: {metrics['f1_score']:.3f}")
print(f"   TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}")

print(f"\n📁 结果文件:")
print(f"   • 结果目录: {result_dir}")
print(f"   • 可视化图表: {result_dir}/visualizations")
print(f"     - ROC分析图: bilstm_roc_analysis.png")
print(f"     - 故障检测时序图: bilstm_fault_detection_timeline.png") 
print(f"     - 性能雷达图: bilstm_performance_radar.png")
print(f"     - 3点检测过程图: bilstm_three_point_process.png")
print(f"   • 性能指标: bilstm_performance_metrics.json")
print(f"   • Excel报告: bilstm_summary.xlsx")

# 综合性能评估
bilstm_score = np.mean(list(performance_metrics["BILSTM"]['classification_metrics'].values()))

print(f"\n🏆 BiLSTM综合性能评估:")
print(f"   综合得分: {bilstm_score:.3f}")

print("\n" + "="*80)
print("BiLSTM测试完成！请查看生成的可视化图表和分析报告。")
print("="*80)