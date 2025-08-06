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
import platform

def setup_chinese_fonts():
    """配置中文字体显示"""
    system = platform.system()
    
    # 根据操作系统选择字体
    if system == "Windows":
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == "Linux":
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:
        chinese_fonts = ['DejaVu Sans', 'Arial Unicode MS']
    
    # 查找可用的中文字体
    available_font = None
    for font in chinese_fonts:
        try:
            # 检查字体是否存在
            font_path = fm.findfont(font)
            if font_path != fm.rcParams['font.sans-serif'][0]:
                available_font = font
                break
        except:
            continue
    
    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font] + plt.rcParams['font.sans-serif']
        print(f"✅ 使用中文字体: {available_font}")
    else:
        print("⚠️ 未找到合适的中文字体，尝试使用默认配置")
        # 使用更通用的字体配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 执行字体配置
setup_chinese_fonts()

# 清理字体缓存并强制刷新
try:
    fm._rebuild()
    print("✅ 字体缓存已清理并重建")
except:
    print("⚠️ 字体缓存清理失败，继续使用当前配置")

#----------------------------------------测试配置------------------------------
print("="*60)
print("🔬 电池故障诊断系统 - Transformer模型测试（混合反馈版本）")
print("="*60)

TEST_MODE = "TRANSFORMER_ONLY"  # 固定为Transformer单模型测试

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
        
        # 指定测试样本：正常样本10,11 和故障样本335,336
        test_normal_samples = ['10', '11']  # 正常样本
        test_fault_samples = ['335', '336']  # 故障样本
        
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
            'normal': ['10', '11'],
            'fault': ['335', '336']
        }

TEST_SAMPLES = load_test_samples()
ALL_TEST_SAMPLES = TEST_SAMPLES['normal'] + TEST_SAMPLES['fault']

# 模型路径配置 (从混合反馈训练结果加载)
MODEL_PATHS = {
    "TRANSFORMER": {
        "transformer_model": "/mnt/bz25t/bzhy/datasave/Transformer/models/transformer_model_hybrid_feedback.pth",
        "net_model": "/mnt/bz25t/bzhy/datasave/Transformer/models/net_model_hybrid_feedback.pth", 
        "netx_model": "/mnt/bz25t/bzhy/datasave/Transformer/models/netx_model_hybrid_feedback.pth"
    }
}

# 检测模式配置
DETECTION_MODES = {
    "three_window": {
        "name": "三窗口检测模式",
        "description": "基于FAI的三窗口故障检测机制（检测->验证->标记）",
        "function": "three_window_fault_detection"
    },
    "five_point": {
        "name": "5点检测模式（原版）", 
        "description": "对于故障样本，如果某点高于阈值且前后相邻点也高于阈值，则标记该点及前后2个点（共5个点）",
        "function": "five_point_fault_detection"
    },
    "five_point_improved": {
        "name": "5点检测模式（改进版）",
        "description": "改进的5点检测：严格的触发条件 + 分级标记范围 + 有效降噪机制",
        "function": "five_point_fault_detection"
    }
}

# 当前使用的检测模式
CURRENT_DETECTION_MODE = "five_point_improved"  # 使用改进的5点检测模式

# 基于FAI的三窗口检测配置 (与BiLSTM测试脚本保持一致)
WINDOW_CONFIG = {
    "detection_window": 25,      # 检测窗口：25个采样点 (12.5分钟)
    "verification_window": 15,   # 验证窗口：15个采样点 (7.5分钟)
    "marking_window": 10,        # 标记窗口：10个采样点 (5分钟)
    "verification_threshold": 0.6 # 验证窗口内FAI异常比例阈值 (60%)
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
if CURRENT_DETECTION_MODE == "three_window":
    print(f"   三窗口参数: {WINDOW_CONFIG}")
else:
    print(f"   5点检测模式: 当前点+前后相邻点高于阈值时，标记5点区域")

#----------------------------------------模型文件检查------------------------------
def check_model_files():
    """检查Transformer模型文件"""
    print("\n🔍 检查Transformer模型文件...")
    
    missing_files = []
    paths = MODEL_PATHS["TRANSFORMER"]
    
    # 检查主模型文件
    for key, path in paths.items():
        if not os.path.exists(path):
            missing_files.append(f"TRANSFORMER: {path}")
            print(f"   ❌ 缺失: {path}")
        else:
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   ✅ 存在: {path} ({file_size:.1f}MB)")
    
    # 检查PCA参数文件 (从混合反馈训练结果加载)
    pca_params_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/pca_params_hybrid_feedback.pkl"
    if not os.path.exists(pca_params_path):
        print(f"   ⚠️  PCA参数pickle文件不存在: {pca_params_path}")
        print(f"   🔍 检查备选的npy文件...")
        
        # 检查备选的npy文件
        pca_npy_files = [
            "/mnt/bz25t/bzhy/datasave/Transformer/models/v_I_hybrid_feedback.npy",
            "/mnt/bz25t/bzhy/datasave/Transformer/models/data_mean_hybrid_feedback.npy",
            "/mnt/bz25t/bzhy/datasave/Transformer/models/data_std_hybrid_feedback.npy",
            "/mnt/bz25t/bzhy/datasave/Transformer/models/T_99_limit_hybrid_feedback.npy"
        ]
        
        npy_files_exist = 0
        for npy_file in pca_npy_files:
            if os.path.exists(npy_file):
                npy_files_exist += 1
                file_size = os.path.getsize(npy_file) / (1024 * 1024)
                print(f"   ✅ 存在: {npy_file} ({file_size:.1f}MB)")
            else:
                print(f"   ❌ 缺失: {npy_file}")
        
        if npy_files_exist >= 3:
            print(f"   ✅ 发现{npy_files_exist}个npy文件，可以重建PCA参数")
        else:
            missing_files.append(f"PCA_PARAMS: 缺少足够的npy文件")
    else:
        file_size = os.path.getsize(pca_params_path) / (1024 * 1024)  # MB
        print(f"   ✅ 存在: {pca_params_path} ({file_size:.1f}MB)")
    
    if missing_files:
        print(f"\n❌ 缺失 {len(missing_files)} 个模型文件:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 解决方案:")
        print("   1. 确保已运行Transformer训练脚本")
        print("   2. 检查模型文件路径是否正确")
        print("   3. 检查文件权限")
        raise FileNotFoundError("请先运行Transformer训练脚本生成所需模型文件")
    
    print("✅ Transformer模型文件检查通过")
    return True

# 执行模型文件检查
check_model_files()

#----------------------------------------三窗口故障检测机制------------------------------
def three_window_fault_detection(fai_values, threshold1, sample_id, config=None):
    """
    基于FAI的三窗口故障检测机制
    
    原理：
    1. 检测窗口：基于FAI统计特性识别异常点
    2. 验证窗口：确认FAI异常的持续性，排除随机波动
    3. 标记窗口：考虑故障的前后影响范围
    
    Args:
        fai_values: FAI序列（综合故障指标）
        threshold1: FAI阈值
        sample_id: 样本ID（用于记录）
        config: 窗口配置,如果为None则使用默认配置
    
    Returns:
        fault_labels: 故障标签序列 (0=正常, 1=故障)
        detection_info: 检测过程详细信息
    """
    # 获取窗口配置
    if config is None:
        config = WINDOW_CONFIG
        
    detection_window = config["detection_window"]
    verification_window = config["verification_window"]
    marking_window = config["marking_window"]
    verification_threshold = config["verification_threshold"]
    
    # 初始化输出
    fault_labels = np.zeros(len(fai_values), dtype=int)
    detection_info = {
        'candidate_points': [],    # 候选故障点
        'verified_points': [],     # 已验证的故障点
        'marked_regions': [],      # 标记的故障区域
        'window_stats': {},        # 窗口统计信息
        'fai_stats': {            # FAI统计信息
            'mean': np.mean(fai_values),
            'std': np.std(fai_values),
            'max': np.max(fai_values),
            'min': np.min(fai_values)
        }
    }
    
    # 阶段1：检测窗口 - 基于FAI统计特性识别异常点
    candidate_points = []
    for i in range(len(fai_values)):
        if fai_values[i] > threshold1:
            candidate_points.append(i)
    
    detection_info['candidate_points'] = candidate_points
    
    if len(candidate_points) == 0:
        # 没有候选点，直接返回
        return fault_labels, detection_info
    
    # 阶段2：验证窗口 - 确认FAI异常的持续性
    verified_points = []
    for candidate in candidate_points:
        # 定义验证窗口范围（前后各半个窗口）
        start_verify = max(0, candidate - verification_window//2)
        end_verify = min(len(fai_values), candidate + verification_window//2)
        verify_data = fai_values[start_verify:end_verify]
        
        # 计算FAI异常比例
        continuous_ratio = np.sum(verify_data > threshold1) / len(verify_data)
        
        # 计算FAI在验证窗口的统计特性
        window_stats = {
            'mean': np.mean(verify_data),
            'std': np.std(verify_data),
            'max': np.max(verify_data),
            'min': np.min(verify_data),
            'duration': end_verify - start_verify
        }
        
        # 基于verification_threshold验证持续性
        if continuous_ratio >= verification_threshold:
            verified_points.append({
                'point': candidate,
                'continuous_ratio': continuous_ratio,
                'verify_range': (start_verify, end_verify),
                'window_stats': window_stats
            })
    
    detection_info['verified_points'] = verified_points
    
    # 阶段3：标记窗口 - 考虑故障的影响范围
    marked_regions = []
    for verified in verified_points:
        candidate = verified['point']
        
        # 定义对称的标记窗口范围
        start_mark = max(0, candidate - marking_window)
        end_mark = min(len(fai_values), candidate + marking_window)
        
        # 提取标记区域的FAI特征
        mark_data = fai_values[start_mark:end_mark]
        region_stats = {
            'mean_fai': np.mean(mark_data),
            'max_fai': np.max(mark_data),
            'std_fai': np.std(mark_data),
            'duration': end_mark - start_mark
        }
        
        # 标记故障区域
        fault_labels[start_mark:end_mark] = 1
        
        marked_regions.append({
            'center': candidate,
            'range': (start_mark, end_mark),
            'length': end_mark - start_mark,
            'region_stats': region_stats
        })
    
    detection_info['marked_regions'] = marked_regions
    
    # 完整的统计信息
    detection_info['window_stats'] = {
        'total_candidates': len(candidate_points),
        'verified_candidates': len(verified_points),
        'total_fault_points': np.sum(fault_labels),
        'fault_ratio': np.sum(fault_labels) / len(fault_labels),
        'mean_continuous_ratio': np.mean([v['continuous_ratio'] for v in verified_points]) if verified_points else 0,
        'mean_region_length': np.mean([m['length'] for m in marked_regions]) if marked_regions else 0
    }
    
    return fault_labels, detection_info

def five_point_fault_detection(fai_values, threshold1, sample_id, config=None):
    """
    改进的5点故障检测机制：增强连续性检测和降噪能力
    
    设计原理：
    1. 严格的触发条件：要求中心点及其邻域满足更严格的一致性
    2. 合理的标记范围：根据故障级别标记不同大小的区域
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
    # 初始化输出
    fault_labels = np.zeros(len(fai_values), dtype=int)
    detection_info = {
        'trigger_points': [],      # 触发5点检测的点
        'marked_regions': [],      # 标记的5点区域
        'detection_stats': {},     # 检测统计信息
        'fai_stats': {            # FAI统计信息
            'mean': np.mean(fai_values),
            'std': np.std(fai_values),
            'max': np.max(fai_values),
            'min': np.min(fai_values)
        }
    }
    
    # 检查是否为故障样本（基于样本ID）
    is_fault_sample = sample_id in TEST_SAMPLES['fault']
    
    # 调试信息
    print(f"   样本{sample_id}: 类型={type(sample_id)}, 故障样本列表={TEST_SAMPLES['fault']}, 是故障样本={is_fault_sample}")
    
    if not is_fault_sample:
        # 正常样本直接返回全0标签
        print(f"   → 样本{sample_id}为正常样本，返回全0标签")
        detection_info['detection_stats'] = {
            'total_trigger_points': 0,
            'total_marked_regions': 0,
            'total_fault_points': 0,
            'fault_ratio': 0.0,
            'detection_mode': 'normal_sample'
        }
        # 为兼容性添加空字段
        detection_info['trigger_points'] = []
        detection_info['marked_regions'] = []
        detection_info['candidate_points'] = []
        detection_info['verified_points'] = []
        
        # 确保fault_labels确实是全0
        fault_labels.fill(0)
        print(f"   → fault_labels总和: {np.sum(fault_labels)} (应该为0)")
        return fault_labels, detection_info
    
    # 获取多级阈值（严格按照源代码Test_.py的方式）
    if config and 'threshold2' in config and 'threshold3' in config:
        threshold2 = config['threshold2']
        threshold3 = config['threshold3']
    else:
        # 按照源代码Test_.py的阈值计算方式
        nm = 3000
        mm = len(fai_values)
        
        if mm > nm:
            # 使用后半段数据计算阈值（与源代码一致）
            baseline_fai = fai_values[nm:mm]
            mean_fai = np.mean(baseline_fai)
            std_fai = np.std(baseline_fai)
            
            threshold1_calc = mean_fai + 3 * std_fai      # 对应源代码threshold1
            threshold2 = mean_fai + 4.5 * std_fai        # 对应源代码threshold2  
            threshold3 = mean_fai + 6 * std_fai          # 对应源代码threshold3
            
            # 验证threshold1是否与传入的一致（调试用）
            print(f"   源代码阈值计算: T1={threshold1_calc:.4f}(传入{threshold1:.4f}), T2={threshold2:.4f}, T3={threshold3:.4f}")
        else:
            # 数据太短，使用全部数据
            mean_fai = np.mean(fai_values)
            std_fai = np.std(fai_values)
            
            threshold1_calc = mean_fai + 3 * std_fai
            threshold2 = mean_fai + 4.5 * std_fai
            threshold3 = mean_fai + 6 * std_fai
            
            print(f"   短数据阈值计算: T1={threshold1_calc:.4f}(传入{threshold1:.4f}), T2={threshold2:.4f}, T3={threshold3:.4f}")
    
    # 故障样本：实施改进的多级5点检测
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
    
    # 三级分级检测实现
    triggers = []
    for i in range(2, len(fai_values) - 2):
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
    
    # 为兼容三窗口检测模式的可视化代码，添加空的兼容字段
    detection_info['candidate_points'] = []  # 5点检测模式中不使用，但为兼容性保留
    detection_info['verified_points'] = []   # 5点检测模式中不使用，但为兼容性保留
    
    # 统计信息（分级检测）
    fault_count = np.sum(fault_labels > 0)  # 任何级别都算故障
    level1_count = np.sum(fault_labels == 1)
    level2_count = np.sum(fault_labels == 2)
    level3_count = np.sum(fault_labels == 3)
    
    detection_info['detection_stats'] = {
        'total_trigger_points': len(trigger_points),
        'total_marked_regions': len(marked_regions),
        'total_fault_points': fault_count,
        'fault_ratio': fault_count / len(fault_labels),
        'detection_mode': 'hierarchical_three_level',
        'level_statistics': {
            'level_1_points': level1_count,
            'level_2_points': level2_count,
            'level_3_points': level3_count,
            'level_1_triggers': level_counts[1],
            'level_2_triggers': level_counts[2],
            'level_3_triggers': level_counts[3]
        },
        'mean_region_length': np.mean([m['length'] for m in marked_regions]) if marked_regions else 0,
        'mean_trigger_fai': np.mean([m['trigger_values']['center'] for m in marked_regions]) if marked_regions else 0,
        'strategy_used': 'strategy_4_hierarchical_detection',
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
            'marking_range': 2,                      # 标记±2个点
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
    """加载Transformer模型"""
    models = {}
    
    print("🔧 开始加载Transformer模型...")
    
    # 加载Transformer模型
    from Train_Transformer_HybridFeedback import TransformerPredictor
    models['transformer'] = TransformerPredictor().to(device)
    
    # 使用安全加载函数
    if not safe_load_model(models['transformer'], 
                          MODEL_PATHS["TRANSFORMER"]["transformer_model"], 
                          "Transformer"):
        raise RuntimeError("Transformer模型加载失败")
    
    # 加载MC-AE模型
    models['net'] = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                              activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    models['netx'] = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, 
                               activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
    
    # 使用安全加载函数
    if not safe_load_model(models['net'], 
                          MODEL_PATHS["TRANSFORMER"]["net_model"], 
                          "MC-AE1"):
        raise RuntimeError("MC-AE1模型加载失败")
    
    if not safe_load_model(models['netx'], 
                          MODEL_PATHS["TRANSFORMER"]["netx_model"], 
                          "MC-AE2"):
        raise RuntimeError("MC-AE2模型加载失败")
    
    # 加载PCA参数 (从混合反馈训练结果加载)
    pca_params_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/pca_params_hybrid_feedback.pkl"
    try:
        with open(pca_params_path, 'rb') as f:
            models['pca_params'] = pickle.load(f)
        print(f"✅ PCA参数已加载: {pca_params_path}")
    except Exception as e:
        print(f"❌ 加载PCA参数失败: {e}")
        print("🔄 尝试从单独的npy文件重建PCA参数...")
        try:
            # 从训练脚本保存的npy文件重建PCA参数
            pca_base_path = "/mnt/bz25t/bzhy/datasave/Transformer/models/"
            models['pca_params'] = {
                'v_I': np.load(f"{pca_base_path}v_I_hybrid_feedback.npy"),
                'v': np.load(f"{pca_base_path}v_hybrid_feedback.npy"),
                'v_ratio': np.load(f"{pca_base_path}v_ratio_hybrid_feedback.npy"),
                'p_k': np.load(f"{pca_base_path}p_k_hybrid_feedback.npy"),
                'data_mean': np.load(f"{pca_base_path}data_mean_hybrid_feedback.npy"),
                'data_std': np.load(f"{pca_base_path}data_std_hybrid_feedback.npy"),
                'T_95_limit': np.load(f"{pca_base_path}T_95_limit_hybrid_feedback.npy"),
                'T_99_limit': np.load(f"{pca_base_path}T_99_limit_hybrid_feedback.npy"),
                'SPE_95_limit': np.load(f"{pca_base_path}SPE_95_limit_hybrid_feedback.npy"),
                'SPE_99_limit': np.load(f"{pca_base_path}SPE_99_limit_hybrid_feedback.npy"),
                'P': np.load(f"{pca_base_path}P_hybrid_feedback.npy"),
                'k': np.load(f"{pca_base_path}k_hybrid_feedback.npy"),
                'P_t': np.load(f"{pca_base_path}P_t_hybrid_feedback.npy"),
                'X': np.load(f"{pca_base_path}X_hybrid_feedback.npy"),
                'data_nor': np.load(f"{pca_base_path}data_nor_hybrid_feedback.npy")
            }
            print(f"✅ PCA参数从npy文件重建成功")
        except Exception as e2:
            print(f"❌ PCA参数重建也失败: {e2}")
        raise RuntimeError("PCA参数加载失败")
    
    return models

#----------------------------------------单样本处理函数------------------------------
def process_single_sample(sample_id, models, config=None):
    """
    处理单个测试样本
    
    Args:
        sample_id: 样本ID
        models: 加载的模型
        config: 窗口配置,如果为None则使用默认配置
    """
    
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
    
    # 计算阈值 - 按源代码方式，每个样本使用自己的数据
    nm = 3000  # 固定值，与源代码一致
    mm = len(fai)  # 数据总长度
    
    # 确保数据长度足够
    if mm > nm:
        # 使用后半段数据计算阈值（源代码逻辑）
        threshold1 = np.mean(fai[nm:mm]) + 3*np.std(fai[nm:mm])
        threshold2 = np.mean(fai[nm:mm]) + 4.5*np.std(fai[nm:mm])
        threshold3 = np.mean(fai[nm:mm]) + 6*np.std(fai[nm:mm])
    else:
        # 数据太短，使用全部数据
        print(f"   ⚠️ 样本{sample_id}数据长度({mm})不足3000，使用全部数据计算阈值")
        threshold1 = np.mean(fai) + 3*np.std(fai)
        threshold2 = np.mean(fai) + 4.5*np.std(fai)
        threshold3 = np.mean(fai) + 6*np.std(fai)
    
    # 根据检测模式选择检测函数
    if CURRENT_DETECTION_MODE == "five_point":
        fault_labels, detection_info = five_point_fault_detection(fai, threshold1, sample_id, config)
    else:
        fault_labels, detection_info = three_window_fault_detection(fai, threshold1, sample_id, config)
    
    # 构建结果
    sample_result = {
        'sample_id': sample_id,
        'model_type': 'TRANSFORMER',
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
        "TRANSFORMER": [],
        "metadata": {
            "test_samples": TEST_SAMPLES,
            "window_config": WINDOW_CONFIG,
            "detection_modes": DETECTION_MODES,
            "current_mode": CURRENT_DETECTION_MODE,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    print(f"\n🚀 开始Transformer模型测试...")
    print(f"检测模式: {DETECTION_MODES[CURRENT_DETECTION_MODE]['name']}")
    print(f"检测描述: {DETECTION_MODES[CURRENT_DETECTION_MODE]['description']}")
    print(f"总共需要处理: {len(ALL_TEST_SAMPLES)} 个样本")
    
    # 加载模型
    print(f"\n{'='*20} 加载模型 {'='*20}")
    models = load_models()
    print(f"✅ Transformer 模型加载完成")
    
    # 单配置测试 (与BiLSTM保持一致)
    print(f"\n{'='*20} 测试 Transformer 模型 {'='*20}")
        if CURRENT_DETECTION_MODE == "three_window":
        print(f"   检测窗口: {WINDOW_CONFIG['detection_window']}")
        print(f"   验证窗口: {WINDOW_CONFIG['verification_window']}")
        print(f"   标记窗口: {WINDOW_CONFIG['marking_window']}")
        print(f"   验证阈值: {WINDOW_CONFIG['verification_threshold']}")
        else:
            print(f"   5点检测模式: 当前点+前后相邻点高于阈值时，标记5点区域")
        
    with tqdm(total=len(ALL_TEST_SAMPLES), desc="Transformer测试进度",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:
            
            for sample_id in ALL_TEST_SAMPLES:
            pbar.set_description(f"Transformer-样本{sample_id}")
            
            try:
                # 使用窗口配置处理样本
                sample_result = process_single_sample(sample_id, models, WINDOW_CONFIG)
                test_results["TRANSFORMER"].append(sample_result)
                    
                    # 输出简要结果
                    metrics = sample_result.get('performance_metrics', {})
                    detection_info = sample_result.get('detection_info', {})
                    
                    if CURRENT_DETECTION_MODE == "three_window":
                        window_stats = detection_info.get('window_stats', {})
                        detection_ratio = window_stats.get('fault_ratio', 0.0)
                    else:
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
    
    print(f"\n✅ Transformer测试完成!")
    print(f"   Transformer: 成功处理 {len(test_results['TRANSFORMER'])} 个样本")
    
    return test_results

#----------------------------------------性能分析函数------------------------------
def calculate_performance_metrics(test_results):
    """计算Transformer性能指标"""
    print("\n🔬 计算Transformer性能指标...")
    
    performance_metrics = {}
    model_results = test_results["TRANSFORMER"]
    
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
            all_true_labels.append(true_label)
            all_fai_values.append(fai_val)
            
            # 修正后的ROC逻辑：
            if true_label == 0:  # 正常样本
                # 正常样本中：综合诊断值 > 阈值1 就是FP，否则就是TN
                prediction = 1 if fai_val > threshold1 else 0
            else:  # 故障样本
                # 故障样本中：需要综合诊断值 > 阈值1 且 三窗口确认为故障 才是TP
                if fai_val > threshold1 and fault_pred == 1:
                    prediction = 1  # TP
                else:
                    prediction = 0  # FN (包括：fai_val <= threshold1 或 fault_pred == 0)
            
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
    
    performance_metrics["TRANSFORMER"] = {
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

# 执行主测试流程
test_results = main_test_process()

#----------------------------------------ROC曲线对比------------------------------
def create_roc_analysis(test_results, performance_metrics, save_path):
    """生成Transformer ROC曲线分析"""
    print("   📈 生成Transformer ROC曲线分析...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_large"])
    
    # === 子图1: 连续阈值ROC曲线 ===
    ax1.set_title('(a) Transformer ROC Curve\n(Continuous Threshold Scan)')
    
    # 使用Transformer结果
    model_results = test_results["TRANSFORMER"]
    if not model_results:
        print("   ⚠️ 警告: 没有可用的Transformer结果")
        return
    
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
        ax1.text(0.5, 0.5, '无数据', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('(a) Transformer ROC Curve\n(无数据)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.grid(True, alpha=0.3)
        return
    
    # 生成连续阈值范围
    thresholds = np.linspace(np.min(all_fai), np.max(all_fai), 100)
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        
        for i, (fai_val, true_label, fault_pred) in enumerate(zip(all_fai, all_labels, all_fault_labels)):
            if true_label == 0:  # 正常样本
                if fai_val > threshold:
                    fp += 1
                else:
                    tn += 1
            else:  # 故障样本
                # 故障样本：需要fai > threshold 且 三窗口确认为故障 才是TP
                if fai_val > threshold and fault_pred == 1:
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
    ax1.plot(fpr_list, tpr_list, color='blue', linewidth=2,
            label=f'Transformer (AUC={auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 固定阈值工作点 ===
    ax2.set_title('(b) Working Point\n(Three-Level Alarm Threshold)')
    
    # 使用Transformer性能指标
    if "TRANSFORMER" not in performance_metrics:
        print("   ⚠️ 警告: 没有可用的性能指标")
        return
    
    metrics = performance_metrics["TRANSFORMER"]['classification_metrics']
    ax2.scatter(metrics['fpr'], metrics['tpr'], 
               s=200, color='blue', 
               label=f'Transformer\n(TPR={metrics["tpr"]:.3f}, FPR={metrics["fpr"]:.3f})',
               marker='o', edgecolors='black', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === 子图3: 性能指标展示 ===
    ax3.set_title('(c) Transformer Classification Metrics')
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_mapping = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall', 'F1-Score': 'f1_score', 'Specificity': 'specificity'}
    transformer_values = [performance_metrics["TRANSFORMER"]['classification_metrics'][metric_mapping[m]] 
                         for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.6
    
    bars = ax3.bar(x, transformer_values, width, color='blue', alpha=0.7)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, transformer_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # === 子图4: 样本级性能展示 ===
    ax4.set_title('(d) Transformer Sample-Level Performance')
    
    sample_metrics = ['Avg φ(Normal)', 'Avg φ(Fault)', 'Anomaly Rate(Normal)', 'Anomaly Rate(Fault)']
    transformer_sample_values = [
        performance_metrics["TRANSFORMER"]['sample_metrics']['avg_fai_normal'],
        performance_metrics["TRANSFORMER"]['sample_metrics']['avg_fai_fault'],
        performance_metrics["TRANSFORMER"]['sample_metrics']['avg_anomaly_ratio_normal'],
        performance_metrics["TRANSFORMER"]['sample_metrics']['avg_anomaly_ratio_fault']
    ]
    
    x = np.arange(len(sample_metrics))
    bars = ax4.bar(x, transformer_sample_values, width, color='blue', alpha=0.7)
    
    ax4.set_xlabel('Sample Metrics')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_metrics, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, transformer_sample_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ Transformer ROC分析图保存至: {save_path}")

#----------------------------------------故障检测时序图------------------------------
def create_fault_detection_timeline(test_results, save_path):
    """生成Transformer故障检测时序图"""
    print("   📊 生成Transformer故障检测时序图...")
    
    # 选择一个故障样本进行可视化
    fault_sample_id = TEST_SAMPLES['fault'][0] if TEST_SAMPLES['fault'] else '335'  # 使用第一个故障样本
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # 找到对应样本的结果
    sample_result = next((r for r in test_results["TRANSFORMER"] if r.get('sample_id') == fault_sample_id), None)
    
    if sample_result is None:
        print(f"   ⚠️ 未找到样本 {fault_sample_id} 的结果，使用第一个可用结果")
        sample_result = test_results["TRANSFORMER"][0] if test_results["TRANSFORMER"] else None
    
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
           label='Transformer FAI')
    ax1.axhline(y=thresholds['threshold1'], color='orange', linestyle='--', alpha=0.7,
              label='一级阈值')
    ax1.axhline(y=thresholds['threshold2'], color='red', linestyle='--', alpha=0.7,
              label='二级阈值')
    ax1.axhline(y=thresholds['threshold3'], color='darkred', linestyle='--', alpha=0.7,
              label='三级阈值')
    
    ax1.set_ylabel('Transformer\n综合诊断指标φ')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Transformer - 样本 {fault_sample_id} (故障样本)')
    
    # 子图2: 故障检测结果
    ax2 = axes[1]
    
    # 将故障标签转换为可视化区域
    fault_regions = np.where(fault_labels == 1, 0.8, 0)
    ax2.fill_between(time_axis, fault_regions, 
                    alpha=0.6, color='blue',
                    label='Transformer Fault Detection')
    
    ax2.set_ylabel('故障检测结果')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Transformer故障检测结果')
    
    # 子图3: 三窗口检测过程
    ax3 = axes[2]
    detection_info = sample_result['detection_info']
    
    ax3.plot(time_axis, fai_values, 'b-', alpha=0.5, label='φ指标值')
    
    # 标记候选点
    if detection_info['candidate_points']:
        ax3.scatter(detection_info['candidate_points'], 
                   [fai_values[i] for i in detection_info['candidate_points']],
                   color='orange', s=30, label='候选点', alpha=0.8)
    
    # 标记验证通过的点
    if detection_info['verified_points']:
        verified_indices = [v['point'] for v in detection_info['verified_points']]
        ax3.scatter(verified_indices,
                   [fai_values[i] for i in verified_indices],
                   color='red', s=50, label='验证点', marker='^')
    
    # 标记故障区域
    for region in detection_info['marked_regions']:
        start, end = region['range']
        ax3.axvspan(start, end, alpha=0.2, color='red', label='标记故障区域')
    
    ax3.set_ylabel('三窗口\n检测过程')
    ax3.set_xlabel('时间步长')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('三窗口检测过程 (Transformer)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ Transformer时序图保存至: {save_path}")

#----------------------------------------性能指标雷达图------------------------------
def create_performance_radar(performance_metrics, save_path):
    """生成Transformer性能指标雷达图"""
    print("   🕸️ 生成Transformer性能指标雷达图...")
    
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
    
    # 使用Transformer性能指标
    if "TRANSFORMER" not in performance_metrics:
        print("   ⚠️ 警告: 没有可用的性能指标")
        return
    
    # 数据预处理：FPR需要转换为控制能力 (1-FPR)
    transformer_values = []
    
    for metric_name, metric_key in radar_metrics.items():
        transformer_val = performance_metrics["TRANSFORMER"]['classification_metrics'][metric_key]
        
        # 特殊处理：误报控制 = 1 - FPR
        if metric_name == '误报控制':
            transformer_val = 1 - transformer_val
            
        transformer_values.append(transformer_val)
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    transformer_values += transformer_values[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_medium"], subplot_kw=dict(projection='polar'))
    
    # 绘制雷达图
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
    plt.title('Transformer性能指标雷达图', 
              pad=20, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 添加性能总结
    transformer_avg = np.mean(transformer_values[:-1])
    
    plt.figtext(0.02, 0.02, f'Transformer综合性能: {transformer_avg:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ Transformer雷达图保存至: {save_path}")

#----------------------------------------三窗口过程可视化------------------------------
def create_three_window_visualization(test_results, save_path):
    """生成Transformer三窗口检测过程可视化"""
    print("   🔍 生成Transformer三窗口过程可视化...")
    
    # 选择一个故障样本进行详细分析
    fault_sample_id = TEST_SAMPLES['fault'][0] if TEST_SAMPLES['fault'] else '335'
    
    fig = plt.figure(figsize=(16, 10))
    
    # 使用GridSpec进行复杂布局
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # === 主图：三窗口检测过程时序图 ===
    ax_main = fig.add_subplot(gs[0, :])
    
    # 选择Transformer结果进行可视化
    transformer_result = next((r for r in test_results["TRANSFORMER"] if r.get('sample_id') == fault_sample_id), None)
    
    if transformer_result is None:
        print(f"   ⚠️ 未找到样本 {fault_sample_id} 的结果，使用第一个可用结果")
        transformer_result = test_results["TRANSFORMER"][0] if test_results["TRANSFORMER"] else None
    
    if transformer_result is None:
        print("   ❌ 没有可用的测试结果")
        return
    
    fai_values = transformer_result.get('fai', [])
    detection_info = transformer_result.get('detection_info', {})
    thresholds = transformer_result.get('thresholds', {})
    threshold1 = thresholds.get('threshold1', 0.0)
    
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
    ax_main.set_title(f'Transformer三窗口故障检测过程 - 样本 {fault_sample_id}', 
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
    window_labels = ['检测窗口\n(25)', '验证窗口\n(15)', '标记窗口\n(10)']
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
    
    # === 子图4：Transformer性能 ===
    ax4 = fig.add_subplot(gs[1, 3])
    
    sample_result = next((r for r in test_results['TRANSFORMER'] if r.get('sample_id') == fault_sample_id), None)
    if sample_result is None:
        sample_result = test_results['TRANSFORMER'][0] if test_results['TRANSFORMER'] else None
    
    if sample_result is None:
        fault_ratio = 0.0
    else:
        detection_info = sample_result.get('detection_info', {})
        window_stats = detection_info.get('window_stats', {})
        fault_ratio = window_stats.get('fault_ratio', 0.0)
    
    bars4 = ax4.bar(['Transformer'], [fault_ratio], color='blue', alpha=0.7)
    ax4.set_title('Transformer\n(故障检测比率)')
    ax4.set_ylabel('故障比率')
    
    for bar, value in zip(bars4, [fault_ratio]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # === 底部：过程说明 ===
    process_text = """
    Transformer三窗口检测过程:
    
    1. 检测窗口 (25点): 扫描候选故障点，条件：φ(FAI) > 阈值
    2. 验证窗口 (15点): 验证候选点，检查连续性 (≥60% 超阈值)
    3. 标记窗口 (±10点): 标记确认的故障区域
    
    优势: 在保持高敏感性的同时减少误报
    """
    
    fig.text(0.02, 0.02, process_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"])
    plt.close()
    
    print(f"   ✅ Transformer三窗口过程图保存至: {save_path}")

#----------------------------------------结果保存函数------------------------------
def save_test_results(test_results, performance_metrics):
    """保存Transformer测试结果"""
    print("\n💾 保存Transformer测试结果...")
    
    # 创建结果目录 - 统一保存到models目录下
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/mnt/bz25t/bzhy/datasave/Transformer/models/test_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{result_dir}/detailed_results", exist_ok=True)
    
    # 1. 保存性能指标JSON
    performance_file = f"{result_dir}/transformer_performance_metrics.json"
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Transformer性能指标保存至: {performance_file}")
    
    # 2. 保存详细结果
    detail_file = f"{result_dir}/detailed_results/transformer_detailed_results.pkl"
    with open(detail_file, 'wb') as f:
        pickle.dump(test_results["TRANSFORMER"], f)
    print(f"   ✅ Transformer详细结果保存至: {detail_file}")
    
    # 3. 保存元数据
    metadata_file = f"{result_dir}/detailed_results/transformer_test_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(test_results['metadata'], f, indent=2, ensure_ascii=False)
    print(f"   ✅ Transformer测试元数据保存至: {metadata_file}")
    
    # 4. 创建Excel总结报告
    summary_file = f"{result_dir}/detailed_results/transformer_summary.xlsx"
    
    with pd.ExcelWriter(summary_file) as writer:
        # 使用Transformer性能指标
        if "TRANSFORMER" not in performance_metrics:
            print("   ⚠️ 警告: 没有可用的性能指标")
            return result_dir
        
        metrics = performance_metrics["TRANSFORMER"]['classification_metrics']
        confusion = performance_metrics["TRANSFORMER"]['confusion_matrix']
        sample_metrics = performance_metrics["TRANSFORMER"]['sample_metrics']
        
        performance_data = [{
            'Model': 'TRANSFORMER',
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
        performance_df.to_excel(writer, sheet_name='Transformer_Performance', index=False)
        
        # 样本详情表
        sample_details = []
        for result in test_results["TRANSFORMER"]:
            # 安全获取detection_info和window_stats
            detection_info = result.get('detection_info', {})
            window_stats = detection_info.get('window_stats', {})
            performance_metrics = result.get('performance_metrics', {})
            
            sample_details.append({
                'Sample_ID': result.get('sample_id', 'Unknown'),
                'True_Label': 'Fault' if result.get('label', 0) == 1 else 'Normal',
                'FAI_Mean': performance_metrics.get('fai_mean', 0.0),
                'FAI_Std': performance_metrics.get('fai_std', 0.0),
                'FAI_Max': performance_metrics.get('fai_max', 0.0),
                'Anomaly_Ratio': performance_metrics.get('anomaly_ratio', 0.0),
                'Fault_Detection_Ratio': window_stats.get('fault_ratio', 0.0),
                'Candidates_Found': window_stats.get('total_candidates', 0),
                'Verified_Points': window_stats.get('verified_candidates', 0)
            })
        
        sample_df = pd.DataFrame(sample_details)
        sample_df.to_excel(writer, sheet_name='Sample_Details', index=False)
    
    print(f"   ✅ Transformer Excel总结报告保存至: {summary_file}")
    
    return result_dir

#----------------------------------------主执行流程------------------------------
print("\n🎨 生成可视化分析...")

# 计算性能指标
performance_metrics = calculate_performance_metrics(test_results)

# 保存测试结果和生成可视化
result_dir = save_test_results(test_results, performance_metrics)

# 生成可视化图表
print("\n🎨 生成Transformer可视化分析...")

# 生成ROC分析图
create_roc_analysis(test_results, performance_metrics, f"{result_dir}/visualizations/transformer_roc_analysis.png")

# 生成故障检测时序图
create_fault_detection_timeline(test_results, f"{result_dir}/visualizations/transformer_fault_detection_timeline.png")

# 生成性能雷达图
create_performance_radar(performance_metrics, f"{result_dir}/visualizations/transformer_performance_radar.png")

# 生成三窗口过程图
create_three_window_visualization(test_results, f"{result_dir}/visualizations/transformer_three_window_process.png")

#----------------------------------------最终总结------------------------------
print("\n" + "="*80)
print("🎉 Transformer模型测试完成！")
print("="*80)

print(f"\n📊 测试结果总结:")
print(f"   • 测试样本: {len(ALL_TEST_SAMPLES)} 个 (正常: {len(TEST_SAMPLES['normal'])}, 故障: {len(TEST_SAMPLES['fault'])})")
print(f"   • 模型类型: Transformer")
print(f"   • 检测模式: {DETECTION_MODES[CURRENT_DETECTION_MODE]['name']}")

print(f"\n🔬 Transformer性能:")
if CURRENT_DETECTION_MODE == "three_window":
    print(f"   • 窗口配置: 检测({WINDOW_CONFIG['detection_window']}) → 验证({WINDOW_CONFIG['verification_window']}) → 标记({WINDOW_CONFIG['marking_window']})")
    else:
    print(f"   • 5点检测模式: 当前点+前后相邻点高于阈值时，标记5点区域")

metrics = performance_metrics["TRANSFORMER"]['classification_metrics']
print(f"   准确率: {metrics['accuracy']:.3f}")
print(f"   精确率: {metrics['precision']:.3f}")
print(f"   召回率: {metrics['recall']:.3f}")
print(f"   F1分数: {metrics['f1_score']:.3f}")
print(f"   TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}")

print(f"\n📁 结果文件:")
print(f"   • 结果目录: {result_dir}")
print(f"   • 可视化图表: {result_dir}/visualizations")
print(f"     - ROC分析图: transformer_roc_analysis.png")
print(f"     - 故障检测时序图: transformer_fault_detection_timeline.png") 
print(f"     - 性能雷达图: transformer_performance_radar.png")
print(f"     - 三窗口过程图: transformer_three_window_process.png")
print(f"   • 性能指标: transformer_performance_metrics.json")
print(f"   • Excel报告: transformer_summary.xlsx")

# 综合性能评估
transformer_score = np.mean(list(performance_metrics["TRANSFORMER"]['classification_metrics'].values()))

print(f"\n🏆 Transformer综合性能评估:")
    print(f"   综合得分: {transformer_score:.3f}")

print("\n" + "="*80)
print("Transformer测试完成！请查看生成的可视化图表和分析报告。")
print("="*80)