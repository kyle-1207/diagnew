# 三模型综合比较可视化脚本
# 对比 BiLSTM, Transformer_Positive, Transformer_PN 三个模型的性能
# 
# 作者: AI Assistant
# 创建时间: 2024
# 

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
import json
import pickle
import glob
from datetime import datetime
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import platform

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体显示
import matplotlib.font_manager as fm
from matplotlib import rcParams

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
    rcParams['figure.dpi'] = 100
    rcParams['figure.autolayout'] = False
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    
    # 5. 强制字体缓存重建
    try:
        fm._rebuild()
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
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("🔄 已切换到安全模式（英文标签）")

# 执行字体配置
setup_chinese_fonts_strict()

print("="*80)
print("🔬 三模型综合比较可视化系统")
print("="*80)
print("📊 对比模型: BiLSTM vs Transformer_Positive vs Transformer_PN")
print("📁 数据来源: Three_model目录下各模型的测试结果")
print("="*80)

#----------------------------------------配置参数------------------------------
# 模型配置
MODEL_CONFIGS = {
    'BILSTM': {
        'name': 'BiLSTM',
        'display_name': 'BiLSTM',
        'color': '#1f77b4',  # 蓝色
        'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/BILSTM/',
        'result_pattern': 'test_results_*'
    },
    'TRANSFORMER_POSITIVE': {
        'name': 'TRANSFORMER_POSITIVE', 
        'display_name': 'Transformer (+)',
        'color': '#ff7f0e',  # 橙色
        'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/transformer_positive/',
        'result_pattern': 'test_results_*'
    },
    'TRANSFORMER_PN': {
        'name': 'TRANSFORMER_PN',
        'display_name': 'Transformer (±)',
        'color': '#2ca02c',  # 绿色
        'base_path': '/mnt/bz25t/bzhy/datasave/Three_model/transformer_PN/',
        'result_pattern': 'test_results_*'
    }
}

# 可视化配置
PLOT_CONFIG = {
    "dpi": 300,
    "figsize_large": (20, 16),
    "figsize_xlarge": (24, 18),
    "figsize_medium": (16, 12), 
    "bbox_inches": "tight"
}

# 性能指标映射
METRIC_MAPPING = {
    'accuracy': '准确率',
    'precision': '精确率',
    'recall': '召回率',
    'f1_score': 'F1分数',
    'specificity': '特异性',
    'tpr': '真正率',
    'fpr': '假正率'
}

#----------------------------------------数据加载模块------------------------------
def find_latest_test_results(base_path, pattern='test_results_*'):
    """查找最新的测试结果目录"""
    search_path = os.path.join(base_path, pattern)
    test_dirs = glob.glob(search_path)
    
    if not test_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_dir = max(test_dirs, key=os.path.getmtime)
    return latest_dir

def safe_load_json(file_path, default=None):
    """安全加载JSON文件"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"   ⚠️ 文件不存在: {file_path}")
            return default
    except Exception as e:
        print(f"   ❌ 加载JSON失败: {file_path}, 错误: {e}")
        return default

def safe_load_pickle(file_path, default=None):
    """安全加载Pickle文件"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"   ⚠️ 文件不存在: {file_path}")
            return default
    except Exception as e:
        print(f"   ❌ 加载Pickle失败: {file_path}, 错误: {e}")
        return default

def load_model_results(model_name, config):
    """加载单个模型的测试结果"""
    print(f"📁 加载 {config['display_name']} 模型结果...")
    
    # 查找最新的测试结果目录
    latest_dir = find_latest_test_results(config['base_path'], config['result_pattern'])
    
    if latest_dir is None:
        print(f"   ❌ 未找到 {model_name} 的测试结果目录")
        return None
    
    print(f"   📂 找到结果目录: {latest_dir}")
    
    # 加载性能指标
    if model_name == 'BILSTM':
        performance_file = os.path.join(latest_dir, 'bilstm_performance_metrics.json')
        detailed_file = os.path.join(latest_dir, 'detailed_results', 'bilstm_detailed_results.pkl')
        metadata_file = os.path.join(latest_dir, 'detailed_results', 'bilstm_test_metadata.json')
    else:
        performance_file = os.path.join(latest_dir, 'performance_metrics.json')
        detailed_file = os.path.join(latest_dir, 'detailed_results.pkl')
        metadata_file = os.path.join(latest_dir, 'test_metadata.json')
    
    # 加载数据
    performance_data = safe_load_json(performance_file, {})
    detailed_data = safe_load_pickle(detailed_file, [])
    metadata = safe_load_json(metadata_file, {})
    
    # 检查数据完整性
    if not performance_data:
        print(f"   ⚠️ {model_name} 性能数据为空")
        return None
    
    result = {
        'model_name': model_name,
        'config': config,
        'performance_data': performance_data,
        'detailed_data': detailed_data,
        'metadata': metadata,
        'result_dir': latest_dir
    }
    
    print(f"   ✅ {config['display_name']} 数据加载完成")
    return result

def standardize_metrics(raw_metrics, model_name):
    """标准化不同模型的性能指标格式"""
    try:
        if model_name == 'BILSTM':
            # BiLSTM使用的格式: performance_data['BILSTM']
            if 'BILSTM' in raw_metrics:
                model_metrics = raw_metrics['BILSTM']
            else:
                model_metrics = raw_metrics
        else:
            # Transformer使用的格式: performance_data['TRANSFORMER']
            if 'TRANSFORMER' in raw_metrics:
                model_metrics = raw_metrics['TRANSFORMER']
            else:
                model_metrics = raw_metrics
        
        # 提取分类指标
        if 'classification_metrics' in model_metrics:
            classification = model_metrics['classification_metrics']
        else:
            classification = model_metrics
        
        # 提取混淆矩阵
        if 'confusion_matrix' in model_metrics:
            confusion = model_metrics['confusion_matrix']
        else:
            confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        
        # 提取样本指标
        if 'sample_metrics' in model_metrics:
            sample_metrics = model_metrics['sample_metrics']
        else:
            sample_metrics = {}
        
        # 标准化格式
        standardized = {
            'classification_metrics': {
                'accuracy': classification.get('accuracy', 0.0),
                'precision': classification.get('precision', 0.0),
                'recall': classification.get('recall', 0.0),
                'f1_score': classification.get('f1_score', 0.0),
                'specificity': classification.get('specificity', 0.0),
                'tpr': classification.get('tpr', 0.0),
                'fpr': classification.get('fpr', 0.0)
            },
            'confusion_matrix': confusion,
            'sample_metrics': sample_metrics
        }
        
        return standardized
        
    except Exception as e:
        print(f"   ❌ 标准化 {model_name} 指标失败: {e}")
        # 返回默认值
        return {
            'classification_metrics': {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'specificity': 0.0, 'tpr': 0.0, 'fpr': 0.0
            },
            'confusion_matrix': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
            'sample_metrics': {}
        }

def load_all_model_results():
    """加载所有模型的测试结果"""
    print("\n🔄 开始加载所有模型的测试结果...")
    
    all_results = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        result = load_model_results(model_name, config)
        if result is not None:
            all_results[model_name] = result
        else:
            print(f"   ⚠️ 跳过 {model_name}，数据加载失败")
    
    print(f"\n✅ 成功加载 {len(all_results)} 个模型的结果")
    return all_results

#----------------------------------------ROC曲线比较------------------------------
def create_three_model_roc_comparison(all_results, save_path):
    """生成三模型ROC曲线比较图"""
    print("   📈 生成三模型ROC曲线比较图...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG["figsize_large"], constrained_layout=True)
    
    # === 子图1: ROC曲线对比 ===
    ax1.set_title('(a) Three-Model ROC Curve Comparison', fontsize=14, fontweight='bold')
    
    model_auc_scores = {}
    
    for model_name, result in all_results.items():
        config = result['config']
        performance_data = result['performance_data']
        
        # 标准化指标
        std_metrics = standardize_metrics(performance_data, model_name)
        
        # 获取ROC数据（如果存在）
        roc_data = None
        if model_name == 'BILSTM' and 'BILSTM' in performance_data:
            roc_data = performance_data['BILSTM'].get('roc_data', None)
        elif 'TRANSFORMER' in performance_data:
            roc_data = performance_data['TRANSFORMER'].get('roc_data', None)
        
        if roc_data and 'true_labels' in roc_data and 'fai_values' in roc_data:
            # 使用存储的ROC数据
            true_labels = np.array(roc_data['true_labels'])
            fai_values = np.array(roc_data['fai_values'])
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(true_labels, fai_values)
            auc_score = auc(fpr, tpr)
            
            ax1.plot(fpr, tpr, color=config['color'], linewidth=2.5,
                    label=f'{config["display_name"]} (AUC={auc_score:.3f})')
            
            model_auc_scores[model_name] = auc_score
        else:
            # 如果没有ROC数据，使用工作点
            metrics = std_metrics['classification_metrics']
            fpr_point = metrics['fpr']
            tpr_point = metrics['tpr']
            
            ax1.scatter(fpr_point, tpr_point, s=200, color=config['color'],
                       label=f'{config["display_name"]} (工作点)',
                       marker='o', edgecolors='black', linewidth=2)
            
            model_auc_scores[model_name] = 0.5  # 默认AUC
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类器')
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # === 子图2: 工作点比较 ===
    ax2.set_title('(b) Working Points Comparison', fontsize=14, fontweight='bold')
    
    for model_name, result in all_results.items():
        config = result['config']
        performance_data = result['performance_data']
        
        std_metrics = standardize_metrics(performance_data, model_name)
        metrics = std_metrics['classification_metrics']
        
        ax2.scatter(metrics['fpr'], metrics['tpr'], s=300, color=config['color'],
                   label=f'{config["display_name"]}\n(TPR={metrics["tpr"]:.3f}, FPR={metrics["fpr"]:.3f})',
                   marker='o', edgecolors='black', linewidth=2, alpha=0.8)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('True Positive Rate (TPR)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # === 子图3: 分类指标对比 ===
    ax3.set_title('(c) Classification Metrics Comparison', fontsize=14, fontweight='bold')
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, (model_name, result) in enumerate(all_results.items()):
        config = result['config']
        performance_data = result['performance_data']
        
        std_metrics = standardize_metrics(performance_data, model_name)
        values = [std_metrics['classification_metrics'][key] for key in metric_keys]
        
        bars = ax3.bar(x + i*width, values, width, 
                      label=config['display_name'], 
                      color=config['color'], alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metrics_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # === 子图4: AUC对比 ===
    ax4.set_title('(d) AUC Scores Comparison', fontsize=14, fontweight='bold')
    
    if model_auc_scores:
        models = list(model_auc_scores.keys())
        auc_values = list(model_auc_scores.values())
        colors = [MODEL_CONFIGS[m]['color'] for m in models]
        display_names = [MODEL_CONFIGS[m]['display_name'] for m in models]
        
        bars = ax4.bar(display_names, auc_values, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, auc_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4.set_ylabel('AUC Score')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)
        
        # 找出最佳模型
        best_model_idx = np.argmax(auc_values)
        best_auc = auc_values[best_model_idx]
        best_name = display_names[best_model_idx]
        
        ax4.text(0.5, 0.95, f'最佳: {best_name} (AUC={best_auc:.3f})', 
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No AUC Data Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ ROC比较图保存至: {save_path}")

#----------------------------------------性能雷达图比较------------------------------
def create_three_model_radar_comparison(all_results, save_path):
    """生成三模型性能雷达图比较"""
    print("   🕸️ 生成三模型性能雷达图比较...")
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_large"], 
                                   subplot_kw=dict(projection='polar'), constrained_layout=True)
    
    # 设置雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # === 子图1: 雷达图叠加显示 ===
    ax1.set_title('Three-Model Performance Radar Chart', pad=20, fontsize=14, fontweight='bold')
    
    model_scores = {}
    
    for model_name, result in all_results.items():
        config = result['config']
        performance_data = result['performance_data']
        
        std_metrics = standardize_metrics(performance_data, model_name)
        
        # 提取雷达图数据
        values = []
        for metric_name, metric_key in radar_metrics.items():
            val = std_metrics['classification_metrics'][metric_key]
            
            # 特殊处理：误报控制 = 1 - FPR
            if metric_name == 'False Alarm Control':
                val = 1 - val
                
            values.append(val)
        
        values += values[:1]  # 闭合
        model_scores[model_name] = np.mean(values[:-1])  # 计算平均分
        
        # 绘制雷达图
        ax1.plot(angles, values, 'o-', linewidth=2.5, 
                label=config['display_name'], color=config['color'])
        ax1.fill(angles, values, alpha=0.15, color=config['color'])
    
    # 设置雷达图标签
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(list(radar_metrics.keys()))
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # === 子图2: 综合评分对比 ===
    ax2.remove()  # 移除极坐标轴
    ax2 = fig.add_subplot(1, 2, 2)  # 添加直角坐标轴
    
    ax2.set_title('Comprehensive Performance Scores', fontsize=14, fontweight='bold')
    
    if model_scores:
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        colors = [MODEL_CONFIGS[m]['color'] for m in models]
        display_names = [MODEL_CONFIGS[m]['display_name'] for m in models]
        
        bars = ax2.bar(display_names, scores, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Comprehensive Score')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # 排序并标注
        sorted_pairs = sorted(zip(display_names, scores), key=lambda x: x[1], reverse=True)
        ranking_text = "Ranking:\n"
        for i, (name, score) in enumerate(sorted_pairs):
            ranking_text += f"{i+1}. {name}: {score:.3f}\n"
        
        ax2.text(0.02, 0.98, ranking_text.strip(), transform=ax2.transAxes, 
                va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ 雷达图比较保存至: {save_path}")

#----------------------------------------故障检测时序比较------------------------------
def create_three_model_timeline_comparison(all_results, save_path):
    """生成三模型故障检测时序比较图"""
    print("   📊 生成三模型故障检测时序比较图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True, constrained_layout=True)
    
    # 尝试找到一个共同的故障样本进行可视化
    target_sample = None
    sample_data = {}
    
    for model_name, result in all_results.items():
        detailed_data = result['detailed_data']
        if detailed_data and len(detailed_data) > 0:
            # 寻找故障样本
            fault_samples = [r for r in detailed_data if r.get('label', 0) == 1]
            if fault_samples:
                sample_result = fault_samples[0]  # 使用第一个故障样本
                target_sample = sample_result.get('sample_id', 'Unknown')
                sample_data[model_name] = sample_result
                
    if not sample_data:
        # 如果没有找到故障样本，使用第一个可用样本
        for model_name, result in all_results.items():
            detailed_data = result['detailed_data']
            if detailed_data and len(detailed_data) > 0:
                sample_result = detailed_data[0]
                target_sample = sample_result.get('sample_id', 'Unknown')
                sample_data[model_name] = sample_result
    
    if not sample_data:
        print("   ⚠️ 没有找到可用的样本数据")
        return
    
    print(f"   📋 使用样本 {target_sample} 进行时序比较")
    
    # 为每个模型生成子图
    for i, (model_name, result) in enumerate(all_results.items()):
        ax = axes[i]
        config = result['config']
        
        if model_name in sample_data:
            sample_result = sample_data[model_name]
            
            fai_values = sample_result.get('fai', [])
            fault_labels = sample_result.get('fault_labels', [])
            thresholds = sample_result.get('thresholds', {})
            detection_info = sample_result.get('detection_info', {})
            
            if len(fai_values) > 0:
                time_axis = np.arange(len(fai_values))
                
                # 绘制FAI时序
                ax.plot(time_axis, fai_values, color=config['color'], linewidth=1.5, alpha=0.8,
                       label=f'{config["display_name"]} φ(FAI)')
                
                # 绘制阈值线
                if 'threshold1' in thresholds:
                    ax.axhline(y=thresholds['threshold1'], color='red', linestyle='--', alpha=0.7,
                              label='Level 1 Threshold')
                
                # 标记故障区域
                if len(fault_labels) > 0:
                    fault_regions = np.where(np.array(fault_labels) > 0, 1, 0)
                    ax.fill_between(time_axis, 0, np.max(fai_values) * fault_regions, 
                                   alpha=0.3, color='red', label='Detected Faults')
                
                # 标记触发点（如果有）
                if detection_info.get('trigger_points'):
                    trigger_points = detection_info['trigger_points']
                    ax.scatter(trigger_points, [fai_values[i] for i in trigger_points if i < len(fai_values)],
                              color='orange', s=30, alpha=0.8, label='Trigger Points')
                
                ax.set_ylabel(f'{config["display_name"]}\nφ Index')
                ax.set_title(f'{config["display_name"]} - Sample {target_sample}')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No Data for {config["display_name"]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{config["display_name"]} - No Data')
        else:
            ax.text(0.5, 0.5, f'No Sample Data for {config["display_name"]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{config["display_name"]} - No Sample Data')
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle(f'Three-Model Fault Detection Timeline Comparison - Sample {target_sample}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ 时序比较图保存至: {save_path}")

#----------------------------------------混淆矩阵比较------------------------------
def create_confusion_matrix_comparison(all_results, save_path):
    """生成三模型混淆矩阵比较图"""
    print("   📊 生成三模型混淆矩阵比较图...")
    
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5), constrained_layout=True)
    
    if num_models == 1:
        axes = [axes]
    
    for i, (model_name, result) in enumerate(all_results.items()):
        ax = axes[i]
        config = result['config']
        performance_data = result['performance_data']
        
        std_metrics = standardize_metrics(performance_data, model_name)
        confusion = std_metrics['confusion_matrix']
        
        # 构建混淆矩阵
        cm = np.array([[confusion['TN'], confusion['FP']], 
                       [confusion['FN'], confusion['TP']]])
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Normal', 'Predicted Fault'],
                   yticklabels=['Actual Normal', 'Actual Fault'])
        
        ax.set_title(f'{config["display_name"]}\nConfusion Matrix', fontsize=12, fontweight='bold')
        
        # 添加性能指标
        metrics = std_metrics['classification_metrics']
        info_text = f"Acc: {metrics['accuracy']:.3f}\nF1: {metrics['f1_score']:.3f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=10)
    
    plt.suptitle('Three-Model Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches=PLOT_CONFIG["bbox_inches"], facecolor='white')
    plt.close()
    
    print(f"   ✅ 混淆矩阵比较图保存至: {save_path}")

#----------------------------------------综合分析报告------------------------------
def generate_comprehensive_analysis(all_results):
    """生成综合分析报告"""
    print("   📋 生成综合分析报告...")
    
    analysis = {
        'summary': {},
        'ranking': {},
        'recommendations': {},
        'detailed_comparison': {}
    }
    
    # 收集所有模型的指标
    model_metrics = {}
    for model_name, result in all_results.items():
        config = result['config']
        performance_data = result['performance_data']
        
        std_metrics = standardize_metrics(performance_data, model_name)
        
        model_metrics[model_name] = {
            'display_name': config['display_name'],
            'metrics': std_metrics['classification_metrics'],
            'confusion': std_metrics['confusion_matrix']
        }
    
    # 计算综合评分
    weights = {
        'accuracy': 0.2,
        'precision': 0.2,
        'recall': 0.2,
        'f1_score': 0.3,
        'specificity': 0.1
    }
    
    composite_scores = {}
    for model_name, data in model_metrics.items():
        score = sum(data['metrics'][metric] * weight for metric, weight in weights.items())
        composite_scores[model_name] = score
    
    # 排序
    ranked_models = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 生成分析报告
    analysis['summary'] = {
        'total_models': len(all_results),
        'evaluation_metrics': list(weights.keys()),
        'best_model': ranked_models[0][0] if ranked_models else None,
        'best_score': ranked_models[0][1] if ranked_models else 0.0
    }
    
    analysis['ranking'] = {
        'composite_scores': composite_scores,
        'ranked_list': ranked_models
    }
    
    # 详细比较
    metric_winners = {}
    for metric in weights.keys():
        best_model = max(model_metrics.items(), key=lambda x: x[1]['metrics'][metric])
        metric_winners[metric] = {
            'model': best_model[0],
            'display_name': best_model[1]['display_name'],
            'score': best_model[1]['metrics'][metric]
        }
    
    analysis['detailed_comparison'] = {
        'metric_winners': metric_winners,
        'model_metrics': model_metrics
    }
    
    # 生成推荐
    if ranked_models:
        best_model_name = ranked_models[0][0]
        best_display_name = model_metrics[best_model_name]['display_name']
        
        recommendations = [
            f"综合性能最佳模型: {best_display_name}",
            f"综合得分: {ranked_models[0][1]:.3f}",
            ""
        ]
        
        # 各指标最佳模型
        recommendations.append("各指标最佳模型:")
        for metric, winner in metric_winners.items():
            recommendations.append(f"  {METRIC_MAPPING.get(metric, metric)}: {winner['display_name']} ({winner['score']:.3f})")
        
        analysis['recommendations'] = recommendations
    
    return analysis

#----------------------------------------主函数------------------------------
def create_comprehensive_comparison_report(all_results, save_dir):
    """创建综合比较报告"""
    print("\n🎨 生成综合比较可视化报告...")
    
    # 创建可视化目录
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 生成各种比较图表
    create_three_model_roc_comparison(all_results, 
                                     os.path.join(vis_dir, 'three_model_roc_comparison.png'))
    
    create_three_model_radar_comparison(all_results, 
                                       os.path.join(vis_dir, 'three_model_radar_comparison.png'))
    
    create_three_model_timeline_comparison(all_results, 
                                          os.path.join(vis_dir, 'three_model_timeline_comparison.png'))
    
    create_confusion_matrix_comparison(all_results, 
                                      os.path.join(vis_dir, 'three_model_confusion_matrix_comparison.png'))
    
    # 生成综合分析
    analysis = generate_comprehensive_analysis(all_results)
    
    # 保存分析报告
    analysis_file = os.path.join(save_dir, 'comprehensive_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # 生成Excel报告
    create_excel_comparison_report(all_results, analysis, save_dir)
    
    print(f"   ✅ 综合比较报告生成完成: {save_dir}")
    return analysis

def create_excel_comparison_report(all_results, analysis, save_dir):
    """创建Excel格式的比较报告"""
    print("   📊 生成Excel比较报告...")
    
    excel_file = os.path.join(save_dir, 'three_model_comparison_report.xlsx')
    
    with pd.ExcelWriter(excel_file) as writer:
        # 1. 综合性能对比表
        performance_data = []
        for model_name, result in all_results.items():
            config = result['config']
            performance_data_raw = result['performance_data']
            
            std_metrics = standardize_metrics(performance_data_raw, model_name)
            metrics = std_metrics['classification_metrics']
            confusion = std_metrics['confusion_matrix']
            
            performance_data.append({
                'Model': config['display_name'],
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
                'Composite_Score': analysis['ranking']['composite_scores'].get(model_name, 0.0)
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Composite_Score', ascending=False)
        performance_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)
        
        # 2. 排名表
        ranking_data = []
        for i, (model_name, score) in enumerate(analysis['ranking']['ranked_list']):
            config = MODEL_CONFIGS[model_name]
            ranking_data.append({
                'Rank': i + 1,
                'Model': config['display_name'],
                'Composite_Score': score,
                'Performance_Level': '优秀' if score >= 0.8 else '良好' if score >= 0.6 else '一般'
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_excel(writer, sheet_name='Model_Ranking', index=False)
        
        # 3. 指标获胜者表
        winners_data = []
        for metric, winner in analysis['detailed_comparison']['metric_winners'].items():
            winners_data.append({
                'Metric': METRIC_MAPPING.get(metric, metric),
                'Best_Model': winner['display_name'],
                'Score': winner['score']
            })
        
        winners_df = pd.DataFrame(winners_data)
        winners_df.to_excel(writer, sheet_name='Metric_Winners', index=False)
    
    print(f"   ✅ Excel报告保存至: {excel_file}")

#----------------------------------------主执行流程------------------------------
def main():
    """主执行函数"""
    print("\n🚀 开始三模型综合比较分析...")
    
    # 加载所有模型结果
    all_results = load_all_model_results()
    
    if len(all_results) == 0:
        print("❌ 没有找到任何模型的测试结果，请先运行各模型的测试脚本")
        return
    
    print(f"\n📊 成功加载 {len(all_results)} 个模型的结果:")
    for model_name, result in all_results.items():
        config = result['config']
        print(f"   • {config['display_name']}: {result['result_dir']}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/mnt/bz25t/bzhy/datasave/Three_model/comparison_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成综合比较报告
    analysis = create_comprehensive_comparison_report(all_results, save_dir)
    
    # 输出总结
    print("\n" + "="*80)
    print("🎉 三模型综合比较分析完成！")
    print("="*80)
    
    print(f"\n📊 比较结果总结:")
    print(f"   • 参与比较的模型: {len(all_results)} 个")
    for model_name, result in all_results.items():
        config = result['config']
        print(f"     - {config['display_name']}")
    
    if analysis['summary']['best_model']:
        best_model_config = MODEL_CONFIGS[analysis['summary']['best_model']]
        print(f"\n🏆 综合性能最佳模型: {best_model_config['display_name']}")
        print(f"   综合得分: {analysis['summary']['best_score']:.3f}")
    
    print(f"\n🎯 性能排名:")
    for i, (model_name, score) in enumerate(analysis['ranking']['ranked_list']):
        config = MODEL_CONFIGS[model_name]
        print(f"   {i+1}. {config['display_name']}: {score:.3f}")
    
    print(f"\n📁 结果文件:")
    print(f"   • 结果目录: {save_dir}")
    print(f"   • 可视化图表: {save_dir}/visualizations/")
    print(f"     - ROC曲线比较: three_model_roc_comparison.png")
    print(f"     - 雷达图比较: three_model_radar_comparison.png") 
    print(f"     - 时序图比较: three_model_timeline_comparison.png")
    print(f"     - 混淆矩阵比较: three_model_confusion_matrix_comparison.png")
    print(f"   • 分析报告: comprehensive_analysis.json")
    print(f"   • Excel报告: three_model_comparison_report.xlsx")
    
    if 'recommendations' in analysis:
        print(f"\n💡 推荐建议:")
        for rec in analysis['recommendations']:
            if rec.strip():
                print(f"   • {rec}")
    
    print("\n" + "="*80)
    print("三模型比较分析完成！请查看生成的可视化图表和分析报告。")
    print("="*80)

if __name__ == "__main__":
    main()
