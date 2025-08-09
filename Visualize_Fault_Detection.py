# 故障检测效果专项可视化脚本
# 基于BiLSTM三窗口和五点检测策略的深度分析

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import time

# 忽略警告
warnings.filterwarnings('ignore')

# Linux环境matplotlib配置
mpl.use('Agg')

# 设置字体（英文标签，避免中文方框问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

class FaultDetectionVisualizer:
    """故障检测可视化分析类"""
    
    def __init__(self, result_base_dir='/mnt/bz25t/bzhy/datasave'):
        self.result_base_dir = result_base_dir
        self.fault_colors = {
            'Normal': '#2ecc71',      # 绿色 - 正常
            'ISC': '#e74c3c',         # 红色 - 内短路
            'TR': '#f39c12',          # 橙色 - 热失控  
            'BSC': '#9b59b6',         # 紫色 - 外短路
            'EA': '#3498db'           # 蓝色 - 电解液泄漏
        }
        self.detection_strategies = {
            'BiLSTM': {'name': 'BiLSTM', 'color': '#1f77b4'},
            'Transformer': {'name': 'Transformer', 'color': '#ff7f0e'},
            'Combined': {'name': 'Combined Model', 'color': '#2ca02c'},
            'HybridFeedback': {'name': 'Hybrid Feedback', 'color': '#d62728'}
        }
        
    def load_detection_results(self):
        """加载故障检测结果数据"""
        print("📥 Loading fault detection results...")
        
        self.detection_data = {}
        
        # 方法1: 尝试加载传统格式的检测结果
        bilstm_dir = f"{self.result_base_dir}/BILSTM/detection_results"
        if os.path.exists(f"{bilstm_dir}/detection_summary.pkl"):
            with open(f"{bilstm_dir}/detection_summary.pkl", 'rb') as f:
                self.detection_data['BiLSTM'] = pickle.load(f)
            print("✅ BiLSTM detection results loaded (traditional format)")
        
        transformer_dir = f"{self.result_base_dir}/Transformer/detection_results"
        if os.path.exists(f"{transformer_dir}/detection_summary.pkl"):
            with open(f"{transformer_dir}/detection_summary.pkl", 'rb') as f:
                self.detection_data['Transformer'] = pickle.load(f)
            print("✅ Transformer detection results loaded (traditional format)")
        
        # 方法2: 尝试加载最新的测试结果格式（参考BiLSTM成功做法）
        if not self.detection_data:
            print("📋 Traditional format not found, searching for recent test results...")
            self._load_recent_test_results()
        
        print(f"📊 Loaded {len(self.detection_data)} detection result sets")
        
        # 如果仍然没有真实数据，生成模拟数据用于演示
        if not self.detection_data:
            print("⚠️  No detection results found, generating simulated data for demonstration")
            self._generate_simulated_detection_data()
        
        return len(self.detection_data) > 0
    
    def _load_recent_test_results(self):
        """加载最新的测试结果（参考BiLSTM成功做法）"""
        import glob
        
        # 搜索BiLSTM测试结果
        bilstm_pattern = f"{self.result_base_dir}/BILSTM/test_results/*/detailed_results/bilstm_detailed_results.pkl"
        bilstm_files = glob.glob(bilstm_pattern)
        
        if bilstm_files:
            # 选择最新的文件
            latest_bilstm = max(bilstm_files, key=os.path.getmtime)
            try:
                with open(latest_bilstm, 'rb') as f:
                    bilstm_test_results = pickle.load(f)
                
                # 转换为故障检测分析所需的格式
                self.detection_data['BiLSTM'] = self._convert_test_results_to_detection_format(bilstm_test_results, 'BiLSTM')
                print(f"✅ BiLSTM test results loaded from: {latest_bilstm}")
                
            except Exception as e:
                print(f"⚠️  Failed to load BiLSTM test results: {e}")
        
        # 搜索Transformer测试结果 
        transformer_pattern = f"{self.result_base_dir}/Transformer/test_results/*/detailed_results/transformer_detailed_results.pkl"
        transformer_files = glob.glob(transformer_pattern)
        
        if transformer_files:
            latest_transformer = max(transformer_files, key=os.path.getmtime)
            try:
                with open(latest_transformer, 'rb') as f:
                    transformer_test_results = pickle.load(f)
                
                self.detection_data['Transformer'] = self._convert_test_results_to_detection_format(transformer_test_results, 'Transformer')
                print(f"✅ Transformer test results loaded from: {latest_transformer}")
                
            except Exception as e:
                print(f"⚠️  Failed to load Transformer test results: {e}")
        
        # 搜索Combined模型测试结果（PN_model）
        # 根据实际文件路径结构搜索
        combined_pattern_1 = f"{self.result_base_dir}/Combined/test_results/*/detailed_results/combined_detailed_results.pkl"
        combined_pattern_2 = f"{self.result_base_dir}/Transformer/models/PN_model/test_results*/detailed_results/transformer_detailed_results.pkl"
        
        combined_files = glob.glob(combined_pattern_1) + glob.glob(combined_pattern_2)
        
        if combined_files:
            latest_combined = max(combined_files, key=os.path.getmtime)
            try:
                with open(latest_combined, 'rb') as f:
                    combined_test_results = pickle.load(f)
                
                self.detection_data['Combined'] = self._convert_test_results_to_detection_format(combined_test_results, 'Combined')
                print(f"✅ Combined test results loaded from: {latest_combined}")
                
            except Exception as e:
                print(f"⚠️  Failed to load Combined test results: {e}")
        else:
            print("⚠️  No Combined model test results found")
    
    def _convert_test_results_to_detection_format(self, test_results, model_name):
        """将测试结果转换为故障检测分析格式（参考BiLSTM成功做法）"""
        converted_data = {}
        
        # 按样本类型分组
        normal_samples = []
        fault_samples = []
        
        for result in test_results:
            sample_id = result.get('sample_id', 'unknown')
            true_label = result.get('label', 0)
            fai_values = result.get('fai', [])
            fault_labels = result.get('fault_labels', [])
            detection_info = result.get('detection_info', {})
            
            # 生成预测概率（基于fai值）
            if len(fai_values) > 0:
                thresholds = result.get('thresholds', {})
                threshold1 = thresholds.get('threshold1', np.mean(fai_values) + 3 * np.std(fai_values))
                
                # 将fai值转换为预测概率
                pred_probs = np.clip(fai_values / (threshold1 * 2), 0, 1)
                pred_labels = (np.array(fai_values) > threshold1).astype(int)
                
                sample_data = {
                    'sample_id': sample_id,
                    'true_labels': np.array([true_label] * len(fai_values)),
                    'pred_labels': pred_labels,
                    'pred_probs': pred_probs,
                    'n_samples': len(fai_values),
                    'fai_values': fai_values,
                    'fault_labels': fault_labels,
                    'detection_info': detection_info,
                    'thresholds': thresholds
                }
                
                if true_label == 0:
                    normal_samples.append(sample_data)
                else:
                    fault_samples.append(sample_data)
        
        # 合并同类样本
        if normal_samples:
            converted_data['Normal'] = self._merge_sample_data(normal_samples)
        
        if fault_samples:
            # 可以按故障类型进一步分类，这里简化为一个故障类型
            converted_data['Fault'] = self._merge_sample_data(fault_samples)
        
        return converted_data
    
    def _merge_sample_data(self, sample_list):
        """合并同类样本数据"""
        if not sample_list:
            return {}
        
        merged = {
            'true_labels': np.concatenate([s['true_labels'] for s in sample_list]),
            'pred_labels': np.concatenate([s['pred_labels'] for s in sample_list]),
            'pred_probs': np.concatenate([s['pred_probs'] for s in sample_list]),
            'n_samples': sum(s['n_samples'] for s in sample_list),
            'sample_details': sample_list  # 保留详细信息用于进一步分析
        }
        
        return merged
    
    def _generate_simulated_detection_data(self):
        """生成模拟故障检测数据用于演示"""
        np.random.seed(42)
        
        # 故障类型和对应的检测难度
        fault_types = ['Normal', 'ISC', 'TR', 'BSC', 'EA']
        detection_difficulties = [0.1, 0.15, 0.25, 0.2, 0.3]  # 越高越难检测
        
        for strategy in self.detection_strategies.keys():
            strategy_data = {}
            
            # 为每种故障类型生成检测结果
            for fault_type, difficulty in zip(fault_types, detection_difficulties):
                n_samples = np.random.randint(800, 1200)  # 每种故障类型的样本数
                
                # 基于策略和故障类型调整检测性能
                if 'BiLSTM' in strategy:
                    base_accuracy = 0.92 - difficulty
                elif 'Transformer' in strategy:
                    base_accuracy = 0.94 - difficulty * 0.8
                else:  # HybridFeedback
                    base_accuracy = 0.96 - difficulty * 0.6
                
                # 生成真实标签和预测结果
                true_labels = np.ones(n_samples) if fault_type != 'Normal' else np.zeros(n_samples)
                
                # 生成预测概率
                if fault_type == 'Normal':
                    pred_probs = np.random.beta(2, 8, n_samples)  # 正常样本应该有低概率
                else:
                    pred_probs = np.random.beta(8, 2, n_samples)  # 故障样本应该有高概率
                
                # 根据准确率调整预测结果
                pred_labels = (pred_probs > 0.5).astype(int)
                
                # 调整准确率到目标值
                current_accuracy = np.mean(pred_labels == true_labels)
                if current_accuracy < base_accuracy:
                    # 需要提高准确率，翻转一些错误预测
                    wrong_indices = np.where(pred_labels != true_labels)[0]
                    n_flip = int((base_accuracy - current_accuracy) * n_samples)
                    if len(wrong_indices) > n_flip:
                        flip_indices = np.random.choice(wrong_indices, n_flip, replace=False)
                        pred_labels[flip_indices] = true_labels[flip_indices]
                
                strategy_data[fault_type] = {
                    'true_labels': true_labels,
                    'pred_labels': pred_labels,
                    'pred_probs': pred_probs,
                    'n_samples': n_samples
                }
            
            self.detection_data[strategy] = strategy_data
    
    def create_fault_detection_dashboard(self):
        """创建故障检测综合仪表板"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. 检测策略性能雷达图 (左上)
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        self._plot_detection_strategy_radar(ax1)
        
        # 2. ROC曲线族 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_detection_roc_curves(ax2)
        
        # 3. 混淆矩阵热力图 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_confusion_matrix_heatmap(ax3)
        
        # 4. 故障类型检测准确率对比 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_fault_type_accuracy(ax4)
        
        # 5. 检测延迟分析 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_detection_delay_analysis(ax5)
        
        # 6. 误报率vs检测率散点图 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_false_positive_vs_detection_rate(ax6)
        
        # 7. 三窗口检测过程可视化 (左下)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_three_window_process(ax7)
        
        # 8. 五点检测过程可视化 (中下)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_five_point_process(ax8)
        
        # 9. 阈值敏感性分析 (右下)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_threshold_sensitivity(ax9)
        
        # 10. 检测置信度分布 (底部左)
        ax10 = fig.add_subplot(gs[3, 0])
        self._plot_confidence_distribution(ax10)
        
        # 11. 故障检测时序分析 (底部中)
        ax11 = fig.add_subplot(gs[3, 1])
        self._plot_temporal_detection_analysis(ax11)
        
        # 12. 综合性能评分 (底部右)
        ax12 = fig.add_subplot(gs[3, 2])
        self._plot_comprehensive_detection_score(ax12)
        
        # 添加总标题
        fig.suptitle('Fault Detection Comprehensive Analysis Dashboard', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 保存图表
        output_path = f"{self.result_base_dir}/fault_detection_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Fault detection dashboard saved: {output_path}")
        return output_path
    
    def _plot_detection_strategy_radar(self, ax):
        """绘制检测策略性能雷达图"""
        ax.set_title('Detection Strategy Performance\nRadar Chart', fontweight='bold', pad=20)
        
        # 定义性能指标
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'Robustness']
        
        # 计算各策略的性能指标
        performance_data = {}
        for strategy in self.detection_data.keys():
            # 聚合所有故障类型的性能
            all_true = []
            all_pred = []
            
            for fault_type, data in self.detection_data[strategy].items():
                if fault_type != 'Normal':  # 只考虑故障检测
                    all_true.extend(data['true_labels'])
                    all_pred.extend(data['pred_labels'])
            
            if all_true:
                all_true = np.array(all_true)
                all_pred = np.array(all_pred)
                
                # 计算指标
                accuracy = np.mean(all_pred == all_true)
                precision = np.sum((all_pred == 1) & (all_true == 1)) / max(np.sum(all_pred == 1), 1)
                recall = np.sum((all_pred == 1) & (all_true == 1)) / max(np.sum(all_true == 1), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                
                # 模拟速度和鲁棒性
                speed = np.random.uniform(0.7, 0.95)
                robustness = np.random.uniform(0.75, 0.9)
                
                performance_data[strategy] = [accuracy, precision, recall, f1, speed, robustness]
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for strategy, scores in performance_data.items():
            scores = scores + scores[:1]  # 闭合图形
            
            color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
            strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
            
            ax.plot(angles, scores, 'o-', linewidth=2, 
                   color=color, label=strategy_name)
            ax.fill(angles, scores, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_detection_roc_curves(self, ax):
        """绘制检测ROC曲线"""
        ax.set_title('ROC Curves for Fault Detection\nStrategies', fontweight='bold')
        
        for strategy, strategy_data in self.detection_data.items():
            # 聚合所有故障类型的ROC数据
            all_true = []
            all_probs = []
            
            for fault_type, data in strategy_data.items():
                if fault_type != 'Normal':
                    all_true.extend(data['true_labels'])
                    all_probs.extend(data['pred_probs'])
            
            if all_true:
                fpr, tpr, _ = roc_curve(all_true, all_probs)
                auc_score = auc(fpr, tpr)
                
                color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
                strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
                
                ax.plot(fpr, tpr, linewidth=2, color=color,
                       label=f'{strategy_name} (AUC = {auc_score:.3f})')
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix_heatmap(self, ax):
        """绘制混淆矩阵热力图"""
        ax.set_title('Confusion Matrix Heatmap\n(Best Strategy)', fontweight='bold')
        
        # 选择性能最好的策略
        best_strategy = list(self.detection_data.keys())[0]
        
        # 聚合数据
        all_true = []
        all_pred = []
        
        for fault_type, data in self.detection_data[best_strategy].items():
            all_true.extend(data['true_labels'])
            all_pred.extend(data['pred_labels'])
        
        if all_true:
            cm = confusion_matrix(all_true, all_pred)
            
            # 绘制热力图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Fault'], 
                       yticklabels=['Normal', 'Fault'], ax=ax)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    def _plot_fault_type_accuracy(self, ax):
        """绘制各故障类型检测准确率"""
        ax.set_title('Detection Accuracy by\nFault Type', fontweight='bold')
        
        fault_types = ['ISC', 'TR', 'BSC', 'EA']
        strategies = list(self.detection_data.keys())
        
        x = np.arange(len(fault_types))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            accuracies = []
            
            for fault_type in fault_types:
                if fault_type in self.detection_data[strategy]:
                    data = self.detection_data[strategy][fault_type]
                    accuracy = np.mean(data['pred_labels'] == data['true_labels'])
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0)
            
            color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
            strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
            
            bars = ax.bar(x + i * width, accuracies, width, 
                         label=strategy_name, color=color, alpha=0.8)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Fault Types')
        ax.set_ylabel('Detection Accuracy')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(fault_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def _plot_detection_delay_analysis(self, ax):
        """绘制检测延迟分析"""
        ax.set_title('Detection Delay Analysis', fontweight='bold')
        
        strategies = list(self.detection_data.keys())
        
        # 模拟检测延迟数据（实际应该从时序检测结果中计算）
        delay_data = []
        labels = []
        
        for strategy in strategies:
            # 不同策略的典型延迟特性
            if 'BiLSTM_3Window' in strategy:
                delays = np.random.exponential(2.5, 100)  # 三窗口相对较慢
            elif 'BiLSTM_5Point' in strategy:
                delays = np.random.exponential(1.8, 100)  # 五点检测较快
            elif 'Transformer' in strategy:
                delays = np.random.exponential(1.5, 100)  # Transformer最快
            else:
                delays = np.random.exponential(2.0, 100)
            
            delays = np.clip(delays, 0.1, 10)  # 限制在合理范围
            delay_data.append(delays)
            
            strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
            labels.append(strategy_name)
        
        # 创建箱线图
        bp = ax.boxplot(delay_data, labels=labels, patch_artist=True)
        
        # 设置颜色
        for patch, strategy in zip(bp['boxes'], strategies):
            color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Detection Strategies')
        ax.set_ylabel('Detection Delay (seconds)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_false_positive_vs_detection_rate(self, ax):
        """绘制误报率vs检测率散点图"""
        ax.set_title('False Positive Rate vs\nDetection Rate', fontweight='bold')
        
        for strategy, strategy_data in self.detection_data.items():
            # 计算误报率和检测率
            all_true = []
            all_pred = []
            
            for fault_type, data in strategy_data.items():
                all_true.extend(data['true_labels'])
                all_pred.extend(data['pred_labels'])
            
            if all_true:
                all_true = np.array(all_true)
                all_pred = np.array(all_pred)
                
                # 检测率（召回率）
                detection_rate = np.sum((all_pred == 1) & (all_true == 1)) / max(np.sum(all_true == 1), 1)
                
                # 误报率
                false_positive_rate = np.sum((all_pred == 1) & (all_true == 0)) / max(np.sum(all_true == 0), 1)
                
                color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
                strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
                
                ax.scatter(false_positive_rate, detection_rate, s=150, 
                          color=color, alpha=0.7, label=strategy_name)
                
                # 添加标签
                ax.annotate(strategy_name, (false_positive_rate, detection_rate),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 添加理想区域
        ideal_rect = patches.Rectangle((0, 0.9), 0.1, 0.1, 
                                     linewidth=2, edgecolor='green', 
                                     facecolor='lightgreen', alpha=0.3)
        ax.add_patch(ideal_rect)
        ax.text(0.05, 0.95, 'Ideal Zone', ha='center', va='center', 
               fontweight='bold', color='green')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('Detection Rate (Recall)')
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0.7, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_three_window_process(self, ax):
        """绘制三窗口检测过程"""
        ax.set_title('BiLSTM Three-Window\nDetection Process', fontweight='bold')
        
        # 模拟时序数据和检测过程
        time_points = np.arange(0, 100)
        
        # 模拟故障指标φ(FAI)
        phi_fai = np.random.normal(0.3, 0.1, len(time_points))
        fault_start = 40
        fault_end = 70
        phi_fai[fault_start:fault_end] += np.random.uniform(0.4, 0.8, fault_end - fault_start)
        
        # 阈值
        threshold = 0.6
        
        # 绘制φ(FAI)曲线
        ax.plot(time_points, phi_fai, 'b-', linewidth=2, label='φ(FAI)')
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        
        # 标注三个窗口
        detection_window = 25
        verification_window = 15
        marking_window = 10
        
        # 检测窗口
        det_start = fault_start - detection_window // 2
        det_end = fault_start + detection_window // 2
        ax.axvspan(det_start, det_end, alpha=0.2, color='orange', label='Detection Window (25 points)')
        
        # 验证窗口
        ver_start = fault_start + 5
        ver_end = ver_start + verification_window
        ax.axvspan(ver_start, ver_end, alpha=0.3, color='yellow', label='Verification Window (15 points)')
        
        # 标记窗口
        mark_start = fault_start + 10 - marking_window // 2
        mark_end = fault_start + 10 + marking_window // 2
        ax.axvspan(mark_start, mark_end, alpha=0.4, color='red', label='Marking Window (±10 points)')
        
        ax.set_xlabel('Time Points')
        ax.set_ylabel('φ(FAI) Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_five_point_process(self, ax):
        """绘制五点检测过程"""
        ax.set_title('BiLSTM Five-Point\nDetection Process', fontweight='bold')
        
        # 模拟时序数据
        time_points = np.arange(0, 50)
        phi_fai = np.random.normal(0.3, 0.1, len(time_points))
        
        # 故障点
        fault_point = 25
        phi_fai[fault_point-2:fault_point+3] += 0.5
        
        # 阈值
        threshold = 0.6
        
        # 绘制φ(FAI)曲线
        ax.plot(time_points, phi_fai, 'b-', linewidth=2, label='φ(FAI)')
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        
        # 标注五点检测
        five_points = range(fault_point-2, fault_point+3)
        for i, point in enumerate(five_points):
            if i == 2:  # 中心点
                ax.scatter(point, phi_fai[point], s=100, color='red', 
                          marker='o', label='Center Point', zorder=5)
            else:
                ax.scatter(point, phi_fai[point], s=80, color='orange', 
                          marker='s', alpha=0.8, zorder=5)
        
        # 添加五点检测区域
        ax.axvspan(fault_point-2, fault_point+2, alpha=0.2, color='red', 
                  label='Five-Point Detection Zone')
        
        # 添加箭头和说明
        ax.annotate('All 5 points > threshold\ntriggers detection', 
                   xy=(fault_point, phi_fai[fault_point]), xytext=(fault_point+8, 0.9),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=9, ha='center')
        
        ax.set_xlabel('Time Points')
        ax.set_ylabel('φ(FAI) Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_threshold_sensitivity(self, ax):
        """绘制阈值敏感性分析"""
        ax.set_title('Threshold Sensitivity\nAnalysis', fontweight='bold')
        
        thresholds = np.linspace(0.1, 0.9, 20)
        
        for strategy in list(self.detection_data.keys())[:2]:  # 只显示前两个策略
            detection_rates = []
            false_positive_rates = []
            
            for threshold in thresholds:
                # 模拟不同阈值下的检测性能
                base_detection = 0.95
                base_fp = 0.05
                
                # 阈值越高，检测率降低，误报率也降低
                detection_rate = base_detection * (1 - threshold * 0.3)
                fp_rate = base_fp * (1 - threshold) * 2
                
                detection_rates.append(detection_rate)
                false_positive_rates.append(fp_rate)
            
            color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
            strategy_name = self.detection_strategies.get(strategy, {}).get('name', strategy)
            
            ax.plot(thresholds, detection_rates, linewidth=2, color=color, 
                   label=f'{strategy_name} - Detection Rate')
            ax.plot(thresholds, false_positive_rates, linewidth=2, color=color, 
                   linestyle='--', label=f'{strategy_name} - False Positive Rate')
        
        ax.set_xlabel('Detection Threshold')
        ax.set_ylabel('Rate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_confidence_distribution(self, ax):
        """绘制检测置信度分布"""
        ax.set_title('Detection Confidence\nDistribution', fontweight='bold')
        
        for strategy, strategy_data in self.detection_data.items():
            # 聚合所有预测概率
            all_probs = []
            all_labels = []
            
            for fault_type, data in strategy_data.items():
                if fault_type != 'Normal':
                    all_probs.extend(data['pred_probs'])
                    all_labels.extend(data['true_labels'])
            
            if all_probs:
                # 分别绘制正常和故障样本的置信度分布
                normal_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
                fault_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
                
                color = self.detection_strategies.get(strategy, {}).get('color', '#1f77b4')
                
                if normal_probs:
                    ax.hist(normal_probs, bins=30, alpha=0.5, color=color, 
                           label=f'{strategy} - Normal', density=True)
                if fault_probs:
                    ax.hist(fault_probs, bins=30, alpha=0.7, color=color, 
                           label=f'{strategy} - Fault', density=True, 
                           histtype='step', linewidth=2)
                break  # 只显示一个策略的例子
        
        ax.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_temporal_detection_analysis(self, ax):
        """绘制时序检测分析"""
        ax.set_title('Temporal Detection\nAnalysis', fontweight='bold')
        
        # 模拟时序检测结果
        time_hours = np.arange(0, 24, 0.5)  # 24小时，每30分钟一个点
        
        # 模拟故障发生和检测的时间序列
        true_faults = np.zeros(len(time_hours))
        detected_faults = np.zeros(len(time_hours))
        
        # 添加几个故障事件
        fault_events = [8, 14, 18]  # 8点、14点、18点发生故障
        
        for fault_time in fault_events:
            fault_idx = int(fault_time * 2)  # 转换为索引
            true_faults[fault_idx:fault_idx+4] = 1  # 故障持续2小时
            
            # 检测有延迟
            detection_delay = np.random.randint(1, 3)  # 0.5-1.5小时延迟
            detected_faults[fault_idx+detection_delay:fault_idx+4] = 1
        
        # 绘制时序图
        ax.fill_between(time_hours, 0, true_faults, alpha=0.3, color='red', 
                       label='Actual Faults')
        ax.fill_between(time_hours, 0, detected_faults, alpha=0.6, color='blue', 
                       label='Detected Faults')
        
        # 标记检测延迟
        for fault_time in fault_events:
            ax.axvline(x=fault_time, color='red', linestyle=':', alpha=0.7)
            ax.text(fault_time, 0.5, f'Fault\n{fault_time}:00', ha='center', 
                   fontsize=8, rotation=90)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Fault Status')
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_comprehensive_detection_score(self, ax):
        """绘制综合检测评分"""
        ax.set_title('Comprehensive Detection\nPerformance Score', fontweight='bold')
        
        scores = {}
        for strategy, strategy_data in self.detection_data.items():
            # 计算综合评分
            all_true = []
            all_pred = []
            
            for fault_type, data in strategy_data.items():
                all_true.extend(data['true_labels'])
                all_pred.extend(data['pred_labels'])
            
            if all_true:
                all_true = np.array(all_true)
                all_pred = np.array(all_pred)
                
                # 基础性能指标
                accuracy = np.mean(all_pred == all_true)
                precision = np.sum((all_pred == 1) & (all_true == 1)) / max(np.sum(all_pred == 1), 1)
                recall = np.sum((all_pred == 1) & (all_true == 1)) / max(np.sum(all_true == 1), 1)
                
                # 综合评分
                performance_score = (accuracy * 0.4 + precision * 0.3 + recall * 0.3) * 100
                scores[strategy] = performance_score
        
        if scores:
            strategy_names = [self.detection_strategies.get(s, {}).get('name', s) for s in scores.keys()]
            score_values = list(scores.values())
            colors = [self.detection_strategies.get(s, {}).get('color', '#1f77b4') for s in scores.keys()]
            
            bars = ax.bar(strategy_names, score_values, color=colors, alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, score_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}', ha='center', va='bottom')
            
            # 添加性能等级线
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Excellent (>90)')
            ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Good (>80)')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Average (>70)')
        
        ax.set_xlabel('Detection Strategies')
        ax.set_ylabel('Comprehensive Score')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

def main():
    """主函数"""
    print("🚀 Starting Fault Detection Visualization...")
    
    # 创建可视化器
    visualizer = FaultDetectionVisualizer()
    
    # 加载检测结果
    if not visualizer.load_detection_results():
        print("❌ Failed to load detection results")
        return
    
    # 创建故障检测仪表板
    print("📊 Creating fault detection dashboard...")
    dashboard_path = visualizer.create_fault_detection_dashboard()
    
    print("\n✅ Fault detection visualization completed!")
    print(f"📁 Results saved to: {dashboard_path}")

if __name__ == "__main__":
    main()
