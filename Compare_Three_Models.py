#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模型对比分析脚本
直接读取已保存的测试结果进行对比可视化
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 设置中文字体
def setup_chinese_fonts():
    """配置中文字体显示"""
    # 尝试系统字体
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS',
        'WenQuanYi Micro Hei', 'Source Han Sans CN'
    ]
    
    chosen = None
    for font in font_candidates:
        try:
            if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                chosen = font
                break
        except:
            continue
    
    if chosen:
        rcParams['font.sans-serif'] = [chosen]
        rcParams['axes.unicode_minus'] = False
        print(f"✅ 使用字体: {chosen}")
    else:
        print("⚠️ 未找到中文字体，使用默认字体")
        rcParams['font.sans-serif'] = ['DejaVu Sans']

# 执行字体配置
setup_chinese_fonts()

class ThreeModelComparator:
    def __init__(self, base_path=None):
        """
        初始化三模型对比器
        
        Args:
            base_path: 三个模型数据的基础路径，如果为None则自动检测
        """
        if base_path is None:
            # 自动检测数据路径
            possible_paths = [
                "Three_model",  # 当前目录下
                "/mnt/bz25t/bzhy/datasave/Three_model",  # Linux服务器路径
                "../Three_model",  # 上级目录
                "../../Three_model"  # 再上级目录
            ]
            
            self.base_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.base_path = path
                    print(f"✅ 找到数据目录: {path}")
                    break
            
            if self.base_path is None:
                print("❌ 未找到Three_model数据目录，请手动指定路径")
                print("可能的路径位置:")
                for path in possible_paths:
                    print(f"   - {path}")
        else:
            self.base_path = base_path
        self.model_configs = {
            'BiLSTM': {
                'folder': 'BILSTM',
                'performance_file': 'bilstm_performance_metrics.json',
                'detailed_file': 'bilstm_detailed_results.pkl',
                'color': '#FF6B6B',  # 红色
                'marker': 'o'
            },
            'Transformer-PN': {
                'folder': 'transformer_PN',
                'performance_file': 'transformer_performance_metrics.json',
                'detailed_file': 'transformer_detailed_results.pkl',
                'color': '#4ECDC4',  # 青色
                'marker': 's'
            },
            'Transformer-Positive': {
                'folder': 'transformer_positive',
                'performance_file': 'transformer_performance_metrics.json',
                'detailed_file': 'transformer_detailed_results.pkl',
                'color': '#45B7D1',  # 蓝色
                'marker': '^'
            }
        }
        
        self.model_data = {}
        self.comparison_results = {}
    
    def load_all_data(self):
        """加载所有三个模型的数据"""
        print("="*60)
        print("🔄 开始加载三模型数据...")
        print("="*60)
        
        if self.base_path is None:
            print("❌ 未指定数据目录，无法加载数据")
            print("\n💡 请尝试以下解决方案:")
            print("1. 手动创建 Three_model 目录结构")
            print("2. 或者运行训练脚本生成数据")
            print("3. 或者指定具体的数据文件路径")
            return False
        
        for model_name, config in self.model_configs.items():
            print(f"\n📂 加载 {model_name} 数据...")
            
            # 构建文件路径
            folder_path = os.path.join(self.base_path, config['folder'])
            performance_path = os.path.join(folder_path, config['performance_file'])
            detailed_path = os.path.join(folder_path, config['detailed_file'])
            
            print(f"   🔍 查找路径: {folder_path}")
            print(f"   📊 性能文件: {performance_path}")
            print(f"   📋 详细文件: {detailed_path}")
            
            # 检查文件是否存在
            if not os.path.exists(performance_path):
                print(f"❌ 性能指标文件不存在: {performance_path}")
                continue
            if not os.path.exists(detailed_path):
                print(f"❌ 详细结果文件不存在: {detailed_path}")
                continue
            
            try:
                # 加载性能指标
                with open(performance_path, 'r', encoding='utf-8') as f:
                    performance_data = json.load(f)
                
                # 加载详细结果
                with open(detailed_path, 'rb') as f:
                    detailed_data = pickle.load(f)
                
                # 保存数据
                self.model_data[model_name] = {
                    'performance': performance_data,
                    'detailed': detailed_data,
                    'config': config
                }
                
                print(f"✅ {model_name} 数据加载成功")
                print(f"   - 性能指标: {len(performance_data)} 项")
                print(f"   - 详细结果: {len(detailed_data)} 个样本")
                
            except Exception as e:
                print(f"❌ {model_name} 数据加载失败: {e}")
        
        print(f"\n✅ 共加载了 {len(self.model_data)} 个模型的数据")
        
        if len(self.model_data) == 0:
            print("\n💡 没有找到任何模型数据，可能的解决方案:")
            print("1. 运行训练脚本生成数据:")
            print("   - Linux/Train_BILSTM.py")
            print("   - Linux/Train_Transformer_HybridFeedback.py") 
            print("   - Linux/Train_Transformer_PN_HybridFeedback.py")
            print("2. 或者手动创建Three_model目录结构")
            print("3. 或者使用create_sample_data()创建示例数据")
        
        return len(self.model_data) > 0
    
    def create_sample_data(self, save_path="Three_model"):
        """创建示例数据结构供测试使用"""
        print(f"\n🔧 创建示例数据结构: {save_path}")
        
        # 创建目录结构
        for model_name, config in self.model_configs.items():
            folder_path = os.path.join(save_path, config['folder'])
            os.makedirs(folder_path, exist_ok=True)
            
            # 创建示例性能指标
            sample_performance = {
                'accuracy': 0.85 + np.random.random() * 0.1,
                'precision': 0.80 + np.random.random() * 0.15,
                'recall': 0.75 + np.random.random() * 0.2,
                'f1_score': 0.82 + np.random.random() * 0.12,
                'auc': 0.88 + np.random.random() * 0.1,
                'specificity': 0.83 + np.random.random() * 0.12,
                'false_positive_rate': 0.05 + np.random.random() * 0.1,
                'early_warning_rate': 0.8 + np.random.random() * 0.15,
                'detection_stability': 0.85 + np.random.random() * 0.1
            }
            
            # 保存性能指标
            performance_path = os.path.join(folder_path, config['performance_file'])
            with open(performance_path, 'w', encoding='utf-8') as f:
                json.dump(sample_performance, f, indent=2, ensure_ascii=False)
            
            # 创建示例详细结果
            sample_detailed = []
            for i in range(100):  # 100个测试样本
                sample_result = {
                    'sample_id': f'sample_{i}',
                    'true_labels': [0, 0, 0, 1, 1] if i % 2 == 0 else [1, 1, 0, 0, 1],
                    'probabilities': [0.1 + np.random.random() * 0.8 for _ in range(5)],
                    'predictions': [1 if p > 0.5 else 0 for p in [0.1 + np.random.random() * 0.8 for _ in range(5)]]
                }
                sample_detailed.append(sample_result)
            
            # 保存详细结果
            detailed_path = os.path.join(folder_path, config['detailed_file'])
            with open(detailed_path, 'wb') as f:
                pickle.dump(sample_detailed, f)
            
            print(f"   ✅ {model_name} 示例数据已创建")
        
        print(f"\n🎉 示例数据结构创建完成: {save_path}")
        print("现在可以重新运行对比分析了！")
    
    def generate_roc_comparison(self, save_path="Three_model/comparison_roc_curves.png"):
        """生成三模型ROC曲线对比图"""
        print("\n🎯 生成ROC曲线对比图...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in self.model_data.items():
            try:
                # 从详细结果中提取ROC数据
                detailed = data['detailed']
                config = data['config']
                
                # 提取所有样本的预测概率和真实标签
                all_probs = []
                all_labels = []
                
                for sample_result in detailed:
                    if 'probabilities' in sample_result and 'true_labels' in sample_result:
                        all_probs.extend(sample_result['probabilities'])
                        all_labels.extend(sample_result['true_labels'])
                
                if len(all_probs) > 0:
                    # 计算ROC曲线
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(all_labels, all_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    # 绘制ROC曲线
                    plt.plot(fpr, tpr, 
                            color=config['color'], 
                            marker=config['marker'],
                            markersize=4,
                            markevery=20,
                            linewidth=2.5,
                            label=f'{model_name} (AUC = {roc_auc:.3f})')
                else:
                    print(f"⚠️ {model_name} 缺少ROC数据")
                    
            except Exception as e:
                print(f"❌ {model_name} ROC数据处理失败: {e}")
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        
        # 设置图表属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('三模型ROC曲线对比分析', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC对比图已保存: {save_path}")
    
    def generate_performance_radar(self, save_path="Three_model/comparison_performance_radar.png"):
        """生成三模型性能雷达图对比"""
        print("\n🎯 生成性能雷达图对比...")
        
        # 定义雷达图指标
        radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 
                        'Specificity', 'Early Warning', 'Detection Stability', 'False Alarm Control']
        
        # 收集数据
        radar_data = {}
        for model_name, data in self.model_data.items():
            performance = data['performance']
            
            # 提取指标值
            values = []
            try:
                values.append(performance.get('accuracy', 0))
                values.append(performance.get('precision', 0))
                values.append(performance.get('recall', 0))
                values.append(performance.get('f1_score', 0))
                values.append(performance.get('specificity', 0))
                values.append(performance.get('early_warning_rate', 0.8))  # 默认值
                values.append(performance.get('detection_stability', 0.85))  # 默认值
                values.append(1 - performance.get('false_positive_rate', 0.1))  # 转换为控制率
                
                radar_data[model_name] = values
            except Exception as e:
                print(f"❌ {model_name} 雷达图数据提取失败: {e}")
        
        if not radar_data:
            print("❌ 无可用的雷达图数据")
            return
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合
        
        # 绘制每个模型
        for model_name, values in radar_data.items():
            config = self.model_data[model_name]['config']
            
            # 闭合数据
            values_closed = values + [values[0]]
            
            # 绘制雷达图
            ax.plot(angles, values_closed, 
                   color=config['color'], 
                   linewidth=3, 
                   label=model_name,
                   marker=config['marker'],
                   markersize=8)
            
            # 填充区域
            ax.fill(angles, values_closed, 
                   color=config['color'], 
                   alpha=0.15)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置标题和图例
        plt.title('三模型性能指标雷达图对比', fontsize=16, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 性能雷达图已保存: {save_path}")
    
    def generate_metrics_comparison(self, save_path="Three_model/comparison_metrics_bar.png"):
        """生成性能指标柱状图对比"""
        print("\n🎯 生成性能指标柱状图对比...")
        
        # 收集关键指标
        metrics_data = []
        for model_name, data in self.model_data.items():
            performance = data['performance']
            
            metrics_data.append({
                'Model': model_name,
                'Accuracy': performance.get('accuracy', 0),
                'Precision': performance.get('precision', 0),
                'Recall': performance.get('recall', 0),
                'F1 Score': performance.get('f1_score', 0),
                'AUC': performance.get('auc', 0),
                'Specificity': performance.get('specificity', 0)
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity']
        colors = [self.model_data[model]['config']['color'] for model in df['Model']]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 绘制柱状图
            bars = ax.bar(df['Model'], df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 设置图表属性
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Score', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
        
        # 总标题
        fig.suptitle('三模型性能指标详细对比', fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图片
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 性能指标对比图已保存: {save_path}")
    
    def generate_summary_report(self, save_path="Three_model/comparison_summary.txt"):
        """生成对比总结报告"""
        print("\n📊 生成对比总结报告...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("🔬 三模型性能对比分析报告")
        report_lines.append("="*80)
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 模型基本信息
        report_lines.append("📋 模型基本信息:")
        for model_name, data in self.model_data.items():
            performance = data['performance']
            detailed = data['detailed']
            
            report_lines.append(f"\n🔸 {model_name}:")
            report_lines.append(f"   - 测试样本数: {len(detailed)}")
            report_lines.append(f"   - 整体准确率: {performance.get('accuracy', 0):.4f}")
            report_lines.append(f"   - AUC值: {performance.get('auc', 0):.4f}")
            report_lines.append(f"   - F1分数: {performance.get('f1_score', 0):.4f}")
        
        # 性能排名
        report_lines.append("\n" + "="*50)
        report_lines.append("🏆 性能指标排名:")
        
        metrics_for_ranking = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']
        
        for metric in metrics_for_ranking:
            metric_values = []
            for model_name, data in self.model_data.items():
                value = data['performance'].get(metric, 0)
                metric_values.append((model_name, value))
            
            # 排序
            metric_values.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append(f"\n📊 {metric.title()}排名:")
            for rank, (model, value) in enumerate(metric_values, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                report_lines.append(f"   {medal} {rank}. {model}: {value:.4f}")
        
        # 综合评估
        report_lines.append("\n" + "="*50)
        report_lines.append("📈 综合评估建议:")
        
        # 计算综合得分
        overall_scores = {}
        for model_name, data in self.model_data.items():
            performance = data['performance']
            score = (
                performance.get('accuracy', 0) * 0.2 +
                performance.get('precision', 0) * 0.2 +
                performance.get('recall', 0) * 0.2 +
                performance.get('f1_score', 0) * 0.2 +
                performance.get('auc', 0) * 0.2
            )
            overall_scores[model_name] = score
        
        # 排序并给出建议
        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append(f"\n🎯 综合性能排名:")
        for rank, (model, score) in enumerate(ranked_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            report_lines.append(f"   {medal} {rank}. {model}: {score:.4f}")
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 同时打印到控制台
        for line in report_lines:
            print(line)
        
        print(f"\n✅ 对比报告已保存: {save_path}")
    
    def run_full_comparison(self):
        """运行完整的三模型对比分析"""
        print("🚀 开始三模型完整对比分析...")
        
        # 1. 加载数据
        if not self.load_all_data():
            print("❌ 数据加载失败，无法进行对比分析")
            return False
        
        # 2. 生成ROC对比图
        self.generate_roc_comparison()
        
        # 3. 生成性能雷达图
        self.generate_performance_radar()
        
        # 4. 生成指标柱状图
        self.generate_metrics_comparison()
        
        # 5. 生成总结报告
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("🎉 三模型对比分析完成！")
        print("="*60)
        print("生成的文件:")
        print("📊 Three_model/comparison_roc_curves.png - ROC曲线对比")
        print("🎯 Three_model/comparison_performance_radar.png - 性能雷达图")
        print("📈 Three_model/comparison_metrics_bar.png - 指标柱状图")
        print("📋 Three_model/comparison_summary.txt - 对比报告")
        
        return True

def main():
    """主函数"""
    print("🔬 三模型对比分析系统")
    print("直接读取已保存的测试结果进行对比")
    
    # 创建对比器
    comparator = ThreeModelComparator()
    
    # 如果没有找到数据，提供创建示例数据的选项
    if comparator.base_path is None:
        print("\n❓ 是否创建示例数据进行测试？")
        print("这将创建Three_model目录结构和示例数据文件")
        
        # 自动创建示例数据（用于演示）
        print("\n🔧 自动创建示例数据...")
        comparator.create_sample_data()
        
        # 重新初始化对比器
        comparator = ThreeModelComparator()
    
    # 运行完整对比
    success = comparator.run_full_comparison()
    
    if success:
        print("\n✅ 对比分析成功完成！")
    else:
        print("\n❌ 对比分析失败！")
        print("💡 如果需要重新创建示例数据，请删除Three_model文件夹后重新运行")

if __name__ == "__main__":
    main()
