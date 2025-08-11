#!/usr/bin/env python3
"""
Three Models Comparison Analysis Script
Directly read saved test results for comparison visualization
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')

# Setup English fonts
def setup_english_fonts():
    """Configure English font display"""
    try:
        rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        rcParams['axes.unicode_minus'] = False
        rcParams['font.family'] = 'sans-serif'
        print("✅ Font configuration completed")
    except Exception as e:
        print(f"⚠️ Font configuration warning: {e}")
        rcParams['font.family'] = 'sans-serif'

# Execute font configuration
setup_english_fonts()

class ThreeModelComparator:
    def __init__(self, base_path=None):
        """
        Initialize Three Model Comparator
        
        Args:
            base_path: Base path for three model data, auto-detect if None
        """
        if base_path is None:
            # Auto-detect data path
            possible_paths = [
                "/mnt/bz25t/bzhy/datasave/Three_model",  # Primary server path
                "Three_model",                            # Local directory
                "../Three_model",
                "../../Three_model"
            ]
            
            self.base_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.base_path = path
                    print(f"✅ Found data directory: {path}")
                    break
            
            if self.base_path is None:
                print("⚠️ No data directory found, will use sample data")
        else:
            self.base_path = base_path
        
        # Model configurations
        self.model_configs = {
            'BiLSTM': {
                'folder': 'BILSTM',  # Match actual directory name
                'performance_file': 'bilstm_performance_metrics.json',
                'detailed_file': 'bilstm_detailed_results.pkl',
                'color': '#FF6B6B',  # Red
                'marker': 'o'
            },
            'Transformer-PN': {
                'folder': 'transformer_PN',  # Match actual directory name
                'performance_file': 'transformer_performance_metrics.json',
                'detailed_file': 'transformer_detailed_results.pkl',
                'color': '#4ECDC4',  # Cyan
                'marker': 's'
            },
            'Transformer-Positive': {
                'folder': 'transformer_positive',  # Match actual directory name
                'performance_file': 'transformer_performance_metrics.json',
                'detailed_file': 'transformer_detailed_results.pkl',
                'color': '#45B7D1',  # Blue
                'marker': '^'
            }
        }
        
        self.model_data = {}
        self.comparison_results = {}
    
    def load_all_data(self):
        """Load all three model data"""
        print("="*60)
        print("🔄 Loading three model data...")
        print("="*60)
        
        if self.base_path is None:
            print("❌ No data directory specified, cannot load data")
            print("\n💡 Try these solutions:")
            print("1. Manually create Three_model directory structure")
            print("2. Or run training scripts to generate data")
            print("3. Or specify specific data file paths")
            return False
        
        for model_name, config in self.model_configs.items():
            print(f"\n📂 Loading {model_name} data...")
            
            # Build file paths
            folder_path = os.path.join(self.base_path, config['folder'])
            performance_path = os.path.join(folder_path, config['performance_file'])
            detailed_path = os.path.join(folder_path, config['detailed_file'])
            
            print(f"   🔍 Search path: {folder_path}")
            print(f"   📊 Performance file: {performance_path}")
            print(f"   📋 Detailed file: {detailed_path}")
            
            # Check if files exist
            if not os.path.exists(performance_path):
                print(f"❌ Performance metrics file not found: {performance_path}")
                continue
            if not os.path.exists(detailed_path):
                print(f"❌ Detailed results file not found: {detailed_path}")
                continue
            
            try:
                # Load performance metrics
                with open(performance_path, 'r', encoding='utf-8') as f:
                    performance_data = json.load(f)
                
                # Load detailed results
                with open(detailed_path, 'rb') as f:
                    detailed_data = pickle.load(f)
                
                # Save data
                self.model_data[model_name] = {
                    'performance': performance_data,
                    'detailed': detailed_data,
                    'config': config
                }
                
                print(f"✅ {model_name} data loaded successfully")
                print(f"   - Performance metrics: {len(performance_data)} items")
                print(f"   - Detailed results: {len(detailed_data)} samples")
                
            except Exception as e:
                print(f"❌ {model_name} data loading failed: {e}")
        
        print(f"\n✅ Loaded {len(self.model_data)} model data")
        
        if len(self.model_data) == 0:
            print("\n💡 No model data found, possible solutions:")
            print("1. Run training scripts to generate data:")
            print("   - Linux/Train_BILSTM.py")
            print("   - Linux/Train_Transformer_HybridFeedback.py") 
            print("   - Linux/Train_Transformer_PN_HybridFeedback.py")
            print("2. Or manually create Three_model directory structure")
            print("3. Or use create_sample_data() to create sample data")
            return False
        
        return True
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        print("🎲 Creating sample data for demonstration...")
        
        # Create sample data structure
        sample_models = ['BiLSTM', 'Transformer-PN', 'Transformer-Positive']
        
        for i, model_name in enumerate(sample_models):
            # Generate random but realistic performance data
            np.random.seed(42 + i)  # Different seed for each model
            
            # Generate sample performance metrics
            accuracy = 0.85 + np.random.normal(0, 0.05) + i * 0.02
            precision = 0.83 + np.random.normal(0, 0.04) + i * 0.015
            recall = 0.82 + np.random.normal(0, 0.04) + i * 0.018
            f1 = 2 * (precision * recall) / (precision + recall)
            
            performance_data = {
                'accuracy': max(0, min(1, accuracy)),
                'precision': max(0, min(1, precision)),
                'recall': max(0, min(1, recall)),
                'f1_score': max(0, min(1, f1)),
                'auc': 0.88 + np.random.normal(0, 0.03) + i * 0.01
            }
            
            # Generate sample detailed results for ROC curve
            n_samples = 1000
            # Generate realistic probability scores
            true_labels = np.random.binomial(1, 0.5, n_samples)
            # Better models should have more separated distributions
            pos_scores = np.random.beta(2 + i, 2, np.sum(true_labels))
            neg_scores = np.random.beta(2, 2 + i, n_samples - np.sum(true_labels))
            
            probabilities = np.zeros(n_samples)
            probabilities[true_labels == 1] = pos_scores
            probabilities[true_labels == 0] = neg_scores
            
            detailed_data = {
                'predictions': probabilities.tolist(),
                'labels': true_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'true_labels': true_labels.tolist()
            }
            
            # Save to model_data
            config = self.model_configs[model_name]
            self.model_data[model_name] = {
                'performance': performance_data,
                'detailed': detailed_data,
                'config': config
            }
            
            print(f"✅ {model_name} sample data created")
            print(f"   - Accuracy: {performance_data['accuracy']:.4f}")
            print(f"   - AUC: {performance_data['auc']:.4f}")
        
        print(f"\n✅ Created sample data for {len(self.model_data)} models")
        return True
    
    def generate_roc_comparison(self, save_path="Three_model/comparison_roc_curves.png"):
        """Generate ROC curves comparison chart"""
        print("\n🎯 Generating ROC curves comparison chart...")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"📁 Created directory: {save_dir}")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in self.model_data.items():
            try:
                # Extract ROC data - check both performance and detailed results
                performance = data['performance']
                detailed = data['detailed']
                config = data['config']
                
                # Extract prediction probabilities and true labels
                all_probs = []
                all_labels = []
                
                # Debug: print data structure
                print(f"   📊 {model_name} data structure check:")
                print(f"      Performance keys: {list(performance.keys()) if isinstance(performance, dict) else 'N/A'}")
                print(f"      Detailed data type: {type(detailed)}")
                
                # Method 1: Extract from performance metrics 
                # Check for direct roc_data (BiLSTM format)
                if isinstance(performance, dict) and 'roc_data' in performance:
                    print(f"      ✅ Found ROC data in performance metrics (BiLSTM format)")
                    roc_data = performance['roc_data']
                    all_labels = roc_data.get('true_labels', [])
                    all_probs = roc_data.get('fai_values', [])
                
                # Check for Transformer format: performance["TRANSFORMER"]["roc_data"]
                elif isinstance(performance, dict) and 'TRANSFORMER' in performance:
                    transformer_data = performance['TRANSFORMER']
                    if isinstance(transformer_data, dict) and 'roc_data' in transformer_data:
                        print(f"      ✅ Found ROC data in performance metrics (Transformer format)")
                        roc_data = transformer_data['roc_data']
                        all_labels = roc_data.get('true_labels', [])
                        all_probs = roc_data.get('fai_values', [])
                
                # Method 2: Try different detailed data structures
                elif isinstance(detailed, dict):
                    # If it's a dictionary, try to extract predictions and labels
                    if 'roc_data' in detailed:
                        print(f"      ✅ Found ROC data in detailed results")
                        roc_data = detailed['roc_data']
                        all_labels = roc_data.get('true_labels', [])
                        all_probs = roc_data.get('fai_values', [])
                    elif 'predictions' in detailed and 'labels' in detailed:
                        all_probs = detailed['predictions']
                        all_labels = detailed['labels']
                    elif 'y_pred_proba' in detailed and 'y_true' in detailed:
                        all_probs = detailed['y_pred_proba']
                        all_labels = detailed['y_true']
                    elif 'probabilities' in detailed and 'true_labels' in detailed:
                        all_probs = detailed['probabilities']
                        all_labels = detailed['true_labels']
                    else:
                        print(f"   ⚠️ {model_name}: Expected data keys not found in detailed dict")
                        print(f"   Available keys: {list(detailed.keys())}")
                        continue
                        
                elif isinstance(detailed, list):
                    print(f"      🔍 Extracting from detailed results list ({len(detailed)} samples)")
                    # If it's a list, iterate through each sample
                    for sample_result in detailed:
                        if isinstance(sample_result, dict):
                            # Check for ROC data in sample performance_metrics
                            perf_metrics = sample_result.get('performance_metrics', {})
                            if 'roc_data' in perf_metrics:
                                roc_data = perf_metrics['roc_data']
                                all_labels.extend(roc_data.get('true_labels', []))
                                all_probs.extend(roc_data.get('fai_values', []))
                            elif 'probabilities' in sample_result and 'true_labels' in sample_result:
                                all_probs.extend(sample_result['probabilities'])
                                all_labels.extend(sample_result['true_labels'])
                            elif 'predictions' in sample_result and 'labels' in sample_result:
                                all_probs.extend(sample_result['predictions'])
                                all_labels.extend(sample_result['labels'])
                            # Extract from fai values and fault labels
                            elif 'fai' in sample_result and 'fault_labels' in sample_result:
                                fai_values = sample_result['fai']
                                fault_labels = sample_result['fault_labels']
                                sample_label = sample_result.get('label', 0)
                                
                                # Use point-level labels for ROC (consistent with BiLSTM script)
                                for i, (fai_val, fault_pred) in enumerate(zip(fai_values, fault_labels)):
                                    if sample_label == 0:  # Normal sample
                                        point_true_label = 0
                                    else:  # Fault sample
                                        point_true_label = fault_pred
                                    
                                    all_labels.append(point_true_label)
                                    all_probs.append(fai_val)
                else:
                    print(f"   ⚠️ {model_name}: Unsupported data format - {type(detailed)}")
                    continue
                
                if len(all_probs) > 0 and len(all_labels) > 0:
                    # Calculate ROC curve
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(all_labels, all_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store AUC in performance data for later use
                    if 'classification_metrics' in data['performance']:
                        data['performance']['classification_metrics']['auc'] = roc_auc
                    elif 'TRANSFORMER' in data['performance']:
                        # Transformer format
                        data['performance']['TRANSFORMER']['classification_metrics']['auc'] = roc_auc
                    else:
                        data['performance']['auc'] = roc_auc
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, color=config['color'], 
                            linewidth=2, marker=config['marker'], markersize=4, markevery=20,
                            label=f'{model_name} (AUC = {roc_auc:.3f})')
                    
                    print(f"   ✅ {model_name} ROC curve generated (AUC = {roc_auc:.3f}, data points: {len(all_probs)})")
                else:
                    print(f"   ❌ {model_name}: No valid data for ROC curve")
                    
            except Exception as e:
                print(f"   ❌ {model_name} ROC curve generation failed: {e}")
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random Classifier')
        
        # Set chart properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - Three Models', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Save image
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC comparison chart saved to: {save_path}")
        return save_path
    
    def generate_performance_comparison_table(self, save_path="Three_model/performance_comparison_table.png"):
        """Generate performance metrics comparison table"""
        print("\n📊 Generating performance metrics comparison table...")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Collect performance metrics
        metrics_data = []
        for model_name, data in self.model_data.items():
            performance = data['performance']
            
            # Handle different performance data structures
            if 'classification_metrics' in performance:
                # BiLSTM format: nested structure
                class_metrics = performance['classification_metrics']
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{class_metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{class_metrics.get('precision', 0):.4f}",
                    'Recall': f"{class_metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{class_metrics.get('f1_score', 0):.4f}",
                    'AUC': f"{performance.get('auc', class_metrics.get('auc', 0)):.4f}"
                })
            else:
                # Flat structure
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{performance.get('accuracy', 0):.4f}",
                    'Precision': f"{performance.get('precision', 0):.4f}",
                    'Recall': f"{performance.get('recall', 0):.4f}",
                    'F1-Score': f"{performance.get('f1_score', 0):.4f}",
                    'AUC': f"{performance.get('auc', 0):.4f}"
                })
        
        if not metrics_data:
            print("❌ No performance data available for comparison table")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Create figure and table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Color the header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(df) + 1):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Performance Metrics Comparison - Three Models', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance comparison table saved to: {save_path}")
        return save_path
    
    def generate_metrics_bar_chart(self, save_path="Three_model/metrics_bar_comparison.png"):
        """Generate bar chart for metrics comparison"""
        print("\n📈 Generating metrics bar chart comparison...")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Collect metrics data
        models = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        colors = []
        
        for model_name, data in self.model_data.items():
            performance = data['performance']
            config = data['config']
            
            models.append(model_name)
            
            # Handle different performance data structures
            if 'classification_metrics' in performance:
                # BiLSTM format: nested structure
                class_metrics = performance['classification_metrics']
                accuracy_scores.append(class_metrics.get('accuracy', 0))
                precision_scores.append(class_metrics.get('precision', 0))
                recall_scores.append(class_metrics.get('recall', 0))
                f1_scores.append(class_metrics.get('f1_score', 0))
                auc_scores.append(performance.get('auc', class_metrics.get('auc', 0)))
            elif 'TRANSFORMER' in performance:
                # Transformer format: performance["TRANSFORMER"]["classification_metrics"]
                transformer_data = performance['TRANSFORMER']
                class_metrics = transformer_data.get('classification_metrics', {})
                accuracy_scores.append(class_metrics.get('accuracy', 0))
                precision_scores.append(class_metrics.get('precision', 0))
                recall_scores.append(class_metrics.get('recall', 0))
                f1_scores.append(class_metrics.get('f1_score', 0))
                auc_scores.append(class_metrics.get('auc', 0))
            else:
                # Flat structure
                accuracy_scores.append(performance.get('accuracy', 0))
                precision_scores.append(performance.get('precision', 0))
                recall_scores.append(performance.get('recall', 0))
                f1_scores.append(performance.get('f1_score', 0))
                auc_scores.append(performance.get('auc', 0))
            
            colors.append(config['color'])
        
        if not models:
            print("❌ No model data available for bar chart")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Metrics Comparison - Three Models', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Metrics to plot
        metrics = [
            ('Accuracy', accuracy_scores),
            ('Precision', precision_scores),
            ('Recall', recall_scores),
            ('F1-Score', f1_scores),
            ('AUC', auc_scores)
        ]
        
        for i, (metric_name, scores) in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title(f'{metric_name} Comparison', fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove unused subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics bar chart saved to: {save_path}")
        return save_path

def main():
    """Main function"""
    print("🚀 Starting Three Models Comparison Analysis")
    print("="*60)
    
    # Create comparator with server path priority
    server_path = "/mnt/bz25t/bzhy/datasave/Three_model"
    if os.path.exists(server_path):
        print(f"🎯 Using server data path: {server_path}")
        comparator = ThreeModelComparator(base_path=server_path)
    else:
        print("🔍 Server path not found, auto-detecting...")
        comparator = ThreeModelComparator()
    
    # Try to load real data first
    if not comparator.load_all_data():
        print("\n📝 No real data found, creating sample data...")
        comparator.create_sample_data()
    
    # Generate all comparison visualizations
    print("\n🎨 Generating comparison visualizations...")
    
    # Generate ROC curves comparison
    roc_path = comparator.generate_roc_comparison()
    
    # Generate performance metrics table
    table_path = comparator.generate_performance_comparison_table()
    
    # Generate metrics bar chart
    bar_path = comparator.generate_metrics_bar_chart()
    
    # Summary
    print("\n📋 Generated visualization files:")
    if roc_path:
        print(f"   📈 ROC Curves: {roc_path}")
    if table_path:
        print(f"   📊 Performance Table: {table_path}")
    if bar_path:
        print(f"   📊 Metrics Bar Chart: {bar_path}")
    
    print("\n🎉 Three Models Comparison Analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
