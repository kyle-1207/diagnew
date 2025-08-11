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
                "Three_model",
                "/mnt/bz25t/bzhy/datasave/Three_model",
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
                'folder': 'BiLSTM',
                'performance_file': 'bilstm_performance_metrics.json',
                'detailed_file': 'bilstm_detailed_results.pkl',
                'color': '#FF6B6B',  # Red
                'marker': 'o'
            },
            'Transformer-PN': {
                'folder': 'transformer_PN',
                'performance_file': 'transformer_performance_metrics.json',
                'detailed_file': 'transformer_detailed_results.pkl',
                'color': '#4ECDC4',  # Cyan
                'marker': 's'
            },
            'Transformer-Positive': {
                'folder': 'transformer_positive',
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
                # Extract ROC data from detailed results
                detailed = data['detailed']
                config = data['config']
                
                # Extract prediction probabilities and true labels
                all_probs = []
                all_labels = []
                
                # Debug: print data structure
                print(f"   📊 {model_name} detailed data type: {type(detailed)}")
                
                # Try different data structures
                if isinstance(detailed, dict):
                    # If it's a dictionary, try to extract predictions and labels
                    if 'predictions' in detailed and 'labels' in detailed:
                        all_probs = detailed['predictions']
                        all_labels = detailed['labels']
                    elif 'y_pred_proba' in detailed and 'y_true' in detailed:
                        all_probs = detailed['y_pred_proba']
                        all_labels = detailed['y_true']
                    elif 'probabilities' in detailed and 'true_labels' in detailed:
                        all_probs = detailed['probabilities']
                        all_labels = detailed['true_labels']
                    else:
                        print(f"   ⚠️ {model_name}: Expected data keys not found")
                        print(f"   Available keys: {list(detailed.keys())}")
                        continue
                elif isinstance(detailed, list):
                    # If it's a list, iterate through each sample
                    for sample_result in detailed:
                        if isinstance(sample_result, dict):
                            if 'probabilities' in sample_result and 'true_labels' in sample_result:
                                all_probs.extend(sample_result['probabilities'])
                                all_labels.extend(sample_result['true_labels'])
                            elif 'predictions' in sample_result and 'labels' in sample_result:
                                all_probs.extend(sample_result['predictions'])
                                all_labels.extend(sample_result['labels'])
                else:
                    print(f"   ⚠️ {model_name}: Unsupported data format - {type(detailed)}")
                    continue
                
                if len(all_probs) > 0 and len(all_labels) > 0:
                    # Calculate ROC curve
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(all_labels, all_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, color=config['color'], 
                            linewidth=2, marker=config['marker'], markersize=4, markevery=20,
                            label=f'{model_name} (AUC = {roc_auc:.3f})')
                    
                    print(f"   ✅ {model_name} ROC curve generated (AUC = {roc_auc:.3f})")
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

def main():
    """Main function"""
    print("🚀 Starting Three Models Comparison Analysis")
    print("="*60)
    
    # Create comparator
    comparator = ThreeModelComparator()
    
    # Try to load real data first
    if not comparator.load_all_data():
        print("\n📝 No real data found, creating sample data...")
        comparator.create_sample_data()
    
    # Generate ROC comparison
    print("\n🎨 Generating comparison visualizations...")
    comparator.generate_roc_comparison()
    
    print("\n🎉 Analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
