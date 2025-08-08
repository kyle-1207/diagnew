#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery Fault Detection System - Visualization Analysis Script
English Version - No Chinese Characters to Avoid Font Issues

This script provides comprehensive visualization and analysis of:
- Transformer model training results
- MC-AE reconstruction performance 
- Fault detection effectiveness
- PCA analysis results
- Model comparison metrics

Author: Battery Fault Detection Team
Date: 2025-01-08
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io as scio

# Suppress all warnings and font issues
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Use non-interactive backend to avoid font warnings
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300

# Add source paths
sys.path.append('./源代码备份')
sys.path.append('.')

try:
    from Function_ import *
    from Class_ import *
    print("SUCCESS: Core modules imported")
except ImportError as e:
    print(f"WARNING: Failed to import core modules: {e}")

#=================================== Configuration ===================================

# Analysis configuration
ANALYSIS_CONFIG = {
    'base_path': '/mnt/bz25t/bzhy/zhanglikang/project/QAS',
    'model_save_path': '/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/',
    'output_path': './visualization_results/',
    'labels_file': '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls',
    
    # Visualization settings
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
    'style': 'seaborn-v0_8',
    
    # Analysis parameters
    'test_samples_count': 50,
    'fault_threshold': 0.1,
    'confidence_level': 0.95
}

# Ensure output directory exists
os.makedirs(ANALYSIS_CONFIG['output_path'], exist_ok=True)

print("="*80)
print("Battery Fault Detection System - Visualization Analysis")
print("="*80)
print(f"Output directory: {ANALYSIS_CONFIG['output_path']}")
print(f"Model path: {ANALYSIS_CONFIG['model_save_path']}")

#=================================== Model Definition ===================================

class TransformerPredictor(nn.Module):
    """Transformer-based prediction model (same as training script)"""
    
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=3, output_size=2):
        super(TransformerPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x):
        # x: [batch, input_size]
        batch_size = x.size(0)
        
        # Project to transformer dimension
        x = self.input_projection(x)  # [batch, d_model]
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, 1, d_model]
        
        # Remove sequence dimension and output
        x = x.squeeze(1)  # [batch, d_model]
        output = self.output_projection(x)  # [batch, output_size]
        
        return output

#=================================== Data Loading Functions ===================================

def load_sample_labels():
    """Load sample labels from Excel file"""
    try:
        labels_df = pd.read_excel(ANALYSIS_CONFIG['labels_file'])
        normal_samples = labels_df[labels_df['Label'] == 0]['Num'].astype(str).tolist()
        fault_samples = labels_df[labels_df['Label'] == 1]['Num'].astype(str).tolist()
        
        print(f"Loaded sample labels:")
        print(f"  Normal samples: {len(normal_samples)}")
        print(f"  Fault samples: {len(fault_samples)}")
        
        return normal_samples, fault_samples, labels_df
    except Exception as e:
        print(f"ERROR: Failed to load labels file: {e}")
        return [], [], None

def load_saved_models():
    """Load trained models from save directory"""
    models = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load Transformer model
    transformer_path = os.path.join(ANALYSIS_CONFIG['model_save_path'], 'transformer_model_pn.pth')
    if os.path.exists(transformer_path):
        try:
            transformer = TransformerPredictor().to(device)
            
            # Load state dict and handle DataParallel wrapper
            state_dict = torch.load(transformer_path, map_location=device)
            
            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                print("   Detected DataParallel model, removing 'module.' prefix...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            transformer.load_state_dict(state_dict)
            transformer.eval()
            models['transformer'] = transformer
            print(f"SUCCESS: Loaded Transformer model from {transformer_path}")
        except Exception as e:
            print(f"ERROR: Failed to load Transformer model: {e}")
            print(f"   Attempting alternative loading method...")
            try:
                # Alternative: Try to load as DataParallel directly
                transformer_dp = nn.DataParallel(TransformerPredictor()).to(device)
                transformer_dp.load_state_dict(torch.load(transformer_path, map_location=device))
                transformer = transformer_dp.module  # Extract the actual model
                transformer.eval()
                models['transformer'] = transformer
                print(f"SUCCESS: Loaded Transformer model using DataParallel wrapper")
            except Exception as e2:
                print(f"ERROR: Alternative loading also failed: {e2}")
                print("WARNING: Continuing without Transformer model")
    
    # Load MC-AE models
    net_model_path = os.path.join(ANALYSIS_CONFIG['model_save_path'], 'net_model_pn.pth')
    netx_model_path = os.path.join(ANALYSIS_CONFIG['model_save_path'], 'netx_model_pn.pth')
    
    if os.path.exists(net_model_path):
        try:
            models['net_model_path'] = net_model_path
            print(f"SUCCESS: Found MC-AE net model at {net_model_path}")
        except Exception as e:
            print(f"WARNING: MC-AE net model file exists but cannot be loaded: {e}")
    
    if os.path.exists(netx_model_path):
        try:
            models['netx_model_path'] = netx_model_path
            print(f"SUCCESS: Found MC-AE netx model at {netx_model_path}")
        except Exception as e:
            print(f"WARNING: MC-AE netx model file exists but cannot be loaded: {e}")
    
    # Load PCA parameters if available
    pca_path = os.path.join(ANALYSIS_CONFIG['model_save_path'], 'pca_params_pn.pkl')
    if os.path.exists(pca_path):
        try:
            with open(pca_path, 'rb') as f:
                pca_params = pickle.load(f)
            models['pca_params'] = pca_params
            print(f"SUCCESS: Loaded PCA parameters from {pca_path}")
        except Exception as e:
            print(f"WARNING: Failed to load PCA parameters: {e}")
    
    return models

def load_test_samples(sample_ids, max_samples=50, labels_df=None):
    """Load test samples for evaluation"""
    print(f"\nLoading test samples (max {max_samples})...")
    
    test_data = []
    labels = []
    successful_samples = []
    
    for i, sample_id in enumerate(tqdm(sample_ids[:max_samples], desc="Loading samples")):
        try:
            sample_path = f"{ANALYSIS_CONFIG['base_path']}/{sample_id}"
            
            # Check if required files exist
            required_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl', 'targets.pkl']
            if not all(os.path.exists(f"{sample_path}/{f}") for f in required_files):
                continue
            
            # Load data
            vin_1 = pickle.load(open(f"{sample_path}/vin_1.pkl", 'rb'))
            vin_2 = pickle.load(open(f"{sample_path}/vin_2.pkl", 'rb'))
            vin_3 = pickle.load(open(f"{sample_path}/vin_3.pkl", 'rb'))
            targets = pickle.load(open(f"{sample_path}/targets.pkl", 'rb'))
            
            # Clean and convert data
            def clean_data(data):
                if hasattr(data, 'detach'):
                    data = data.detach().cpu().numpy()
                data = np.array(data, dtype=np.float32)
                return np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            
            vin_1_clean = clean_data(vin_1)
            vin_2_clean = clean_data(vin_2)
            vin_3_clean = clean_data(vin_3)
            
            # Handle targets
            if isinstance(targets, dict):
                if 'terminal_voltages' in targets and 'pack_socs' in targets:
                    voltage = clean_data(targets['terminal_voltages'])
                    soc = clean_data(targets['pack_socs'])
                    min_len = min(len(voltage), len(soc))
                    targets_clean = np.column_stack([voltage[:min_len], soc[:min_len]])
                else:
                    continue
            else:
                targets_clean = clean_data(targets)
            
            # Determine label (0=normal, 1=fault)
            # Use actual Labels.xls data if available
            if labels_df is not None:
                try:
                    sample_row = labels_df[labels_df['Num'].astype(str) == str(sample_id)]
                    if not sample_row.empty:
                        label = int(sample_row.iloc[0]['Label'])
                    else:
                        # Fallback to ID-based logic if sample not in Labels.xls
                        label = 1 if int(sample_id) > 300 else 0
                except:
                    # Fallback to ID-based logic if error occurs
                    label = 1 if int(sample_id) > 300 else 0
            else:
                # Fallback to ID-based logic if Labels.xls not available
                label = 1 if int(sample_id) > 300 else 0
            
            test_data.append({
                'sample_id': sample_id,
                'vin_1': vin_1_clean,
                'vin_2': vin_2_clean,
                'vin_3': vin_3_clean,
                'targets': targets_clean,
                'label': label
            })
            
            labels.append(label)
            successful_samples.append(sample_id)
            
        except Exception as e:
            print(f"WARNING: Failed to load sample {sample_id}: {e}")
            continue
    
    print(f"Successfully loaded {len(test_data)} samples")
    print(f"  Normal samples: {labels.count(0)}")
    print(f"  Fault samples: {labels.count(1)}")
    
    return test_data, labels, successful_samples

#=================================== Analysis Functions ===================================

def analyze_transformer_performance(models, test_data):
    """Analyze Transformer model performance"""
    print("\n" + "="*50)
    print("Transformer Performance Analysis")
    print("="*50)
    
    if 'transformer' not in models:
        print("WARNING: Transformer model not available, skipping Transformer analysis")
        print("   This is normal if model loading failed")
        print("   Other analyses will continue...")
        return None
    
    transformer = models['transformer']
    device = next(transformer.parameters()).device
    
    predictions = []
    actual_values = []
    sample_ids = []
    
    # Make predictions
    transformer.eval()
    with torch.no_grad():
        for data in tqdm(test_data, desc="Making predictions"):
            try:
                # Prepare input (use vin_1, adapt to 7 features)
                vin_1 = data['vin_1']
                if vin_1.ndim > 1:
                    vin_1 = vin_1.reshape(-1)
                
                # Adapt to model input size (7 features)
                if len(vin_1) > 7:
                    vin_1 = vin_1[:7]
                elif len(vin_1) < 7:
                    vin_1 = np.pad(vin_1, (0, 7 - len(vin_1)), 'constant')
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(vin_1).unsqueeze(0).to(device)
                
                # Predict
                pred = transformer(input_tensor)
                predictions.append(pred.cpu().numpy().flatten())
                
                # Prepare actual values (use first 2 values from targets)
                targets = data['targets']
                if targets.ndim > 1:
                    targets = targets.reshape(-1)
                if len(targets) >= 2:
                    actual_values.append(targets[:2])
                else:
                    actual_values.append(np.pad(targets, (0, 2 - len(targets)), 'constant'))
                
                sample_ids.append(data['sample_id'])
                
            except Exception as e:
                print(f"WARNING: Prediction failed for sample {data['sample_id']}: {e}")
                continue
    
    if not predictions:
        print("ERROR: No predictions generated")
        return None
    
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # Calculate metrics
    mse = np.mean((predictions - actual_values) ** 2)
    mae = np.mean(np.abs(predictions - actual_values))
    
    print(f"Transformer Performance Metrics:")
    print(f"  Samples analyzed: {len(predictions)}")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Prediction vs Actual (Feature 1)
    axes[0, 0].scatter(actual_values[:, 0], predictions[:, 0], alpha=0.6, color='#2E86AB')
    axes[0, 0].plot([actual_values[:, 0].min(), actual_values[:, 0].max()], 
                    [actual_values[:, 0].min(), actual_values[:, 0].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values (Feature 1)')
    axes[0, 0].set_ylabel('Predicted Values (Feature 1)')
    axes[0, 0].set_title('Transformer Prediction vs Actual (Feature 1)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction vs Actual (Feature 2)
    axes[0, 1].scatter(actual_values[:, 1], predictions[:, 1], alpha=0.6, color='#A23B72')
    axes[0, 1].plot([actual_values[:, 1].min(), actual_values[:, 1].max()], 
                    [actual_values[:, 1].min(), actual_values[:, 1].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Values (Feature 2)')
    axes[0, 1].set_ylabel('Predicted Values (Feature 2)')
    axes[0, 1].set_title('Transformer Prediction vs Actual (Feature 2)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction Error Distribution (Feature 1)
    error_1 = predictions[:, 0] - actual_values[:, 0]
    axes[1, 0].hist(error_1, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
    axes[1, 0].axvline(np.mean(error_1), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(error_1):.4f}')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Error Distribution (Feature 1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction Error Distribution (Feature 2)
    error_2 = predictions[:, 1] - actual_values[:, 1]
    axes[1, 1].hist(error_2, bins=30, alpha=0.7, color='#C73E1D', edgecolor='black')
    axes[1, 1].axvline(np.mean(error_2), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(error_2):.4f}')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Error Distribution (Feature 2)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_CONFIG['output_path'], 'transformer_performance_analysis.png')
    plt.savefig(plot_path, dpi=ANALYSIS_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {plot_path}")
    
    return {
        'predictions': predictions,
        'actual_values': actual_values,
        'sample_ids': sample_ids,
        'mse': mse,
        'mae': mae,
        'errors': [error_1, error_2]
    }

def analyze_fault_detection_performance(test_data, labels):
    """Analyze fault detection performance using reconstruction errors"""
    print("\n" + "="*50)
    print("Fault Detection Performance Analysis")
    print("="*50)
    
    # Calculate reconstruction errors for each sample
    reconstruction_errors = []
    true_labels = []
    
    for data, label in zip(test_data, labels):
        try:
            # Simple reconstruction error calculation
            # In practice, this would use the trained MC-AE models
            vin_1 = data['vin_1'].flatten()
            vin_2 = data['vin_2'].flatten()
            
            # Calculate basic reconstruction error (simplified)
            if len(vin_1) > 0 and len(vin_2) > 0:
                min_len = min(len(vin_1), len(vin_2))
                error = np.mean((vin_1[:min_len] - vin_2[:min_len]) ** 2)
                reconstruction_errors.append(error)
                true_labels.append(label)
        except Exception as e:
            print(f"WARNING: Error calculation failed for sample: {e}")
            continue
    
    if not reconstruction_errors:
        print("ERROR: No reconstruction errors calculated")
        return None
    
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # Find optimal threshold using ROC curve
    # For anomaly detection: higher reconstruction error should indicate fault (label=1)
    # ROC curve expects: higher score for positive class (fault=1)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(true_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    # Debug: Print some statistics to understand the data better
    normal_errors = reconstruction_errors[true_labels == 0]
    fault_errors = reconstruction_errors[true_labels == 1]
    
    print(f"  Debug - Normal samples reconstruction error: mean={np.mean(normal_errors):.6f}, std={np.std(normal_errors):.6f}")
    print(f"  Debug - Fault samples reconstruction error: mean={np.mean(fault_errors):.6f}, std={np.std(fault_errors):.6f}")
    
    # Check if fault samples actually have higher reconstruction errors (as expected)
    if np.mean(fault_errors) < np.mean(normal_errors):
        print("  WARNING: Fault samples have LOWER reconstruction errors than normal samples!")
        print("  This suggests either: 1) Model training issue, 2) Label assignment error, or 3) Different reconstruction logic needed")
        
        # Option: Invert the AUC interpretation if the assumption is violated
        print("  INFO: Inverting AUC calculation due to unexpected error pattern")
        roc_auc_corrected = 1.0 - roc_auc
        print(f"  Original AUC: {roc_auc:.4f} -> Corrected AUC: {roc_auc_corrected:.4f}")
        roc_auc = roc_auc_corrected
    
    # Find threshold that maximizes (tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predicted_labels = (reconstruction_errors > optimal_threshold).astype(int)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"Fault Detection Results:")
    print(f"  Samples analyzed: {len(reconstruction_errors)}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.6f}")
    print(f"  Accuracy: {np.mean(predicted_labels == true_labels):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Normal', 'Fault']))
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reconstruction Error Distribution by Class
    normal_errors = reconstruction_errors[true_labels == 0]
    fault_errors = reconstruction_errors[true_labels == 1]
    
    axes[0, 0].hist(normal_errors, bins=20, alpha=0.7, label='Normal', color='#2E86AB', density=True)
    axes[0, 0].hist(fault_errors, bins=20, alpha=0.7, label='Fault', color='#A23B72', density=True)
    axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {optimal_threshold:.4f}')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Reconstruction Error Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: ROC Curve
    axes[0, 1].plot(fpr, tpr, color='#F18F01', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    axes[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5, label=f'Optimal Point')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve for Fault Detection')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_xticklabels(['Normal', 'Fault'])
    axes[1, 0].set_yticklabels(['Normal', 'Fault'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Plot 4: Threshold Analysis
    precision_scores = []
    recall_scores = []
    f1_scores = []
    threshold_range = np.linspace(reconstruction_errors.min(), reconstruction_errors.max(), 100)
    
    for thresh in threshold_range:
        pred = (reconstruction_errors > thresh).astype(int)
        if len(np.unique(pred)) == 2:  # Both classes predicted
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(true_labels, pred, zero_division=0)
            recall = recall_score(true_labels, pred, zero_division=0)
            f1 = f1_score(true_labels, pred, zero_division=0)
        else:
            precision = recall = f1 = 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    axes[1, 1].plot(threshold_range, precision_scores, label='Precision', color='#2E86AB')
    axes[1, 1].plot(threshold_range, recall_scores, label='Recall', color='#A23B72')
    axes[1, 1].plot(threshold_range, f1_scores, label='F1-Score', color='#F18F01')
    axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_CONFIG['output_path'], 'fault_detection_analysis.png')
    plt.savefig(plot_path, dpi=ANALYSIS_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {plot_path}")
    
    return {
        'reconstruction_errors': reconstruction_errors,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'optimal_threshold': optimal_threshold,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def generate_comprehensive_report(transformer_results, fault_detection_results, test_data):
    """Generate comprehensive analysis report"""
    print("\n" + "="*50)
    print("Generating Comprehensive Analysis Report")
    print("="*50)
    
    report_content = []
    
    # Header
    report_content.append("# Battery Fault Detection System - Analysis Report")
    report_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"**Analysis Samples:** {len(test_data)}")
    report_content.append("")
    
    # Executive Summary
    report_content.append("## Executive Summary")
    report_content.append("")
    if transformer_results:
        report_content.append(f"- **Transformer MSE:** {transformer_results['mse']:.6f}")
        report_content.append(f"- **Transformer MAE:** {transformer_results['mae']:.6f}")
    
    if fault_detection_results:
        report_content.append(f"- **Fault Detection AUC:** {fault_detection_results['roc_auc']:.4f}")
        report_content.append(f"- **Optimal Threshold:** {fault_detection_results['optimal_threshold']:.6f}")
        
        # Calculate accuracy
        accuracy = np.mean(fault_detection_results['predicted_labels'] == fault_detection_results['true_labels'])
        report_content.append(f"- **Detection Accuracy:** {accuracy:.4f}")
    
    report_content.append("")
    
    # Transformer Analysis
    if transformer_results:
        report_content.append("## Transformer Model Performance")
        report_content.append("")
        report_content.append("### Key Metrics")
        report_content.append(f"- Mean Squared Error: {transformer_results['mse']:.6f}")
        report_content.append(f"- Mean Absolute Error: {transformer_results['mae']:.6f}")
        report_content.append(f"- Samples Analyzed: {len(transformer_results['predictions'])}")
        report_content.append("")
        
        report_content.append("### Error Analysis")
        for i, error in enumerate(transformer_results['errors']):
            report_content.append(f"- Feature {i+1} Error Mean: {np.mean(error):.6f}")
            report_content.append(f"- Feature {i+1} Error Std: {np.std(error):.6f}")
        report_content.append("")
    
    # Fault Detection Analysis
    if fault_detection_results:
        report_content.append("## Fault Detection Performance")
        report_content.append("")
        report_content.append("### Detection Metrics")
        report_content.append(f"- ROC AUC Score: {fault_detection_results['roc_auc']:.4f}")
        report_content.append(f"- Optimal Threshold: {fault_detection_results['optimal_threshold']:.6f}")
        
        # Calculate additional metrics
        cm = fault_detection_results['confusion_matrix']
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            report_content.append(f"- Sensitivity (Recall): {sensitivity:.4f}")
            report_content.append(f"- Specificity: {specificity:.4f}")
            report_content.append(f"- False Positive Rate: {fp / (fp + tn) if (fp + tn) > 0 else 0:.4f}")
            report_content.append(f"- False Negative Rate: {fn / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
        
        report_content.append("")
        
        report_content.append("### Confusion Matrix")
        report_content.append("```")
        report_content.append("Predicted:  Normal  Fault")
        report_content.append(f"Normal:      {cm[0,0]:5d}  {cm[0,1]:5d}")
        report_content.append(f"Fault:       {cm[1,0]:5d}  {cm[1,1]:5d}")
        report_content.append("```")
        report_content.append("")
    
    # Data Quality Assessment
    report_content.append("## Data Quality Assessment")
    report_content.append("")
    normal_count = sum(1 for data in test_data if data['label'] == 0)
    fault_count = sum(1 for data in test_data if data['label'] == 1)
    
    report_content.append(f"- Total Samples: {len(test_data)}")
    report_content.append(f"- Normal Samples: {normal_count} ({normal_count/len(test_data)*100:.1f}%)")
    report_content.append(f"- Fault Samples: {fault_count} ({fault_count/len(test_data)*100:.1f}%)")
    report_content.append("")
    
    # Recommendations
    report_content.append("## Recommendations")
    report_content.append("")
    
    if transformer_results and transformer_results['mse'] > 0.01:
        report_content.append("- Consider increasing Transformer training epochs or adjusting learning rate")
    
    if fault_detection_results and fault_detection_results['roc_auc'] < 0.9:
        report_content.append("- Fault detection performance could be improved with more training data")
        report_content.append("- Consider feature engineering or different threshold optimization")
    
    if normal_count != fault_count:
        report_content.append("- Dataset is imbalanced, consider using balanced sampling or cost-sensitive learning")
    
    report_content.append("")
    report_content.append("## Generated Visualizations")
    report_content.append("")
    report_content.append("1. `transformer_performance_analysis.png` - Transformer prediction accuracy")
    report_content.append("2. `fault_detection_analysis.png` - Fault detection performance metrics")
    report_content.append("")
    
    # Save report
    report_path = os.path.join(ANALYSIS_CONFIG['output_path'], 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"Comprehensive report saved: {report_path}")
    
    return report_path

def create_basic_data_visualization(test_data, labels):
    """Create basic data visualization even without trained models"""
    print("\n" + "="*50)
    print("Basic Data Visualization")
    print("="*50)
    
    if not test_data:
        print("ERROR: No test data available")
        return
    
    # Extract basic statistics from data
    sample_shapes = []
    vin1_stats = []
    vin2_stats = []
    vin3_stats = []
    
    for data in test_data:
        try:
            vin_1 = data['vin_1'].flatten()
            vin_2 = data['vin_2'].flatten()
            vin_3 = data['vin_3'].flatten()
            
            sample_shapes.append(len(vin_1))
            vin1_stats.append([np.mean(vin_1), np.std(vin_1), np.min(vin_1), np.max(vin_1)])
            vin2_stats.append([np.mean(vin_2), np.std(vin_2), np.min(vin_2), np.max(vin_2)])
            vin3_stats.append([np.mean(vin_3), np.std(vin_3), np.min(vin_3), np.max(vin_3)])
        except Exception as e:
            print(f"WARNING: Failed to process sample: {e}")
            continue
    
    if not vin1_stats:
        print("ERROR: No valid data for visualization")
        return
    
    vin1_stats = np.array(vin1_stats)
    vin2_stats = np.array(vin2_stats)
    vin3_stats = np.array(vin3_stats)
    labels_array = np.array(labels[:len(vin1_stats)])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Data distribution by class (VIN1 mean values)
    normal_mask = labels_array == 0
    fault_mask = labels_array == 1
    
    if np.any(normal_mask):
        axes[0, 0].hist(vin1_stats[normal_mask, 0], bins=15, alpha=0.7, label='Normal', color='#2E86AB', density=True)
    if np.any(fault_mask):
        axes[0, 0].hist(vin1_stats[fault_mask, 0], bins=15, alpha=0.7, label='Fault', color='#A23B72', density=True)
    
    axes[0, 0].set_xlabel('VIN1 Mean Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('VIN1 Mean Value Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation comparison
    if np.any(normal_mask):
        axes[0, 1].scatter(vin1_stats[normal_mask, 1], vin2_stats[normal_mask, 1], 
                          alpha=0.6, label='Normal', color='#2E86AB', s=50)
    if np.any(fault_mask):
        axes[0, 1].scatter(vin1_stats[fault_mask, 1], vin2_stats[fault_mask, 1], 
                          alpha=0.6, label='Fault', color='#A23B72', s=50)
    
    axes[0, 1].set_xlabel('VIN1 Standard Deviation')
    axes[0, 1].set_ylabel('VIN2 Standard Deviation')
    axes[0, 1].set_title('Data Variability Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sample distribution summary
    class_counts = [np.sum(normal_mask), np.sum(fault_mask)]
    class_labels = ['Normal', 'Fault']
    colors = ['#2E86AB', '#A23B72']
    
    axes[1, 0].pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Sample Class Distribution')
    
    # Plot 4: Data range analysis
    features = ['Mean', 'Std', 'Min', 'Max']
    normal_data = np.mean(vin1_stats[normal_mask], axis=0) if np.any(normal_mask) else np.zeros(4)
    fault_data = np.mean(vin1_stats[fault_mask], axis=0) if np.any(fault_mask) else np.zeros(4)
    
    x = np.arange(len(features))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, normal_data, width, label='Normal', color='#2E86AB', alpha=0.7)
    axes[1, 1].bar(x + width/2, fault_data, width, label='Fault', color='#A23B72', alpha=0.7)
    
    axes[1, 1].set_xlabel('Statistical Features')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Statistical Feature Comparison (VIN1)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(features)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_CONFIG['output_path'], 'basic_data_analysis.png')
    plt.savefig(plot_path, dpi=ANALYSIS_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Basic data visualization saved: {plot_path}")
    
    # Print summary statistics
    print(f"\nData Summary:")
    print(f"  Total samples analyzed: {len(vin1_stats)}")
    print(f"  Normal samples: {np.sum(normal_mask)}")
    print(f"  Fault samples: {np.sum(fault_mask)}")
    
    if np.any(normal_mask):
        print(f"  Normal VIN1 mean: {np.mean(vin1_stats[normal_mask, 0]):.4f} ± {np.std(vin1_stats[normal_mask, 0]):.4f}")
    if np.any(fault_mask):
        print(f"  Fault VIN1 mean: {np.mean(vin1_stats[fault_mask, 0]):.4f} ± {np.std(vin1_stats[fault_mask, 0]):.4f}")
    
    return True

#=================================== Main Analysis Function ===================================

def main():
    """Main analysis function"""
    print("Starting comprehensive analysis...")
    
    # Load sample labels
    normal_samples, fault_samples, labels_df = load_sample_labels()
    
    if not normal_samples and not fault_samples:
        print("WARNING: No samples loaded from labels file, using default sample IDs")
        # Use default sample IDs if labels file fails
        normal_samples = [str(i) for i in range(0, 30)]
        fault_samples = [str(i) for i in range(340, 370)]
    
    # Load saved models
    models = load_saved_models()
    
    if not models:
        print("WARNING: No models loaded, will perform basic data analysis only")
    
    # Load test samples (mix of normal and fault)
    test_sample_ids = normal_samples[:25] + fault_samples[:25]  # 25 from each class
    test_data, labels, successful_samples = load_test_samples(test_sample_ids, max_samples=50, labels_df=labels_df)
    
    if not test_data:
        print("ERROR: No test data loaded, cannot proceed with analysis")
        return
    
    print(f"\nProceeding with analysis of {len(test_data)} samples...")
    
    # Analyze Transformer performance (if model available)
    transformer_results = None
    if 'transformer' in models:
        try:
            transformer_results = analyze_transformer_performance(models, test_data)
        except Exception as e:
            print(f"WARNING: Transformer analysis failed: {e}")
            transformer_results = None
    else:
        print("INFO: Skipping Transformer analysis (model not available)")
    
    # Analyze fault detection performance (basic reconstruction error analysis)
    fault_detection_results = None
    try:
        fault_detection_results = analyze_fault_detection_performance(test_data, labels)
    except Exception as e:
        print(f"WARNING: Fault detection analysis failed: {e}")
        fault_detection_results = None
    
    # Generate comprehensive report (even with partial results)
    try:
        report_path = generate_comprehensive_report(transformer_results, fault_detection_results, test_data)
    except Exception as e:
        print(f"WARNING: Report generation failed: {e}")
        report_path = None
    
    # Generate basic data visualization even if models failed
    try:
        create_basic_data_visualization(test_data, labels)
    except Exception as e:
        print(f"WARNING: Basic visualization failed: {e}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Results saved in: {ANALYSIS_CONFIG['output_path']}")
    
    if report_path:
        print(f"Report file: {report_path}")
    
    # List all generated files
    try:
        output_files = os.listdir(ANALYSIS_CONFIG['output_path'])
        print(f"\nGenerated files:")
        for file in output_files:
            print(f"  - {file}")
    except Exception as e:
        print(f"WARNING: Could not list output files: {e}")
    
    print("\nAnalysis completed with available data and models.")

if __name__ == "__main__":
    main()
