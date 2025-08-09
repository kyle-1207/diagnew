#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positive-Negative Hybrid Feedback Training Script
Battery Fault Detection System based on Transformer

Training sample configuration:
- Train samples: 0-100 (basic training data)
- Positive feedback samples: 101-120 (normal samples, reduce false positives)
- Negative feedback samples: 340-350 (fault samples, enhance discrimination)

Model save path: /mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from datetime import datetime
import time
from tqdm import tqdm
import json
import pandas as pd

# Add source code path
sys.path.append('./源代码备份')
sys.path.append('.')

# Import necessary modules
from Function_ import *
from Class_ import *
from create_dataset import series_to_supervised
from sklearn import preprocessing
import scipy.io as scio
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*findfont.*')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress matplotlib font warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

#=================================== Configuration ===================================

def load_sample_labels():
    """Load sample label information from Labels.xls"""
    try:
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        labels_df = pd.read_excel(labels_path)
        
        # Extract normal and fault samples
        normal_samples = labels_df[labels_df['Label'] == 0]['Num'].astype(str).tolist()
        fault_samples = labels_df[labels_df['Label'] == 1]['Num'].astype(str).tolist()
        
        print(f"Sample labels loaded from Labels.xls:")
        print(f"   Normal samples: {len(normal_samples)}")
        print(f"   Fault samples: {len(fault_samples)}")
        print(f"   Total samples: {len(labels_df)}")
        
        return normal_samples, fault_samples, labels_df
    except Exception as e:
        print(f"Failed to load Labels.xls: {e}")
        print("Using default sample configuration")
        # Return default config
        normal_samples = [str(i) for i in range(0, 50)]
        fault_samples = [str(i) for i in range(340, 360)]
        return normal_samples, fault_samples, None

# Load sample labels
normal_samples, fault_samples, labels_df = load_sample_labels()

# Positive-Negative hybrid feedback training configuration
PN_HYBRID_FEEDBACK_CONFIG = {
    # Sample configuration (dynamically loaded from Labels.xls)
    'train_samples': normal_samples,  # Use all normal samples
    'positive_feedback_samples': normal_samples[100:120] if len(normal_samples) > 100 else normal_samples[-20:],
    'negative_feedback_samples': fault_samples[:20] if len(fault_samples) >= 20 else fault_samples,
    
    # Training phase configuration
    'training_phases': {
        'phase1_transformer': {
            'epochs': 50,
            'description': 'Basic Transformer training'
        },
        'phase2_mcae': {
            'epochs': 80,
            'description': 'MC-AE training (using Transformer enhanced data)'
        },
        'phase3_feedback': {
            'epochs': 30,
            'description': 'Positive-negative feedback hybrid optimization'
        }
    },
    
    # Positive feedback configuration
    'positive_feedback': {
        'enable': True,
        'weight': 0.3,
        'start_epoch': 10,
        'frequency': 5,
        'target_fpr': 0.01,
        'adjustment_factor': 0.1
    },
    
    # Negative feedback configuration
    'negative_feedback': {
        'enable': True,
        'alpha': 0.4,
        'beta': 1.2,
        'margin': 0.15,
        'start_epoch': 20,
        'evaluation_frequency': 3,
        'min_separation': 0.1
    },
    
    # Model save path
    'save_base_path': '/mnt/bz25t/bzhy/datasave/Transformer/models/PN_model/',
    
    # 4xA100 training parameter optimization
    'batch_size': 4096,  # Increase batch_size for multi-GPU efficiency
    'learning_rate': 0.001,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

print("Training Configuration - Positive-Negative Hybrid Feedback:")
print(f"   Train samples: {len(PN_HYBRID_FEEDBACK_CONFIG['train_samples'])} (normal)")
print(f"   Positive feedback samples: {len(PN_HYBRID_FEEDBACK_CONFIG['positive_feedback_samples'])} (normal)")
print(f"   Negative feedback samples: {len(PN_HYBRID_FEEDBACK_CONFIG['negative_feedback_samples'])} (fault)")
print(f"   Model save path: {PN_HYBRID_FEEDBACK_CONFIG['save_base_path']}")

if labels_df is not None:
    print(f"\nSample Distribution Statistics:")
    print(f"   Total normal samples: {len(normal_samples)}")
    print(f"   Total fault samples: {len(fault_samples)}")
    print(f"   Train sample examples: {PN_HYBRID_FEEDBACK_CONFIG['train_samples'][:5]}...")
    print(f"   Fault sample examples: {PN_HYBRID_FEEDBACK_CONFIG['negative_feedback_samples'][:5]}...")

# Ensure save directory exists
os.makedirs(PN_HYBRID_FEEDBACK_CONFIG['save_base_path'], exist_ok=True)

#=================================== Device Configuration ===================================

device = torch.device(PN_HYBRID_FEEDBACK_CONFIG['device'])
print(f"\nDevice Configuration: {device}")

if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")

#=================================== Helper Functions ===================================

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: allocated {allocated:.2f}GB, reserved {reserved:.2f}GB")

def physics_based_data_processing_silent(data, feature_type='general'):
    """Silent physics-based data processing"""
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
        is_tensor = True
        original_dtype = data.dtype
        original_device = data.device
    else:
        data_np = np.array(data)
        is_tensor = False
    
    if data_np.size == 0:
        return data if not is_tensor else torch.tensor(data_np, dtype=original_dtype, device=original_device)
    
    # Process NaN and Inf
    for col in range(data_np.shape[1] if len(data_np.shape) > 1 else 1):
        if len(data_np.shape) > 1:
            col_data = data_np[:, col]
        else:
            col_data = data_np
            
        # Process NaN
        if np.isnan(col_data).any():
            valid_mask = ~np.isnan(col_data)
            if valid_mask.any():
                median_val = np.median(col_data[valid_mask])
                if len(data_np.shape) > 1:
                    data_np[~valid_mask, col] = median_val
                else:
                    data_np[~valid_mask] = median_val
        
        # Process Inf
        if np.isinf(col_data).any():
            finite_mask = np.isfinite(col_data)
            if finite_mask.any():
                max_finite = np.max(col_data[finite_mask])
                min_finite = np.min(col_data[finite_mask])
                if len(data_np.shape) > 1:
                    data_np[col_data == np.inf, col] = max_finite
                    data_np[col_data == -np.inf, col] = min_finite
                else:
                    data_np[col_data == np.inf] = max_finite
                    data_np[col_data == -np.inf] = min_finite
    
    # Apply physics constraints
    if feature_type == 'voltage':
        data_np = np.clip(data_np, 2.5, 4.2)
    elif feature_type == 'soc':
        data_np = np.clip(data_np, 0.0, 1.0)
    elif feature_type == 'temperature':
        data_np = np.clip(data_np, -40, 80)
    
    if is_tensor:
        return torch.tensor(data_np, dtype=original_dtype, device=original_device)
    else:
        return data_np

#=================================== Contrastive Loss Function ===================================

class ContrastiveMCAELoss(nn.Module):
    """Contrastive learning loss function for MC-AE negative feedback training"""
    
    def __init__(self, alpha=0.4, beta=1.2, margin=0.15):
        super(ContrastiveMCAELoss, self).__init__()
        self.alpha = alpha      # Normal sample weight
        self.beta = beta        # Fault sample weight
        self.margin = margin    # Contrastive margin
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, recon_normal, target_normal, recon_fault=None, target_fault=None):
        # Normal sample reconstruction loss (want to minimize)
        positive_loss = self.mse_loss(recon_normal, target_normal)
        
        if recon_fault is not None and target_fault is not None:
            # Fault sample reconstruction loss (want to maximize, but with boundary)
            fault_loss = self.mse_loss(recon_fault, target_fault)
            
            # Contrastive loss: encourage fault samples to have higher reconstruction error
            negative_loss = torch.clamp(self.margin - fault_loss, min=0.0)
            
            # Total loss
            total_loss = self.alpha * positive_loss + self.beta * negative_loss
            
            return total_loss, positive_loss, negative_loss
        else:
            return positive_loss, positive_loss, torch.tensor(0.0, device=positive_loss.device)

#=================================== Transformer Model ===================================

class TransformerPredictor(nn.Module):
    """Transformer-based prediction model"""
    
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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

def load_sample_data(sample_id, data_type='train'):
    """Load single sample data"""
    try:
        # Load all samples from server QAS directory
        base_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS'
        sample_path = f"{base_path}/{sample_id}"
        
        # Check if files exist
        required_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl', 'targets.pkl']
        for file_name in required_files:
            file_path = f"{sample_path}/{file_name}"
            if not os.path.exists(file_path):
                print(f"   File not found: {file_path}")
                return None
        
        # Load data files
        vin_1 = pickle.load(open(f"{sample_path}/vin_1.pkl", 'rb'))
        vin_2 = pickle.load(open(f"{sample_path}/vin_2.pkl", 'rb'))
        vin_3 = pickle.load(open(f"{sample_path}/vin_3.pkl", 'rb'))
        targets = pickle.load(open(f"{sample_path}/targets.pkl", 'rb'))
        
        # Process targets data special case
        if isinstance(targets, dict):
            if 'terminal_voltages' in targets and 'pack_socs' in targets:
                pass  # Keep dict format
            elif 'data' in targets:
                targets = targets['data']
            elif len(targets) == 1:
                key = list(targets.keys())[0]
                targets = targets[key]
        
        # Clean data for PyTorch compatibility
        def clean_data_for_torch(data, data_name):
            if data is None:
                return None
            
            if hasattr(data, 'detach'):
                data_np = data.detach().cpu().numpy()
            else:
                data_np = np.array(data)
            
            if data_np.dtype == np.object_:
                try:
                    if data_np.ndim == 1:
                        cleaned = []
                        for item in data_np:
                            try:
                                if isinstance(item, (int, float, np.integer, np.floating)):
                                    cleaned.append(float(item))
                                elif hasattr(item, 'item'):
                                    cleaned.append(float(item.item()))
                                else:
                                    cleaned.append(0.0)
                            except:
                                cleaned.append(0.0)
                        data_np = np.array(cleaned, dtype=np.float32)
                    else:
                        original_shape = data_np.shape
                        flat_cleaned = []
                        for item in data_np.flat:
                            try:
                                if isinstance(item, (int, float, np.integer, np.floating)):
                                    flat_cleaned.append(float(item))
                                elif hasattr(item, 'item'):
                                    flat_cleaned.append(float(item.item()))
                                else:
                                    flat_cleaned.append(0.0)
                            except:
                                flat_cleaned.append(0.0)
                        data_np = np.array(flat_cleaned, dtype=np.float32).reshape(original_shape)
                except Exception as e:
                    print(f"   Data cleaning failed for {data_name}: {e}")
                    if hasattr(data_np, 'shape'):
                        data_np = np.zeros(data_np.shape, dtype=np.float32)
                    else:
                        data_np = np.array([0.0], dtype=np.float32)
            
            if not data_np.dtype.kind in ['f', 'i', 'u', 'b']:
                try:
                    data_np = data_np.astype(np.float32)
                except Exception as e:
                    data_np = np.zeros_like(data_np, dtype=np.float32)
            
            return data_np
        
        # Clean all data
        vin_1 = clean_data_for_torch(vin_1, "vin_1")
        vin_2 = clean_data_for_torch(vin_2, "vin_2")
        vin_3 = clean_data_for_torch(vin_3, "vin_3")
        
        # Special handling for targets
        if isinstance(targets, dict):
            cleaned_targets = {}
            for key, value in targets.items():
                cleaned_targets[key] = clean_data_for_torch(value, f"targets['{key}']")
            targets = cleaned_targets
        else:
            targets = clean_data_for_torch(targets, "targets")
        
        return {
            'vin_1': vin_1,
            'vin_2': vin_2, 
            'vin_3': vin_3,
            'targets': targets,
            'sample_id': sample_id
        }
    except Exception as e:
        print(f"   Failed to load sample {sample_id}: {e}")
        return None

def verify_sample_exists(sample_id):
    """Verify if sample exists"""
    base_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS'
    sample_path = f"{base_path}/{sample_id}"
    
    required_files = ['vin_1.pkl', 'vin_2.pkl', 'vin_3.pkl', 'targets.pkl']
    for file_name in required_files:
        file_path = f"{sample_path}/{file_name}"
        if not os.path.exists(file_path):
            return False
    return True

def filter_existing_samples(sample_ids, sample_type="samples"):
    """Filter out actually existing samples"""
    print(f"Verifying {sample_type} existence...")
    existing_samples = []
    
    for sample_id in sample_ids:
        if verify_sample_exists(sample_id):
            existing_samples.append(sample_id)
    
    print(f"   Original {sample_type}: {len(sample_ids)}")
    print(f"   Existing {sample_type}: {len(existing_samples)}")
    
    if len(existing_samples) < len(sample_ids):
        missing_samples = set(sample_ids) - set(existing_samples)
        print(f"   Missing {sample_type}: {list(missing_samples)}")
    
    return existing_samples

def load_training_data(sample_ids):
    """Load training data"""
    print(f"\nLoading training data ({len(sample_ids)} samples)...")
    
    all_vin1, all_targets = [], []
    successful_samples = []
    
    for sample_id in tqdm(sample_ids[:100], desc="Loading train samples"):  # Limit to first 100 for testing
        data = load_sample_data(sample_id, 'train')
        if data is not None:
            all_vin1.append(data['vin_1'])
            all_targets.append(data['targets'])
            successful_samples.append(sample_id)
    
    if not all_vin1:
        raise ValueError("No training samples loaded successfully!")
    
    # Combine data
    processed_vin1 = []
    processed_targets = []
    
    def safe_convert_to_numpy(data, data_name):
        if data is None:
            return None
        
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                return None
        
        if data.dtype == np.object_:
            try:
                flat_data = []
                for item in data.flat:
                    try:
                        if isinstance(item, (int, float, np.integer, np.floating)):
                            flat_data.append(float(item))
                        elif hasattr(item, 'item'):
                            flat_data.append(float(item.item()))
                        else:
                            flat_data.append(0.0)
                    except:
                        flat_data.append(0.0)
                
                data = np.array(flat_data, dtype=np.float32).reshape(data.shape)
            
            except Exception as e:
                data = np.zeros(data.shape, dtype=np.float32)
        
        if data.dtype.kind not in ['f', 'i', 'u', 'b']:
            try:
                data = data.astype(np.float32)
            except Exception as e:
                data = np.zeros_like(data, dtype=np.float32)
        
        return data
    
    for i, (vin1, targets) in enumerate(zip(all_vin1, all_targets)):
        vin1_converted = safe_convert_to_numpy(vin1, f"sample{i}_vin1")
        
        if isinstance(targets, dict):
            if 'terminal_voltages' in targets and 'pack_socs' in targets:
                try:
                    voltages = safe_convert_to_numpy(targets['terminal_voltages'], f"sample{i}_voltages")
                    socs = safe_convert_to_numpy(targets['pack_socs'], f"sample{i}_socs")
                    
                    if voltages is not None and socs is not None:
                        min_len = min(len(voltages), len(socs))
                        targets_converted = np.column_stack([voltages[:min_len], socs[:min_len]])
                    else:
                        continue
                except Exception as e:
                    continue
            else:
                continue
        else:
            targets_converted = safe_convert_to_numpy(targets, f"sample{i}_targets")
        
        if vin1_converted is not None and targets_converted is not None:
            # Ensure 2D
            if vin1_converted.ndim == 1:
                vin1_converted = vin1_converted.reshape(1, -1)
            elif vin1_converted.ndim > 2:
                vin1_converted = vin1_converted.reshape(vin1_converted.shape[0], -1)
            
            if targets_converted.ndim == 1:
                targets_converted = targets_converted.reshape(1, -1)
            elif targets_converted.ndim > 2:
                targets_converted = targets_converted.reshape(targets_converted.shape[0], -1)
            
            processed_vin1.append(vin1_converted)
            processed_targets.append(targets_converted)
    
    try:
        vin1_combined = np.vstack(processed_vin1)
        targets_combined = np.vstack(processed_targets)
    except ValueError as e:
        vin1_combined = np.concatenate(processed_vin1, axis=0)
        targets_combined = np.concatenate(processed_targets, axis=0)
    
    print(f"   Successfully loaded {len(successful_samples)} samples")
    print(f"   Data shapes: vin1 {vin1_combined.shape}, targets {targets_combined.shape}")
    
    return vin1_combined, targets_combined, successful_samples

def load_feedback_data(sample_ids, data_type='feedback'):
    """Load feedback data"""
    print(f"\nLoading {data_type} data ({len(sample_ids)} samples)...")
    
    all_data = []
    successful_samples = []
    
    for sample_id in tqdm(sample_ids, desc=f"Loading {data_type} samples"):
        data = load_sample_data(sample_id, 'feedback')
        if data is not None:
            all_data.append(data)
            successful_samples.append(sample_id)
    
    print(f"   Successfully loaded {len(successful_samples)} {data_type} samples")
    return all_data, successful_samples

#=================================== Dataset Classes ===================================

class TransformerDataset(Dataset):
    """Transformer training dataset"""
    
    def __init__(self, vin1_data, targets_data):
        # Ensure input data is 2D
        if isinstance(vin1_data, np.ndarray):
            if vin1_data.ndim == 1:
                vin1_data = vin1_data.reshape(1, -1)
            elif vin1_data.ndim > 2:
                vin1_data = vin1_data.reshape(vin1_data.shape[0], -1)
        
        if isinstance(targets_data, np.ndarray):
            if targets_data.ndim == 1:
                targets_data = targets_data.reshape(1, -1)
            elif targets_data.ndim > 2:
                targets_data = targets_data.reshape(targets_data.shape[0], -1)
        
        print(f"   Dataset input shapes: vin1 {np.array(vin1_data).shape}, targets {np.array(targets_data).shape}")
        
        self.vin1_data = torch.FloatTensor(vin1_data)
        self.targets_data = torch.FloatTensor(targets_data)
        
        print(f"   Dataset tensor shapes: vin1 {self.vin1_data.shape}, targets {self.targets_data.shape}")
        
        # Data processing
        self.vin1_data = physics_based_data_processing_silent(self.vin1_data, 'general')
        self.targets_data = physics_based_data_processing_silent(self.targets_data, 'general')
    
    def __len__(self):
        return len(self.vin1_data)
    
    def __getitem__(self, idx):
        return self.vin1_data[idx], self.targets_data[idx]

class MCDataset(Dataset):
    """MC-AE training dataset"""
    
    def __init__(self, x, y, z, q):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) 
        self.z = torch.FloatTensor(z)
        self.q = torch.FloatTensor(q)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]

#=================================== Main Training Function ===================================

def main():
    """Main training function"""
    print("="*80)
    print("Starting Positive-Negative Hybrid Feedback Training")
    print("="*80)
    
    config = PN_HYBRID_FEEDBACK_CONFIG
    
    #=== Stage 1: Load training data ===
    print("\n" + "="*50)
    print("Stage 1: Data Loading")
    print("="*50)
    
    # Filter existing samples
    existing_train_samples = filter_existing_samples(config['train_samples'], "train samples")
    existing_positive_samples = filter_existing_samples(config['positive_feedback_samples'], "positive feedback samples")
    existing_negative_samples = filter_existing_samples(config['negative_feedback_samples'], "negative feedback samples")
    
    # Ensure sufficient samples for training
    if len(existing_train_samples) < 10:
        print(f"ERROR: Insufficient training samples, only {len(existing_train_samples)}, recommend at least 10")
        return
    
    # Load basic training data
    train_vin1, train_targets, successful_train = load_training_data(existing_train_samples)
    
    # Load positive feedback data
    positive_data, successful_positive = load_feedback_data(
        existing_positive_samples, 'positive feedback'
    )
    
    # Load negative feedback data  
    negative_data, successful_negative = load_feedback_data(
        existing_negative_samples, 'negative feedback'
    )
    
    print(f"\nData loading completed:")
    print(f"   Train samples: {len(successful_train)}")
    print(f"   Positive feedback samples: {len(successful_positive)}") 
    print(f"   Negative feedback samples: {len(successful_negative)}")
    
    #=== Stage 2: Transformer basic training ===
    print("\n" + "="*50)
    print("Stage 2: Transformer Basic Training")
    print("="*50)
    
    # Data dimension validation and correction
    print(f"   Original data shapes: train_vin1 {train_vin1.shape}, train_targets {train_targets.shape}")
    
    # 4xA100 GPU cluster - full dataset training configuration
    print(f"   4xA100 GPU cluster environment, using full dataset training")
    print(f"   Original sample count: {train_vin1.shape[0]:,}")
    print(f"   Using all samples for large-scale training, fully utilizing GPU cluster performance")
    print(f"   Estimated batch count: {train_vin1.shape[0] // config['batch_size']:,} batches/epoch")
    
    # GPU cluster configuration display
    if torch.cuda.device_count() >= 2:
        print(f"   Detected {torch.cuda.device_count()} GPUs, enabling data parallel training")
        for i in range(min(torch.cuda.device_count(), 2)):  # Use GPU0 and GPU1
            print(f"      GPU{i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   Only detected {torch.cuda.device_count()} GPU")
    
    # Fixed model dimension configuration
    model_input_size = 7
    model_output_size = 2
    
    print(f"   Data dimension analysis:")
    print(f"      train_vin1 original shape: {train_vin1.shape}")
    print(f"      train_targets original shape: {train_targets.shape}")
    print(f"      Model expects: input_size={model_input_size}, output_size={model_output_size}")
    
    # Adjust data to match model expectations
    if train_vin1.ndim > 2:
        print(f"   Flattening vin1 from {train_vin1.shape} to 2D")
        train_vin1 = train_vin1.reshape(train_vin1.shape[0], -1)
    
    if train_targets.ndim > 2:
        print(f"   Flattening targets from {train_targets.shape} to 2D")
        train_targets = train_targets.reshape(train_targets.shape[0], -1)
    
    print(f"   Shapes after flattening: vin1 {train_vin1.shape}, targets {train_targets.shape}")
    
    # Adjust vin1 feature dimensions
    if train_vin1.shape[1] != model_input_size:
        if train_vin1.shape[1] > model_input_size:
            train_vin1 = train_vin1[:, :model_input_size]
            print(f"   Truncated vin1 to first {model_input_size} features: {train_vin1.shape}")
        else:
            padding_shape = (train_vin1.shape[0], model_input_size - train_vin1.shape[1])
            padding = np.zeros(padding_shape, dtype=train_vin1.dtype)
            train_vin1 = np.concatenate([train_vin1, padding], axis=1)
            print(f"   Padded vin1 to {model_input_size} features: {train_vin1.shape}")
    
    # Adjust targets output dimensions
    if train_targets.shape[1] != model_output_size:
        if train_targets.shape[1] > model_output_size:
            train_targets = train_targets[:, :model_output_size]
            print(f"   Truncated targets to first {model_output_size} outputs: {train_targets.shape}")
        else:
            padding_shape = (train_targets.shape[0], model_output_size - train_targets.shape[1])
            padding = np.zeros(padding_shape, dtype=train_targets.dtype)
            train_targets = np.concatenate([train_targets, padding], axis=1)
            print(f"   Padded targets to {model_output_size} outputs: {train_targets.shape}")
    
    print(f"   Final data shapes: vin1 {train_vin1.shape}, targets {train_targets.shape}")
    
    # Data quality check and cleaning
    if np.isnan(train_vin1).any() or np.isinf(train_vin1).any():
        print(f"   Cleaning vin1 anomalous values...")
        train_vin1 = np.nan_to_num(train_vin1, nan=0.0, posinf=1.0, neginf=0.0)
    
    if np.isnan(train_targets).any() or np.isinf(train_targets).any():
        print(f"   Cleaning targets anomalous values...")
        train_targets = np.nan_to_num(train_targets, nan=0.0, posinf=1.0, neginf=0.0)
    
    print(f"   vin1 range: [{train_vin1.min():.6f}, {train_vin1.max():.6f}]")
    print(f"   targets range: [{train_targets.min():.6f}, {train_targets.max():.6f}]")
    
    # Create Transformer model - using fixed dimensions
    transformer = TransformerPredictor(
        input_size=model_input_size, 
        d_model=128, 
        nhead=8, 
        num_layers=3, 
        output_size=model_output_size
    ).to(device)
    
    # Multi-GPU data parallel support
    if torch.cuda.device_count() >= 2:
        print(f"   Enabling DataParallel, using GPUs: 0, 1")
        transformer = nn.DataParallel(transformer, device_ids=[0, 1])
        print(f"   Data parallel model created, will train distributedly on 2 GPUs")
    else:
        print(f"   Single GPU model created: input_size={model_input_size}, output_size={model_output_size}")
    
    # Display model parameter count
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"   Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create data loader - multi-GPU optimization
    train_dataset = TransformerDataset(train_vin1, train_targets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"   DataLoader config: batch_size={config['batch_size']}, num_workers=8")
    
    # Training configuration
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=config['learning_rate'])
    transformer_criterion = nn.MSELoss()
    transformer_scheduler = optim.lr_scheduler.StepLR(transformer_optimizer, step_size=20, gamma=0.8)
    
    # Training loop
    transformer_losses = []
    phase1_epochs = config['training_phases']['phase1_transformer']['epochs']
    
    print(f"Starting Transformer training ({phase1_epochs} epochs)...")
    
    for epoch in range(phase1_epochs):
        transformer.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase1_epochs}")
        for batch_vin1, batch_targets in pbar:
            batch_vin1 = batch_vin1.to(device)
            batch_targets = batch_targets.to(device)
            
            # Check and fix tensor dimensions
            if batch_vin1.dim() == 1:
                batch_vin1 = batch_vin1.unsqueeze(0)
            elif batch_vin1.dim() > 2:
                batch_size = batch_vin1.size(0)
                batch_vin1 = batch_vin1.view(batch_size, -1)
            
            if batch_targets.dim() == 1:
                batch_targets = batch_targets.unsqueeze(0)
            elif batch_targets.dim() > 2:
                batch_size = batch_targets.size(0)
                batch_targets = batch_targets.view(batch_size, -1)
            
            # Forward pass
            transformer_optimizer.zero_grad()
            predictions = transformer(batch_vin1)
            loss = transformer_criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            transformer_optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = np.mean(epoch_losses)
        transformer_losses.append(avg_loss)
        transformer_scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, LR={transformer_scheduler.get_last_lr()[0]:.6f}")
            print_gpu_memory()
    
    print("Transformer basic training completed")
    
    # Save Transformer model
    transformer_save_path = os.path.join(config['save_base_path'], 'transformer_model_pn.pth')
    
    # Handle DataParallel model saving
    transformer_to_save = transformer.module if isinstance(transformer, nn.DataParallel) else transformer
    torch.save(transformer_to_save.state_dict(), transformer_save_path)
    print(f"   Model saved: {transformer_save_path}")
    
    # Save training history
    training_history = {
        'losses': transformer_losses,
        'epochs': phase1_epochs,
        'final_loss': transformer_losses[-1] if transformer_losses else 0.0,
        'model_config': {
            'input_size': model_input_size,
            'output_size': model_output_size,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3
        },
        'training_config': {
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'device': str(device)
        },
        'data_info': {
            'train_samples': len(successful_train),
            'positive_samples': len(successful_positive),
            'negative_samples': len(successful_negative),
            'data_shape': train_vin1.shape
        },
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Save to multiple locations for compatibility
    history_save_paths = [
        os.path.join(config['save_base_path'], 'pn_training_history.pkl'),
        '/mnt/bz25t/bzhy/datasave/pn_training_history.pkl',
        './pn_training_history.pkl'
    ]
    
    for history_path in history_save_paths:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)
            print(f"   Training history saved: {history_path}")
        except Exception as e:
            print(f"   Failed to save training history to {history_path}: {e}")
    
    print("\nTraining completed successfully!")
    print(f"Final Transformer loss: {transformer_losses[-1]:.6f}")
    print(f"Model saved to: {transformer_save_path}")
    print(f"Training history saved to {len([p for p in history_save_paths if os.path.exists(p)])} locations")

if __name__ == "__main__":
    main()
