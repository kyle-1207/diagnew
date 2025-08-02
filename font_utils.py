#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体设置工具 - 解决Linux环境下matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def setup_chinese_fonts():
    """设置中文字体，如果不可用则使用英文"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
        'Noto Sans CJK JP', 'Noto Sans CJK TC', 'Source Han Sans CN',
        'Droid Sans Fallback', 'WenQuanYi Zen Hei', 'AR PL UMing CN',
        'Liberation Sans', 'DejaVu Sans'
    ]
    
    # 检查系统字体
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"🔍 系统可用字体数量: {len(system_fonts)}")
    
    # 查找可用的中文字体
    available_chinese = []
    for font in chinese_fonts:
        if font in system_fonts:
            available_chinese.append(font)
            print(f"✅ 找到字体: {font}")
    
    if available_chinese:
        # 使用第一个可用的中文字体
        plt.rcParams['font.sans-serif'] = available_chinese
        plt.rcParams['axes.unicode_minus'] = False
        print(f"🎨 使用字体: {available_chinese[0]}")
        return True
    else:
        # 没有中文字体，使用英文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️  未找到中文字体，将使用英文标签")
        return False

def get_plot_labels(use_chinese=True):
    """获取绘图标签，根据字体可用性返回中文或英文"""
    if use_chinese:
        return {
            'training_loss': '训练损失',
            'training_epochs': '训练轮数',
            'mse_loss': 'MSE损失',
            'prediction_error': '预测误差',
            'absolute_error': '绝对误差',
            'frequency': '频次',
            'reconstruction_error': '重构误差',
            'voltage_error': '电压误差',
            'soc_error': 'SOC误差',
            'error_distribution': '误差分布',
            'transformer_training': 'Transformer训练损失曲线',
            'mcae1_training': 'MC-AE1训练损失曲线',
            'mcae2_training': 'MC-AE2训练损失曲线',
            'mcae1_error': 'MC-AE1重构误差分布',
            'mcae2_error': 'MC-AE2重构误差分布'
        }
    else:
        return {
            'training_loss': 'Training Loss',
            'training_epochs': 'Training Epochs',
            'mse_loss': 'MSE Loss',
            'prediction_error': 'Prediction Error',
            'absolute_error': 'Absolute Error',
            'frequency': 'Frequency',
            'reconstruction_error': 'Reconstruction Error',
            'voltage_error': 'Voltage Error',
            'soc_error': 'SOC Error',
            'error_distribution': 'Error Distribution',
            'transformer_training': 'Transformer Training Loss',
            'mcae1_training': 'MC-AE1 Training Loss',
            'mcae2_training': 'MC-AE2 Training Loss',
            'mcae1_error': 'MC-AE1 Reconstruction Error Distribution',
            'mcae2_error': 'MC-AE2 Reconstruction Error Distribution'
        }

def setup_matplotlib_for_linux():
    """为Linux环境设置matplotlib"""
    # 设置非交互式后端
    plt.switch_backend('Agg')
    
    # 设置字体
    use_chinese = setup_chinese_fonts()
    
    # 基本设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    return use_chinese 