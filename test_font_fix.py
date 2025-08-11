#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文字体修复的简单脚本
用于验证在Linux服务器上的中文显示效果
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import platform
import numpy as np

# 全局变量：标记是否支持中文显示
CHINESE_FONT_AVAILABLE = False

def setup_chinese_fonts_strict():
    """Linux服务器环境中文字体配置（增强版）"""
    global CHINESE_FONT_AVAILABLE
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
                CHINESE_FONT_AVAILABLE = True
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
            CHINESE_FONT_AVAILABLE = True
            print(f"🔍 通过扫描找到中文字体: {chosen}")
        else:
            chosen = 'DejaVu Sans'
            CHINESE_FONT_AVAILABLE = False
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
        plt.text(0.5, 0.5, 'Test Chinese: 测试', fontsize=10)
        plt.close()
        if CHINESE_FONT_AVAILABLE:
            print("✅ 中文字体测试通过，将使用中文标签")
        else:
            print("⚠️ 中文字体测试未通过，将使用英文标签")
    except Exception as e:
        print(f"⚠️ 中文字体测试失败: {e}")
        # 降级到安全模式
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        CHINESE_FONT_AVAILABLE = False
        print("🔄 已切换到安全模式（英文标签）")
    
    return CHINESE_FONT_AVAILABLE

# 常用图表标签字典（中英文对照）
CHART_LABELS = {
    'φ指标值': 'φ Index Value',
    '触发点': 'Trigger Points', 
    '验证': 'Verified',
    '无数据': 'No Data',
    '故障区域': 'Fault Region',
    '检测过程': 'Detection Process',
    '时间步': 'Time Step',
    '测试中文': 'Test Chinese'
}

def get_chart_label(chinese_key):
    """获取图表标签（自动中英文切换）"""
    global CHINESE_FONT_AVAILABLE
    if CHINESE_FONT_AVAILABLE:
        return chinese_key
    else:
        return CHART_LABELS.get(chinese_key, chinese_key)  # 如果没有对应的英文，返回原文

def test_font_display():
    """测试字体显示效果"""
    # 初始化字体
    CHINESE_FONT_AVAILABLE = setup_chinese_fonts_strict()
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 子图1：基本文本测试
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.5, get_chart_label('测试中文'), ha='center', va='center', fontsize=14)
    ax1.set_title(f'Font Test: {get_chart_label("测试中文")}')
    ax1.set_xlabel(get_chart_label('时间步'))
    ax1.set_ylabel(get_chart_label('φ指标值'))
    
    # 子图2：图例测试  
    ax2 = axes[0, 1]
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax2.plot(x, y, label=get_chart_label('φ指标值'))
    ax2.axhline(y=0.5, linestyle='--', label=get_chart_label('触发点'))
    ax2.legend()
    ax2.set_title(get_chart_label('检测过程'))
    
    # 子图3：空数据测试
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, get_chart_label('无数据'), ha='center', va='center')
    ax3.set_title(f'Empty Data: {get_chart_label("无数据")}')
    
    # 子图4：故障区域测试
    ax4 = axes[1, 1]
    x = np.linspace(0, 10, 100)
    y = np.random.random(100)
    ax4.plot(x, y, label=get_chart_label('φ指标值'))
    ax4.axvspan(3, 7, alpha=0.3, color='red', label=get_chart_label('故障区域'))
    ax4.legend()
    ax4.set_title(get_chart_label('故障区域'))
    
    plt.tight_layout()
    
    # 保存测试图像
    output_path = '/tmp/font_test_result.png'
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 测试图表已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存图表失败: {e}")
    
    # 显示字体信息
    print(f"\n📊 字体测试结果:")
    print(f"   中文字体可用: {CHINESE_FONT_AVAILABLE}")
    print(f"   当前字体配置: {rcParams['font.sans-serif']}")
    print(f"   系统平台: {platform.system()}")
    
    plt.close()

if __name__ == "__main__":
    print("🧪 开始中文字体显示测试...")
    test_font_display()
    print("🎉 字体测试完成！")
