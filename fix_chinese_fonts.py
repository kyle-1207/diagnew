#!/usr/bin/env python3
"""
修复中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def install_chinese_fonts():
    """安装中文字体"""
    print("🔧 安装中文字体...")
    
    # 常见中文字体列表
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi',  # Windows
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN',  # Linux
        'PingFang SC', 'Hiragino Sans GB', 'STHeiti'  # macOS
    ]
    
    # 检查系统字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"系统可用字体数量: {len(available_fonts)}")
    
    # 查找中文字体
    found_chinese_fonts = []
    for font in chinese_fonts:
        if font in available_fonts:
            found_chinese_fonts.append(font)
            print(f"✅ 找到中文字体: {font}")
    
    if not found_chinese_fonts:
        print("❌ 未找到中文字体，尝试安装...")
        
        # 尝试安装字体
        try:
            import subprocess
            # 安装文泉驿微米黑字体
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-wqy-microhei'], check=True)
            print("✅ 字体安装完成")
        except Exception as e:
            print(f"❌ 字体安装失败: {e}")
    
    return found_chinese_fonts

def configure_matplotlib_fonts():
    """配置matplotlib字体"""
    print("🎨 配置matplotlib字体...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = [
        'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 
        'Noto Sans CJK SC', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试中文显示
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '中文测试\nChinese Test', 
            fontsize=20, ha='center', va='center')
    ax.set_title('字体测试 Font Test')
    ax.set_xlabel('X轴 X-Axis')
    ax.set_ylabel('Y轴 Y-Axis')
    
    plt.savefig('font_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 字体配置完成，测试图片保存为: font_test.png")

def fix_plot_labels():
    """修复图表标签"""
    print("📊 修复图表标签...")
    
    # 定义中文标签映射
    label_mapping = {
        'accuracy': '准确率',
        'precision': '精确率', 
        'recall': '召回率',
        'f1_score': 'F1分数',
        'specificity': '特异性',
        'tpr': '真正例率',
        'fpr': '假正例率',
        'avg_fai_normal': '平均φ(正常)',
        'avg_fai_fault': '平均φ(故障)',
        'anomaly_ratio_normal': '异常率(正常)',
        'anomaly_ratio_fault': '异常率(故障)'
    }
    
    return label_mapping

def main():
    """主函数"""
    print("🔧 修复中文字体显示问题")
    print("=" * 50)
    
    # 1. 安装字体
    fonts = install_chinese_fonts()
    
    # 2. 配置matplotlib
    configure_matplotlib_fonts()
    
    # 3. 获取标签映射
    labels = fix_plot_labels()
    
    print("\n📋 修复建议:")
    print("1. 运行此脚本修复字体问题")
    print("2. 重启Jupyter内核")
    print("3. 重新生成可视化图表")
    print("4. 检查font_test.png确认字体正常")
    
    print("\n💡 如果仍有问题:")
    print("- 手动安装字体: sudo apt-get install fonts-wqy-microhei")
    print("- 重启系统")
    print("- 使用英文标签替代")

if __name__ == "__main__":
    main() 