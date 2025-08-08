#!/bin/bash
# Linux系统中文字体安装脚本

echo "🔧 正在安装Linux中文字体包..."

# 更新包列表
echo "📦 更新包列表..."
sudo apt-get update

# 安装中文字体包
echo "🈚 安装中文字体..."
sudo apt-get install -y \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    fonts-arphic-ukai \
    fonts-arphic-uming \
    fonts-droid-fallback \
    language-pack-zh-hans

# 更新字体缓存
echo "🔄 更新字体缓存..."
sudo fc-cache -fv

# 清理matplotlib缓存
echo "🧹 清理matplotlib缓存..."
rm -rf ~/.cache/matplotlib
rm -rf ~/.matplotlib

echo "✅ 字体安装完成！"
echo "📝 可用中文字体列表："
fc-list :lang=zh-cn family | sort | uniq

echo ""
echo "🔍 验证字体安装："
python3 -c "
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name.lower()
    if any(keyword in font_name for keyword in ['cjk', 'han', 'hei', 'kai', 'ming', 'noto', 'wenquanyi']):
        chinese_fonts.append(font.name)

if chinese_fonts:
    print('✅ 找到以下中文字体：')
    for font in set(chinese_fonts):
        print(f'   - {font}')
else:
    print('❌ 未找到中文字体')

# 测试中文显示
try:
    plt.figure(figsize=(6, 2))
    plt.text(0.5, 0.5, '中文字体测试：BiLSTM故障检测', 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.savefig('/tmp/chinese_font_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 中文显示测试通过，测试图片保存为: /tmp/chinese_font_test.png')
except Exception as e:
    print(f'❌ 中文显示测试失败: {e}')
"

echo ""
echo "🎉 安装完成！现在可以运行测试脚本了："
echo "   python3 Test_combine_BILSTMonly.py"
echo "   python3 Test_combine_transonly.py"
