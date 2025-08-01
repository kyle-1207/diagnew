import matplotlib.pyplot as plt
import numpy as np

# 数据准备
epochs = [0, 50, 100, 150, 200, 250]

# Transformer模型数据
transformer_mcae1_losses = [0.003703, 0.000923, 0.000400, 0.000306, 0.000269, 0.000247]
transformer_mcae2_losses = [0.028183, 0.005062, 0.005006, 0.004989, 0.004980, 0.004974]

# BiLSTM模型数据
bilstm_mcae1_losses = [0.0032, 0.0007, 0.0005, 0.0005, 0.0005, 0.0004]
bilstm_mcae2_losses = [0.0381, 0.0314, 0.0303, 0.0297, 0.0294, 0.0292]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 第一组MC-AE对比图
ax1.plot(epochs, transformer_mcae1_losses, 'b-o', linewidth=2, label='Transformer MC-AE1')
ax1.plot(epochs, bilstm_mcae1_losses, 'r--s', linewidth=2, label='BiLSTM MC-AE1')
ax1.set_title('第一组MC-AE训练损失对比\nFirst Group MC-AE Training Loss Comparison', pad=20)
ax1.set_xlabel('训练轮数 / Epochs')
ax1.set_ylabel('损失值 / Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_yscale('log')

# 第二组MC-AE对比图
ax2.plot(epochs, transformer_mcae2_losses, 'g-o', linewidth=2, label='Transformer MC-AE2')
ax2.plot(epochs, bilstm_mcae2_losses, 'm--s', linewidth=2, label='BiLSTM MC-AE2')
ax2.set_title('第二组MC-AE训练损失对比\nSecond Group MC-AE Training Loss Comparison', pad=20)
ax2.set_xlabel('训练轮数 / Epochs')
ax2.set_ylabel('损失值 / Loss')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_yscale('log')

# 添加总标题
fig.suptitle('Transformer vs BiLSTM: MC-AE训练损失对比\nTransformer vs BiLSTM: MC-AE Training Loss Comparison', 
             fontsize=14, y=1.05)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('mcae_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 对比图已保存为 mcae_comparison.png")

# 计算和显示性能差异
print("\n📊 性能对比分析:")
print("\n第一组MC-AE (MC-AE1):")
final_diff1 = (transformer_mcae1_losses[-1] - bilstm_mcae1_losses[-1]) / bilstm_mcae1_losses[-1] * 100
print(f"最终损失值差异: {abs(final_diff1):.2f}%")
print(f"Transformer: {transformer_mcae1_losses[-1]:.6f}")
print(f"BiLSTM: {bilstm_mcae1_losses[-1]:.6f}")

print("\n第二组MC-AE (MC-AE2):")
final_diff2 = (transformer_mcae2_losses[-1] - bilstm_mcae2_losses[-1]) / bilstm_mcae2_losses[-1] * 100
print(f"最终损失值差异: {abs(final_diff2):.2f}%")
print(f"Transformer: {transformer_mcae2_losses[-1]:.6f}")
print(f"BiLSTM: {bilstm_mcae2_losses[-1]:.6f}")