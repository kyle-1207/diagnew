# 中文注释：导入常用库和自定义模块
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import math
import math
from create_dataset import series_to_supervised
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs
import scipy.stats as stats
import seaborn as sns
import pickle

# GPU设备配置
import os
# 使用指定的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一张GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 打印GPU信息
if torch.cuda.is_available():
    print(f"🚀 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   当前GPU: {torch.cuda.current_device()}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  未检测到GPU，使用CPU训练")

# 中文注释：忽略警告信息
warnings.filterwarnings('ignore')

#----------------------------------------BiLSTM基准训练配置------------------------------
print("="*50)
print("BiLSTM基准训练模式")
print("直接使用原始vin_2[x[0]]和vin_3[x[0]]数据")
print("跳过Transformer训练，直接进行MC-AE训练")
print("="*50)

#----------------------------------------数据加载------------------------------
# 从Labels.xls加载训练样本ID（0-200号）
def load_train_samples():
    """从Labels.xls加载训练样本ID"""
    try:
        import pandas as pd
        labels_path = '/mnt/bz25t/bzhy/zhanglikang/project/QAS/Labels.xls'
        df = pd.read_excel(labels_path)
        
        # 提取0-200范围的样本
        all_samples = df['Num'].tolist()
        train_samples = [i for i in all_samples if 0 <= i <= 200]
        
        print(f"📋 从Labels.xls加载训练样本:")
        print(f"   训练样本范围: 0-200")
        print(f"   实际可用样本: {len(train_samples)} 个")
        print(f"   样本ID: {train_samples[:10]}..." if len(train_samples) > 10 else f"   样本ID: {train_samples}")
        
        return train_samples
    except Exception as e:
        print(f"❌ 加载Labels.xls失败: {e}")
        print("⚠️  使用默认样本范围 0-20")
        return list(range(21))

train_samples = load_train_samples()
print(f"使用QAS目录中的{len(train_samples)}个样本进行训练")

#----------------------------------------MC-AE训练数据准备（直接使用原始数据）------------------------
print("="*50)
print("阶段1: 准备MC-AE训练数据（使用原始BiLSTM数据）")
print("="*50)

# 中文注释：加载MC-AE模型输入特征（vin_2.pkl和vin_3.pkl）
# 合并所有训练样本的vin_2和vin_3数据
all_vin2_data = []
all_vin3_data = []

for sample_id in train_samples:
    vin2_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_2.pkl'
    vin3_path = f'/mnt/bz25t/bzhy/zhanglikang/project/QAS/{sample_id}/vin_3.pkl'
    
    # 加载原始vin_2和vin_3数据
    with open(vin2_path, 'rb') as file:
        vin2_data = pickle.load(file)
        print(f"原始样本 {sample_id} vin_2: {vin2_data.shape}")
    
    with open(vin3_path, 'rb') as file:
        vin3_data = pickle.load(file)
        print(f"原始样本 {sample_id} vin_3: {vin3_data.shape}")
    
    # 直接使用原始数据，不进行任何替换
    print(f"样本 {sample_id}: 使用原始BiLSTM输出数据")
    print(f"  vin_2形状: {vin2_data.shape}")
    print(f"  vin_3形状: {vin3_data.shape}")
    print(f"  原始vin_2[x[0]]范围: [{vin2_data[:, 0].min():.3f}, {vin2_data[:, 0].max():.3f}]")
    print(f"  原始vin_3[x[0]]范围: [{vin3_data[:, 0].min():.3f}, {vin3_data[:, 0].max():.3f}]")
    
    all_vin2_data.append(vin2_data)
    all_vin3_data.append(vin3_data)

# 合并数据
combined_tensor = torch.cat(all_vin2_data, dim=0)
combined_tensorx = torch.cat(all_vin3_data, dim=0)

print(f"合并后vin_2数据形状: {combined_tensor.shape}")
print(f"合并后vin_3数据形状: {combined_tensorx.shape}")

#----------------------------------------MC-AE多通道自编码器训练--------------------------
print("="*50)
print("阶段2: 训练MC-AE异常检测模型（使用原始BiLSTM数据）")
print("="*50)

# 中文注释：定义特征切片维度
# vin_2.pkl
dim_x = 2
dim_y = 110
dim_z = 110
dim_q = 3

# 中文注释：分割特征张量
x_recovered = combined_tensor[:, :dim_x]
y_recovered = combined_tensor[:, dim_x:dim_x + dim_y]
z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z]
q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:]

# vin_3.pkl
dim_x2 = 2
dim_y2 = 110
dim_z2 = 110
dim_q2= 4

x_recovered2 = combined_tensorx[:, :dim_x2]
y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]

# 训练超参数配置
EPOCH = 300
LR = 5e-4
BATCHSIZE = 1000  # 增大批次大小以提高GPU利用率

# 用于记录训练损失
train_losses_mcae1 = []
train_losses_mcae2 = []

# 中文注释：自定义多输入数据集类（本地定义，非Class_.py中的Dataset）
class Dataset(Dataset):
    def __init__(self, x, y, z, q):
        self.x = x.to(torch.double)
        self.y = y.to(torch.double)
        self.z = z.to(torch.double)
        self.q = q.to(torch.double)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]

# 中文注释：用DataLoader批量加载多通道特征数据
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=BATCHSIZE, shuffle=False)

# 中文注释：初始化MC-AE模型
net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device)
netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
l1_lambda = 0.01
loss_f = nn.MSELoss()
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_u):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        net = net.double()
        recon_im , recon_p = net(x,z,q)
        loss_u = loss_f(y,recon_im)
        total_loss += loss_u.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_u.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    train_losses_mcae1.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE1 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))

# 中文注释：全量推理，获得重构误差
train_loader2 = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=len(x_recovered), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loader2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    net = net.double()
    recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = AA - yTrainU

# 中文注释：第二组特征的MC-AE训练
train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=BATCHSIZE, shuffle=False)
optimizer = torch.optim.Adam(netx.parameters(), lr=LR)
loss_f = nn.MSELoss()
avg_loss_list_x = []
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        netx = netx.double()
        recon_im , z  = netx(x,z,q)
        loss_x = loss_f(y,recon_im)
        total_loss += loss_x.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_x.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    train_losses_mcae2.append(avg_loss)
    if epoch % 50 == 0:
        print('MC-AE2 Epoch: {:2d} | Average Loss: {:.6f}'.format(epoch, avg_loss))

train_loaderx2 = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=len(x_recovered2), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loaderx2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    netx = netx.double()
    recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = BB - yTrainX

# 中文注释：诊断特征提取与PCA分析
df_data = DiagnosisFeature(ERRORU,ERRORX)

v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data,0.95,0.95)

# 训练结束后自动保存模型和分析结果
print("="*50)
print("保存BiLSTM基准训练结果")
print("="*50)

# 绘制训练结果
print("📈 绘制BiLSTM训练曲线...")

# Linux环境字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Linux环境matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 子图1: MC-AE1训练损失曲线
ax1 = axes[0, 0]
epochs = range(1, len(train_losses_mcae1) + 1)
ax1.plot(epochs, train_losses_mcae1, 'b-', linewidth=2, label='MC-AE1 Training Loss')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('MSE损失')
ax1.set_title('MC-AE1训练损失曲线')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_yscale('log')

# 子图2: MC-AE2训练损失曲线 
ax2 = axes[0, 1]
ax2.plot(epochs, train_losses_mcae2, 'r-', linewidth=2, label='MC-AE2 Training Loss')
ax2.set_xlabel('训练轮数')
ax2.set_ylabel('MSE损失')
ax2.set_title('MC-AE2训练损失曲线')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_yscale('log')

# 子图3: MC-AE1重构误差分布
ax3 = axes[1, 0]
reconstruction_errors_1 = ERRORU.flatten()
ax3.hist(np.abs(reconstruction_errors_1), bins=50, alpha=0.7, color='blue', 
         label=f'MC-AE1重构误差 (均值: {np.mean(np.abs(reconstruction_errors_1)):.4f})')
ax3.set_xlabel('绝对重构误差')
ax3.set_ylabel('频数')
ax3.set_title('MC-AE1重构误差分布')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4: MC-AE2重构误差分布
ax4 = axes[1, 1]
reconstruction_errors_2 = ERRORX.flatten()
ax4.hist(np.abs(reconstruction_errors_2), bins=50, alpha=0.7, color='red',
         label=f'MC-AE2重构误差 (均值: {np.mean(np.abs(reconstruction_errors_2)):.4f})')
ax4.set_xlabel('绝对重构误差')
ax4.set_ylabel('频数')
ax4.set_title('MC-AE2重构误差分布')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{result_dir}/bilstm_training_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ BiLSTM训练结果图已保存: {result_dir}/bilstm_training_results.png")

# 1. 创建结果目录
result_dir = '/mnt/bz25t/bzhy/zhanglikang/project/models'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 2. 保存诊断特征DataFrame
df_data.to_excel(f'{result_dir}/diagnosis_feature_bilstm_baseline.xlsx', index=False)
df_data.to_csv(f'{result_dir}/diagnosis_feature_bilstm_baseline.csv', index=False)
print(f"✓ 保存诊断特征: {result_dir}/diagnosis_feature_bilstm_baseline.xlsx/csv")

# 3. 保存PCA分析主要结果
np.save(f'{result_dir}/v_I_bilstm_baseline.npy', v_I)
np.save(f'{result_dir}/v_bilstm_baseline.npy', v)
np.save(f'{result_dir}/v_ratio_bilstm_baseline.npy', v_ratio)
np.save(f'{result_dir}/p_k_bilstm_baseline.npy', p_k)
np.save(f'{result_dir}/data_mean_bilstm_baseline.npy', data_mean)
np.save(f'{result_dir}/data_std_bilstm_baseline.npy', data_std)
np.save(f'{result_dir}/T_95_limit_bilstm_baseline.npy', T_95_limit)
np.save(f'{result_dir}/T_99_limit_bilstm_baseline.npy', T_99_limit)
np.save(f'{result_dir}/SPE_95_limit_bilstm_baseline.npy', SPE_95_limit)
np.save(f'{result_dir}/SPE_99_limit_bilstm_baseline.npy', SPE_99_limit)
np.save(f'{result_dir}/P_bilstm_baseline.npy', P)
np.save(f'{result_dir}/k_bilstm_baseline.npy', k)
np.save(f'{result_dir}/P_t_bilstm_baseline.npy', P_t)
np.save(f'{result_dir}/X_bilstm_baseline.npy', X)
np.save(f'{result_dir}/data_nor_bilstm_baseline.npy', data_nor)
print(f"✓ 保存PCA分析结果: {result_dir}/*_bilstm_baseline.npy")

# 4. 保存CombinedAE模型参数
torch.save(net.state_dict(), f'{result_dir}/net_model_bilstm_baseline.pth')
torch.save(netx.state_dict(), f'{result_dir}/netx_model_bilstm_baseline.pth')
print(f"✓ 保存MC-AE模型: {result_dir}/net_model_bilstm_baseline.pth, {result_dir}/netx_model_bilstm_baseline.pth")

# 5. 保存训练历史
training_history = {
    'mcae1_losses': train_losses_mcae1,
    'mcae2_losses': train_losses_mcae2,
    'final_mcae1_loss': train_losses_mcae1[-1],
    'final_mcae2_loss': train_losses_mcae2[-1],
    'mcae1_reconstruction_error_mean': np.mean(np.abs(ERRORU)),
    'mcae1_reconstruction_error_std': np.std(np.abs(ERRORU)),
    'mcae2_reconstruction_error_mean': np.mean(np.abs(ERRORX)),
    'mcae2_reconstruction_error_std': np.std(np.abs(ERRORX)),
    'training_samples': len(train_samples),
    'epochs': EPOCH,
    'learning_rate': LR,
    'batch_size': BATCHSIZE
}

import pickle
with open(f'{result_dir}/bilstm_training_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)
print(f"✓ 保存训练历史: {result_dir}/bilstm_training_history.pkl")

print("="*50)
print("🎉 BiLSTM基准训练完成！")
print("="*50)
print("BiLSTM基准模式总结：")
print("1. ✅ 跳过Transformer训练阶段")
print("2. ✅ 直接使用原始vin_2[x[0]]和vin_3[x[0]]数据")
print("3. ✅ 保持Pack Modeling输出vin_2[x[1]]和vin_3[x[1]]不变")
print("4. ✅ MC-AE使用原始BiLSTM数据进行训练")
print("5. ✅ 所有模型和结果文件添加'_bilstm_baseline'后缀")
print("")
print("📊 比对说明：")
print("   - 此模式建立BiLSTM基准性能")
print("   - 可与Transformer模式进行公平对比")
print("   - 便于评估Transformer替换的效果")
print("   - 训练时间更短，适合快速验证") 