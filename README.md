# 锂电池故障检测 - Linux环境版本

## 📋 项目简介
基于Transformer和MC-AE的锂电池故障检测系统，针对Linux服务器环境优化。

## 🚀 快速开始

### 1. 环境安装
```bash
# 给脚本执行权限
chmod +x *.sh

# 运行安装脚本
./install_env.sh

# 检查环境配置
./check_environment.sh
```

### 2. 数据准备
确保数据文件结构如下（QAS与diagnosis目录同级）：
```
../QAS/
├── Labels.xls                    # 样本标签文件
├── 0/
│   ├── vin_1.pkl
│   ├── vin_2.pkl
│   ├── vin_3.pkl
│   └── targets.pkl (训练后生成)
├── 1/
│   ├── vin_1.pkl
│   ├── vin_2.pkl
│   ├── vin_3.pkl
│   └── targets.pkl (训练后生成)
└── ... (0-392样本)
```

**样本分配**:
- **训练样本**: 0-200 (201个样本)
- **测试正常样本**: 201-334 (134个样本)
- **测试故障样本**: 335-392 (58个样本)

### 3. 运行流程

#### 方式一：一键运行完整流程
```bash
chmod +x run_training.sh
./run_training.sh
```

#### 方式二：分步运行
```bash
# 第一步：生成训练样本的targets.pkl
python3 precompute_targets.py

# 第一步补充：生成测试样本的targets.pkl
python3 precompute_targets_test.py

# 第二步：训练BiLSTM基准模型
python3 Train_BILSTM.py

# 第三步：训练Transformer模型
python3 Train_Transformer.py

# 第四步：性能测试
python3 Test_combine.py
```

## 📁 文件说明

### 核心脚本
- `precompute_targets.py` - 预计算训练样本真实值数据
- `precompute_targets_test.py` - 预计算测试样本真实值数据
- `Train_Transformer.py` - 主训练脚本
- `data_loader_transformer.py` - 数据加载器
- `Test_combine.py` - 性能测试脚本

### 工具类
- `Function_.py` - 核心函数库
- `Class_.py` - 模型类定义
- `create_dataset.py` - 数据集工具

### 配置文件
- `requirements.txt` - 依赖包列表
- `install_env.sh` - 环境安装脚本
- `check_environment.sh` - 环境检查脚本
- `run_training.sh` - 一键训练脚本

## 🔧 环境要求

- Python 3.8+
- CUDA 12.3+ (支持A100 GPU)
- 内存: 32GB+ (推荐)
- 存储: 50GB+ (用于数据文件)
- 数据路径: `../QAS/` (与diagnosis目录同级)

## 📊 性能优化

### GPU训练
```bash
# 检查GPU状态
nvidia-smi

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0
```

### 内存优化
```bash
# 如果内存不足，可以调整batch_size
# 在Train_Transformer.py中修改BATCH_SIZE参数
```

## 🐛 常见问题

### 1. 字体显示问题
Linux环境可能缺少中文字体，已自动配置为使用系统字体。

### 2. CUDA版本不匹配
```bash
# 重新安装匹配的PyTorch版本
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 权限问题
```bash
# 给脚本执行权限
chmod +x *.sh
chmod +x *.py
```

## 📞 技术支持
如有问题，请检查：
1. Python版本是否符合要求
2. 依赖包是否正确安装
3. 数据文件路径是否正确
4. GPU驱动是否正常 