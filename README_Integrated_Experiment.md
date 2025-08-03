# Transformer集成训练测试系统

## 概述

这是一个集成的Transformer训练测试系统，实现了两种参数配置的对比实验，支持反向传播机制、三窗口故障检测，以及完整的可视化分析。

## 实验配置

### 训练和测试数据
- **训练集**: QAS样本0-9（10个正常样本）
- **测试集**: 
  - 正常样本: 10, 11
  - 故障样本: 335, 336

### 两种实验配置

#### 方案1: 原始参数规模
```python
config_original = {
    'd_model': 128,
    'n_heads': 8, 
    'n_layers': 6,
    'd_ff': 512,
    'dropout': 0.1,
    'save_suffix': '_original'
}
```

#### 方案2: 增强参数规模(+50%)
```python
config_enhanced = {
    'd_model': 192,      # 128 * 1.5
    'n_heads': 12,       # 8 * 1.5
    'n_layers': 6,       # 保持不变
    'd_ff': 768,         # 512 * 1.5
    'dropout': 0.2,      # 增加dropout防止过拟合
    'save_suffix': '_enhanced'
}
```

## 核心功能

### 1. 反向传播机制
- **反馈频率**: 每5个epoch执行一次
- **反馈权重**: α = 0.1
- **目标**: 通过假阳性反馈提升模型泛化能力

### 2. 三窗口故障检测
- **检测窗口**: 100个采样点 - 寻找候选故障点
- **验证窗口**: 50个采样点 - 检查持续性
- **标记窗口**: ±50个采样点 - 标记故障区域

### 3. 诊断阈值计算
按照`Test_combine_transonly.py`的方法:
```python
nm = 3000  # 固定值
if data_length > nm:
    threshold1 = mean(fai[nm:]) + 3*std(fai[nm:])
    threshold2 = mean(fai[nm:]) + 4.5*std(fai[nm:])
    threshold3 = mean(fai[nm:]) + 6*std(fai[nm:])
```

## 文件结构

```
Linux/
├── Train_Transformer_Integrated.py          # 主集成脚本
├── Train_Transformer_Integrated_main.py     # 测试和可视化函数
├── README_Integrated_Experiment.md          # 使用说明
└── modelsfl_original/                       # 原始参数实验结果
    ├── checkpoint.pkl                       # 断点续算文件
    ├── transformer_model.pth               # Transformer模型
    ├── net_model.pth                       # MC-AE模型1
    ├── netx_model.pth                      # MC-AE模型2
    ├── *.npy                               # PCA参数文件
    └── visualizations/                     # 可视化结果
└── modelsfl_enhanced/                      # 增强参数实验结果
    └── [相同的文件结构]
└── modelsfl_comparison/                    # 对比分析结果
    ├── complete_results.pkl               # 完整实验结果
    └── visualizations/
        └── experiment_comparison.png       # 对比图表
```

## 使用方法

### 运行完整实验
```bash
cd Linux/
python Train_Transformer_Integrated.py
```

### 断点续算
如果训练中断，脚本会自动检测`checkpoint.pkl`文件并从中断点继续。

### 查看结果
训练完成后，结果保存在对应的`modelsfl_*`目录中：
- 模型文件: `*.pth`
- PCA参数: `*.npy` 
- 断点文件: `checkpoint.pkl`
- 可视化图表: `visualizations/`

## 评估指标

### 主要指标
- **ROC-AUC**: 主要评估指标
- **PR-AUC**: 更适合不平衡数据
- **F1-Score**: 综合评估
- **假阳性率(FPR)**: 重点关注
- **真阳性率(TPR)**: 检测能力

### 训练指标
- **收敛速度**: 损失下降速度
- **训练稳定性**: 损失波动程度
- **计算效率**: 训练时间和GPU利用率

## 可视化输出

### 对比图表包含
1. **ROC曲线对比**: 两种配置的检测性能对比
2. **性能指标对比**: 准确率、精确率、召回率等
3. **训练损失对比**: 训练过程的损失变化
4. **样本级性能对比**: 正常/故障样本的FAI分布

### 单独图表
每个实验还会生成：
- 训练损失曲线
- MC-AE重构误差分布
- 三窗口检测过程图
- 故障检测时序图

## 系统要求

### 硬件要求
- **GPU**: 建议双GPU（A100等）
- **显存**: 至少16GB
- **内存**: 至少32GB

### 软件要求
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **CUDA**: 11.3+

### 依赖包
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm openpyxl
pip install pickle5  # 如果需要
```

## 注意事项

### Linux系统中文字体
脚本会自动检测和配置中文字体：
- 优先使用系统中文字体
- 如无中文字体，自动切换到英文标签
- 支持字体: WenQuanYi Micro Hei, Noto Sans CJK SC等

### 内存管理
- 启用混合精度训练节省显存
- 支持数据并行提升训练速度
- 自动清理GPU缓存

### 错误处理
- 支持断点续算，避免重复训练
- 详细的错误日志和异常处理
- 自动检查数据质量和模型文件

## 扩展功能

### 自定义配置
可以修改`EXPERIMENT_CONFIGS`添加更多配置：
```python
EXPERIMENT_CONFIGS = {
    'custom': {
        'name': '自定义配置',
        'd_model': 256,
        'n_heads': 16,
        # ... 其他参数
        'save_suffix': '_custom'
    }
}
```

### 添加新的评估指标
在`calculate_performance_metrics`函数中添加新指标。

### 自定义可视化
在`create_comparison_visualization`函数中添加新的图表类型。

## 常见问题

### Q: 如何修改训练样本范围？
A: 修改`TRAIN_SAMPLES`和`TEST_SAMPLES`变量。

### Q: 如何调整反向传播参数？
A: 修改`FEEDBACK_CONFIG`中的参数。

### Q: 如何关闭中文显示？
A: 设置`use_chinese = False`。

### Q: 如何增加新的窗口参数？
A: 修改`WINDOW_CONFIG`中的参数。

## 开发计划

- [ ] 完整的MC-AE训练集成
- [ ] 更多的反向传播策略
- [ ] 实时训练监控界面
- [ ] 自动超参数调优
- [ ] 分布式训练支持

## 更新日志

### v1.0.0 (2024-08-03)
- 初始版本发布
- 支持双配置对比实验
- 集成三窗口故障检测
- 支持断点续算
- Linux中文字体自适应