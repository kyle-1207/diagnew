# Transformer混合反馈策略实验系统

## 概述

本系统实现了基于混合反馈策略的Transformer训练和测试，通过数据隔离和自适应触发机制，有效减少假阳性率。

## 核心设计

### 数据配置
- **数据路径**: `/mnt/bz25t/bzhy/zhanglikang/project/QAS/`
- **训练样本**: QAS 0-7（8个正常样本）
- **反馈样本**: QAS 8-9（2个正常样本）
- **测试样本**: 正常10、11 + 故障335、336

### 三窗口故障检测策略

#### 设计原理
基于FAI（综合故障指标）的三窗口检测机制，采用固定策略确保检测的客观性和可重现性。

#### 窗口配置
```python
WINDOW_CONFIG = {
    "detection_window": 50,      # 检测窗口：50个采样点 (25分钟)
    "verification_window": 30,   # 验证窗口：30个采样点 (15分钟)
    "marking_window": 40,        # 标记窗口：40个采样点 (20分钟)
    "verification_threshold": 0.4 # 验证窗口内FAI异常比例阈值 (40%)
}
```

#### 时间尺度设计
- **检测窗口 (25分钟)**: 基于电池热时间常数，足够捕捉故障发展过程
- **验证窗口 (15分钟)**: 基于电化学反应时间，确认故障持续性
- **标记窗口 (20分钟)**: 基于故障传播时间，考虑前后影响范围
- **采样间隔**: 30秒/点（基于论文数据采集标准）

#### 三阶段检测流程

##### 阶段1：检测窗口
- **目标**: 基于FAI统计特性识别异常点
- **方法**: 扫描FAI序列，标记超过阈值的点
- **输出**: 候选故障点列表

##### 阶段2：验证窗口
- **目标**: 确认FAI异常的持续性，排除随机波动
- **方法**: 在候选点前后各15分钟内检查FAI异常比例
- **验证条件**: 40%以上的采样点超过阈值
- **输出**: 已验证的故障点

##### 阶段3：标记窗口
- **目标**: 考虑故障的前后影响范围
- **方法**: 在验证点前后各20分钟内标记故障区域
- **统计信息**: 计算标记区域的FAI特征（均值、最大值、标准差）

#### 策略优势

##### 1. 科学性
- **基于物理原理**: 窗口大小基于电池故障的时间特性
- **统一标准**: 所有样本使用相同的检测标准
- **客观评价**: 避免多特征组合的主观性

##### 2. 可靠性
- **持续性验证**: 40%阈值确保故障的持续性
- **噪声过滤**: 有效排除瞬时波动和随机噪声
- **范围完整性**: 考虑故障的前兆和后续影响

##### 3. 实用性
- **固定参数**: 便于工程部署和维护
- **可重现性**: 任何人运行都得到相同结果
- **可解释性**: 检测逻辑清晰，易于理解和调试

#### 性能统计
```python
detection_stats = {
    'total_candidates': len(candidate_points),      # 候选点总数
    'verified_candidates': len(verified_points),    # 验证点总数
    'total_fault_points': np.sum(fault_labels),     # 标记故障点总数
    'fault_ratio': np.sum(fault_labels) / len(fault_labels),  # 故障比例
    'mean_continuous_ratio': np.mean([v['continuous_ratio'] for v in verified_points]),  # 平均持续比例
    'mean_region_length': np.mean([m['length'] for m in marked_regions])  # 平均标记区域长度
}
```

### 混合反馈策略

#### 1. 反馈触发机制
```python
FEEDBACK_CONFIG = {
    'min_epochs_before_feedback': 20,    # 至少训练20个epoch
    'base_feedback_interval': 15,        # 基础反馈间隔
    'adaptive_threshold': 0.03,          # 自适应触发阈值（3%）
    'max_feedback_interval': 25,         # 最大反馈间隔
    'feedback_weight': 0.2,              # 反馈权重
    'mcae_feedback_weight': 0.8,         # MC-AE反馈权重
    'transformer_feedback_weight': 0.2   # Transformer反馈权重
}
```

#### 2. 分级触发条件
- **预警阈值**: 1%假阳性率（仅记录，不反馈）
- **标准触发**: 3%假阳性率（标准反馈强度0.3）
- **严重触发**: 5%假阳性率（强化反馈强度0.6）
- **紧急触发**: 10%假阳性率（最大反馈强度1.0）
- **固定间隔**: 每15个epoch执行一次反馈
- **兜底触发**: 最多25个epoch必须反馈一次

#### 3. 分层反馈
- **MC-AE反馈（主要）**: 权重0.8，直接优化重构误差
- **Transformer反馈（辅助）**: 权重0.2，改善特征表示

## 文件结构

```
/mnt/bz25t/bzhy/zhanglikang/project/diagnosis/Linux/
├── Train_Test_Integrated.py          # 主集成脚本（混合反馈策略）
├── Run_Experiment.py                 # 快速测试脚本
├── Function_.py                      # 现有函数库
├── Class_.py                         # 现有类库
├── Comprehensive_calculation.py      # 综合诊断计算
└── README_混合反馈策略.md            # 本说明文档

/mnt/bz25t/bzhy/zhanglikang/project/QAS/
├── 0/                               # 训练样本0
├── 1/                               # 训练样本1
├── ...
├── 10/                              # 测试正常样本10
├── 11/                              # 测试正常样本11
├── 335/                             # 测试故障样本335
├── 336/                             # 测试故障样本336
└── Labels.xls                       # 样本标签文件
```

## 使用方法

### 1. 快速测试
```bash
cd /mnt/bz25t/bzhy/zhanglikang/project/diagnosis/Linux
python Run_Experiment.py
```

### 2. 完整实验
```bash
cd /mnt/bz25t/bzhy/zhanglikang/project/diagnosis/Linux
python Train_Test_Integrated.py
```

### 3. 运行测试脚本
```bash
# 运行三窗口检测测试
cd /mnt/bz25t/bzhy/zhanglikang/project/diagnosis/Linux
python Test_combine_transonly.py
```

### 4. 查看结果
```bash
# 查看训练历史
ls modelsfl_*/

# 查看可视化结果
ls modelsfl_*/visualizations/

# 查看测试结果
ls /mnt/bz25t/bzhy/datasave/Transformer/
```

## 实验配置

### 两种参数配置对比

#### 原始参数规模
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

#### 增强参数规模(+50%)
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

### 1. 反馈指标计算
- 假阳性率统计
- 特征漂移检测
- 样本级指标记录

### 2. 智能反馈触发
- 多条件触发机制
- 自适应阈值调整
- 反馈频率控制

### 3. 分层反馈优化
- MC-AE重构误差优化
- Transformer特征表示改善
- 权重自适应调整

### 4. 训练监控
- 实时反馈统计
- 训练损失记录
- 性能指标跟踪

### 5. 三窗口检测
- FAI序列分析
- 多阶段故障检测
- 持续性验证
- 故障区域标记

## 输出结果

### 训练阶段
- 模型权重文件: `transformer_model.pth`
- 训练历史: `training_history.pkl`
- 反馈统计: 包含反馈次数、假阳性率等

### 测试阶段
- ROC曲线分析
- 故障检测结果
- 性能指标对比
- 三窗口检测结果
- FAI异常率统计

### 可视化
- 训练损失曲线
- ROC曲线对比
- 故障检测可视化
- 反馈效果分析
- 三窗口检测过程可视化
- FAI时间序列分析

## 优势特点

### 1. 数据隔离
- 训练样本与反馈样本分离
- 避免过拟合
- 更客观的反馈信号

### 2. 自适应机制
- 根据模型状态动态调整
- 避免过度反馈
- 提高训练效率

### 3. 分层优化
- MC-AE直接优化诊断指标
- Transformer改善特征表示
- 互补优化效果

### 4. 时间控制
- 智能触发减少计算开销
- 预期时间增加10-25%
- 效果与效率平衡

## 预期效果

### 性能提升
- 假阳性率降低15-30%
- ROC AUC提升5-15%
- 检测准确率改善
- 三窗口检测准确率提升
- FAI异常检测稳定性增强

### 训练稳定性
- 反馈机制稳定
- 收敛速度适中
- 泛化能力增强

## 注意事项

### 1. 数据要求
- 确保QAS样本0-9可用
- 数据格式正确
- 文件路径正确

### 2. 环境配置
- PyTorch环境
- 中文字体支持
- 足够的内存和GPU

### 3. 参数调整
- 反馈阈值可根据实际情况调整
- 权重配置可优化
- 触发频率可调
- 三窗口参数可根据数据特性调整
- 验证阈值可根据误检率优化

## 故障排除

### 常见问题
1. **数据加载失败**: 检查文件路径和格式
2. **反馈不触发**: 检查阈值设置
3. **训练不稳定**: 调整学习率和权重
4. **内存不足**: 减少batch size或模型大小
5. **三窗口检测异常**: 检查FAI计算和阈值设置
6. **误检率过高**: 调整验证阈值（当前40%）

### 调试建议
- 使用`Run_Experiment.py`快速验证
- 查看详细日志输出
- 检查中间结果保存

## 扩展方向

### 1. 更多反馈策略
- 多级反馈机制
- 动态权重调整
- 在线学习优化

### 2. 模型改进
- 注意力机制优化
- 损失函数改进
- 架构搜索

### 3. 应用扩展
- 其他数据集适配
- 实时检测系统
- 边缘设备部署 