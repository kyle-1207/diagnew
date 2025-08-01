import numpy as np
from scipy.stats import chi2, f

def Comprehensive_calculation(X, data_mean, data_std, v, p_k, v_I, T_99_limit, SPE_99_limit, P, time):
    """
    综合诊断指标计算，严格按照论文Note S3公式实现
    
    参数：
        X: 原始特征数据 (n_samples, n_features)
        data_mean, data_std: 标准化参数
        v: 特征值列向量 (n_components, 1)
        p_k: 主成分载荷矩阵 (n_features, n_components)
        v_I: 主成分特征值的逆 (n_components, n_components)
        T_99_limit, SPE_99_limit: 控制限
        P: 主成分矩阵 (n_features, n_components)
        time: 样本索引或时间戳
        
    返回：
        lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std
    """
    print(f"开始综合诊断计算（按照论文Note S3标准）...")
    print(f"输入数据形状: X={X.shape if hasattr(X, 'shape') else len(X)}")
    
    # 导入原有Function_.py中的函数
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '源代码备份'))
    from Function_ import cont, SPE
    
    # 确保所有输入都是numpy数组
    X = np.array(X)
    data_mean = np.array(data_mean).flatten()
    data_std = np.array(data_std).flatten()
    v = np.array(v).flatten()
    
    # 数据验证
    n_samples, n_features = X.shape
    n_components = p_k.shape[1]
    
    print(f"数据维度: samples={n_samples}, features={n_features}, components={n_components}")
    
    # 数值稳定性处理
    data_std = np.maximum(data_std, 1e-8)
    v_selected = v[:n_components]
    v_selected = np.maximum(v_selected, 1e-8)
    
    # 1. 数据标准化和中心化 (按照论文公式)
    print("1. 数据标准化...")
    # 论文公式: Xij = (Xij - x̄j) / sj
    Xc = (X - data_mean) / data_std
    
    # 2. 主成分得分计算 (Score matrix T)
    print("2. 计算主成分得分...")
    # 论文公式: ti = Xi^T * Pi
    t_total = np.dot(Xc, p_k)
    
    # 3. 残差矩阵计算
    print("3. 计算残差...")
    # 论文公式: X̂ = T * P^T (重构)
    X_hat = np.dot(t_total, p_k.T)
    # 残差: E = X - X̂
    E = Xc - X_hat
    
    # 4. T²统计量计算 (按照论文标准公式)
    print("4. 计算T²统计量...")
    # 论文公式: T² = Xi^T * Pk * S^(-1) * Pk^T * Xi
    # 简化为: T² = Σ(ti²/λi) 其中ti是主成分得分，λi是特征值
    T2 = np.sum((t_total ** 2) / v_selected, axis=1)
    
    # 5. SPE统计量计算 (Q统计量)
    print("5. 计算SPE统计量...")
    # 论文公式: SPE = ei * ei^T = Xi(I - Pk*Pk^T)Xi^T
    SPE_values = np.sum(E ** 2, axis=1)
    q_total = SPE_values
    
    # 6. 综合故障指示器 φ(fai) 计算
    print("6. 计算综合故障指示器...")
    
    # 控制限处理 - 处理0维numpy数组
    chi2_T2 = float(T_99_limit) if hasattr(T_99_limit, 'item') else (T_99_limit if np.isscalar(T_99_limit) else T_99_limit[0])
    delta2 = float(SPE_99_limit) if hasattr(SPE_99_limit, 'item') else (SPE_99_limit if np.isscalar(SPE_99_limit) else SPE_99_limit[0])
    
    # 数值稳定性
    chi2_T2 = max(chi2_T2, 1e-8)
    delta2 = max(delta2, 1e-8)
    
    # 论文公式: φ = SPE(x)/δ² + T²(x)/χ²
    fai = SPE_values / delta2 + T2 / chi2_T2
    FAI = fai
    
    print(f"   fai范围: [{fai.min():.4f}, {fai.max():.4f}]")
    print(f"   fai均值: {fai.mean():.4f}")
    
    # 7. 故障概率计算 (按照论文Note S3)
    print("7. 计算故障概率...")
    # 论文提到φ服从卡方分布，计算P(χ²h > φ)
    # 自由度h通常等于主成分数量
    h = n_components
    fault_probability = 1 - chi2.cdf(fai, df=h)
    
    # 8. 贡献度分析 - 使用原有Function_.py中的cont函数
    print("8. 计算贡献度（使用原有cont函数）...")
    
    # 准备cont函数所需的参数
    lamda_matrix = np.diag(v_selected)  # cont函数需要对角矩阵格式
    
    # 确保lamda_matrix是二维的
    if lamda_matrix.ndim == 1:
        lamda_matrix = np.diag(lamda_matrix)
    
    # 调试信息
    print(f"   lamda_matrix形状: {lamda_matrix.shape}")
    print(f"   p_k形状: {p_k.shape}")
    print(f"   data_mean形状: {data_mean.shape}")
    print(f"   data_std形状: {data_std.shape}")
    
    # 为每个样本计算贡献度
    contTT_list = []
    contQ_list = []
    CONTN_list = []
    
    for i in range(n_samples):
        try:
            # 确保输入参数维度正确
            X_test_sample = np.array(Xc[i]).flatten()  # 确保是1维
            
            # 调用原有的cont函数
            contT_single, contQ_single = cont(
                n_features,           # X_col: 特征数量
                data_mean,           # data_mean: 均值
                data_std,            # data_std: 标准差
                X_test_sample,       # X_test: 当前样本
                p_k,                # P: 主成分矩阵
                n_components,       # num_pc: 主成分数量
                lamda_matrix,       # lamda: 特征值对角矩阵
                chi2_T2             # T2UCL1: T²控制限
            )
            
            contTT_list.append(contT_single)
            contQ_list.append(contQ_single)
            
            # 综合贡献度 (按照论文方法)
            if T2[i] + SPE_values[i] > 1e-8:
                contn_single = (contT_single + contQ_single) / (T2[i] + SPE_values[i])
            else:
                contn_single = np.zeros(n_features)
            CONTN_list.append(contn_single)
            
        except Exception as e:
            print(f"   样本{i}贡献度计算出错: {e}")
            # 使用零向量作为默认值
            contTT_list.append(np.zeros(n_features))
            contQ_list.append(np.zeros(n_features))
            CONTN_list.append(np.zeros(n_features))
    
    # 转换为numpy数组
    contTT = np.array(contTT_list) if contTT_list else np.zeros((n_samples, n_features))
    contQ = np.array(contQ_list) if contQ_list else np.zeros((n_samples, n_features))
    CONTN = np.array(CONTN_list) if CONTN_list else np.zeros((n_samples, n_features))
    
    # 9. 报警等级计算 (保留新增的三级报警机制)
    print("9. 计算报警等级...")
    level, maxlevel, threshold1, threshold2, threshold3 = calculate_alarm_levels_paper_method(fai, fault_probability)
    
    # 10. 其他统计参数 (按照论文定义)
    lamda = v_selected
    
    # 论文中的参数
    g = h  # 自由度
    kesi = 0.99  # 置信水平
    f_time = n_samples  # 时间长度
    
    # 主成分解释的方差比例
    X_ratio = np.sum(v_selected) / np.sum(v) if np.sum(v) > 0 else 0
    
    # 综合贡献度
    CContn = np.mean(CONTN, axis=0) if CONTN.size > 0 else np.zeros(n_features)
    
    # S矩阵 (协方差矩阵的对角化形式)
    S = np.diag(v_selected)
    
    print("综合诊断计算完成!")
    print(f"最大报警等级: {maxlevel}")
    print(f"主成分解释方差比例: {X_ratio:.4f}")
    print(f"平均故障概率: {np.mean(fault_probability):.6f}")
    
    return (lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai, 
            f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std)

def calculate_alarm_levels_paper_method(fai, fault_probability):
    """
    按照论文方法计算三级报警等级
    结合统计控制和概率阈值
    """
    # 使用后段数据计算基准统计量
    nm = min(3000, len(fai) // 2)
    mm = len(fai)
    
    if mm > nm:
        baseline_fai = fai[nm:mm]
        baseline_prob = fault_probability[nm:mm]
    else:
        baseline_fai = fai
        baseline_prob = fault_probability
        
    fai_mean = np.mean(baseline_fai)
    fai_std = np.std(baseline_fai)
    
    # 论文中的三级报警阈值 (基于统计控制图理论)
    threshold1 = fai_mean + 3 * fai_std    # 一级报警 (99.7%置信)
    threshold2 = fai_mean + 4.5 * fai_std  # 二级报警 (99.99%置信)
    threshold3 = fai_mean + 6 * fai_std    # 三级报警 (99.9999%置信)
    
    # 也可以基于概率阈值
    prob_threshold1 = 0.05  # 5%故障概率
    prob_threshold2 = 0.10  # 10%故障概率  
    prob_threshold3 = 0.20  # 20%故障概率
    
    # 报警等级分配 (综合统计和概率两种方法)
    level = np.zeros_like(fai, dtype=int)
    
    # 基于fai值的报警
    level[fai > threshold1] = 1
    level[fai > threshold2] = 2
    level[fai > threshold3] = 3
    
    # 基于概率的报警 (如果概率高，提升报警等级)
    level[fault_probability > prob_threshold1] = np.maximum(level[fault_probability > prob_threshold1], 1)
    level[fault_probability > prob_threshold2] = np.maximum(level[fault_probability > prob_threshold2], 2)
    level[fault_probability > prob_threshold3] = np.maximum(level[fault_probability > prob_threshold3], 3)
    
    maxlevel = np.max(level)
    
    print(f"   报警阈值: L1={threshold1:.4f}, L2={threshold2:.4f}, L3={threshold3:.4f}")
    print(f"   概率阈值: P1={prob_threshold1:.2f}, P2={prob_threshold2:.2f}, P3={prob_threshold3:.2f}")
    print(f"   报警点数: L1={np.sum(level==1)}, L2={np.sum(level==2)}, L3={np.sum(level==3)}")
    
    return level, maxlevel, threshold1, threshold2, threshold3 