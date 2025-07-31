import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import torch.nn as nn
import torch
from scipy import stats
from scipy.stats import chi2
from scipy.stats import norm

from create_dataset import series_to_supervised
# 中文注释：计算电压模态和电压偏差
def calculate_volt_modepi(volt_all):
    """
    Simplified function to calculate volt_modepi and volt_di
    """
    # 中文注释：计算电压模态
    volt_mode = volt_all.mean(axis=1)
    # 中文注释：计算电压标准差
    volt_std = volt_all.std(axis=1)
    # 中文注释：计算电压Lambda
    volt_lamda = 1 / volt_std

    # 中文注释：计算电压Pi1
    volt_pi1 = (1 / (2 * np.pi * volt_all.pow(3))).mul(volt_lamda, axis=0).pow(0.5)
    # 中文注释：计算电压Pi2
    volt_pi2 = (((-1) * ((volt_all.sub(volt_mode, axis=0))).pow(2).mul(volt_lamda, axis=0)) / (2 * volt_all.mul(volt_mode.pow(2), axis=0))).apply(np.exp)
    # 中文注释：计算电压Pi
    volt_pi = volt_pi1 * volt_pi2

    # 中文注释：计算电压模态Pi
    volt_modepi = ((volt_pi * volt_all).sum(axis=1)) / (volt_pi.sum(axis=1))
    # 中文注释：计算电压偏差
    volt_di = volt_all.sub(volt_modepi, axis=0)

    return volt_modepi, volt_di


# 中文注释：求解器
def solvers(volt_modepi, volt_di, volt_all, soc, b, current, temp_avg):
    # 中文注释：获取电压数据维度
    num = volt_all.shape[1]
    # 中文注释：初始化状态矩阵P
    P = np.eye(2)
    # 中文注释：初始化状态矩阵Pi
    Pi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化Pi的第一个元素
    Pi[0, 0:num] = 1000 * np.ones((1, num))
    # 中文注释：初始化过程噪声矩阵Q
    Q = np.array([[1e-5, 0], [0, 1e-5]])
    # 中文注释：初始化测量噪声矩阵R
    R = 0.1
    # 中文注释：初始化过程噪声协方差Qi
    Qi = 1e-4
    # 中文注释：初始化测量噪声协方差Ri
    Ri = 2

    # 中文注释：初始化系统噪声协方差RO
    RO = 1e-3
    # 中文注释：初始化系统噪声协方差RP
    RP = 5.2203e-4
    # 中文注释：初始化系统噪声协方差CP
    CP = 5e3
    # 中文注释：初始化DRi
    DRi = 3 * 10 ** (-6) * np.ones(volt_all.shape[1])
    # 中文注释：计算时间常数t
    t = RP * CP
    # 中文注释：初始化SOC
    SOC = np.zeros(volt_all.shape[0])
    # 中文注释：初始化最小SOC
    smin = np.zeros(volt_all.shape[0])
    # 中文注释：初始化RH
    RH = np.zeros(volt_all.shape[0])

    # 中文注释：初始化SOC的第一个元素
    SOC[0] = 0.01 * soc[0]
    # 中文注释：初始化smin的第一个元素
    smin[0] = 0.01 * soc[0] - 0.001
    # 中文注释：初始化RH的第一个元素
    RH[0] = 0.01 * soc[0]

    # 中文注释：初始化状态向量x
    x = np.zeros((2, volt_all.shape[0]))
    # 中文注释：初始化前一个状态向量xpre
    xpre = np.zeros((2, volt_all.shape[0] + 1))
    # 中文注释：初始化x的第一个元素
    x[:, 0] = np.array([0, soc[0]]).T
    # 中文注释：初始化状态矩阵xi
    xi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化状态矩阵Xi
    Xi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化Xi的第一个元素
    Xi[0, 0:num] = SOC[0] * np.ones((1, num))
    # 中文注释：初始化状态矩阵xiphi
    xiphi = np.zeros((volt_all.shape[0] + 1, num))

    # 中文注释：初始化状态转移矩阵A
    A = np.array([[np.exp(-1 / t), 0], [0, 1]])
    # 中文注释：初始化状态转移矩阵B
    B = np.array([(1 - np.exp(-1 / t)) * RP, -1 / (23 * 3600)])
    # 中文注释：初始化SOCi
    SOCi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化SOCi的第一个元素
    SOCi = SOC[0] * np.ones((1, num))
    # 中文注释：初始化电流I
    I = current
    # 中文注释：初始化温度temp
    temp = temp_avg
    # 中文注释：初始化OCV
    OCV = [0] * volt_all.shape[0]
    # 中文注释：初始化OCVi
    OCVi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化DUi
    DUi = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化Ui
    Ui = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化前一个Ui
    Uipre = np.zeros((volt_all.shape[0] + 1, num))
    # 中文注释：初始化ei
    ei = np.zeros((volt_all.shape[0], num))
    # 中文注释：初始化DUt
    DUt = np.array(volt_di)
    # 中文注释：初始化C
    C = np.zeros((volt_all.shape[0], 2))
    # 中文注释：初始化U
    U = [0] * volt_all.shape[0]
    # 中文注释：初始化前一个U
    Upre = [0] * (volt_all.shape[0] + 1)
    # 中文注释：初始化Spre
    Spre = [0] * (volt_all.shape[0] + 1)
    # 中文注释：初始化Ut
    Ut = volt_modepi
    # 中文注释：初始化V
    V = [0] * volt_all.shape[0]
    # 中文注释：初始化e
    e = [0] * volt_all.shape[0]

    # 中文注释：循环计算每个时间步
    for k in range(1, volt_all.shape[0] - 1):
        # 中文注释：计算当前状态x
        x[:, k] = np.dot(A, x[:, k - 1]) + B * I.iloc[k - 1]
        # 中文注释：温度限制
        if x[1, k] > 1:
            x[1, k] = 1
        # 中文注释：计算OCV
        OCV[k] = b[17] * x[1, k] ** 17 + b[16] * x[1, k] ** 16 + b[15] * x[1, k] ** 15 + b[14] * x[1, k] ** 14 + b[13] * \
                 x[1, k] ** 13 + \
                 b[12] * x[1, k] ** 12 + b[11] * x[1, k] ** 11 + b[10] * x[1, k] ** 10 + b[9] * x[1, k] ** 9 + b[8] * x[
                     1, k] ** 8 + \
                 b[7] * x[1, k] ** 7 + b[6] * x[1, k] ** 6 + b[5] * x[1, k] ** 5 + b[4] * x[1, k] ** 4 + b[3] * x[
                     1, k] ** 3 + \
                 b[2] * x[1, k] ** 2 + b[1] * x[1, k] + b[0] + b[18] * temp.iloc[k] + b[19] * temp.iloc[k] ** 2 + b[
                     20] * temp.iloc[k] ** 3
        # 中文注释：计算C
        C[k, :] = [-1, b[17] * x[1, k] ** 16 * 17 + b[16] * x[1, k] ** 15 * 16 + b[15] * x[1, k] ** 14 * 15 + b[14] * x[
            1, k] ** 13 * 14 +
                   b[13] * x[1, k] ** 12 * 13 + b[12] * x[1, k] ** 11 * 12 + b[11] * x[1, k] ** 10 * 11 + b[10] * x[
                       1, k] ** 9 * 10 +
                   b[9] * x[1, k] ** 8 * 9 + b[8] * x[1, k] ** 7 * 8 + b[7] * 7 * x[1, k] ** 6 + b[6] * 6 * x[
                       1, k] ** 5 +
                   b[5] * 5 * x[1, k] ** 4 + b[4] * 4 * x[1, k] ** 3 + 3 * b[3] * x[1, k] ** 2 + 2 * b[2] * x[1, k] + b[
                       1]]
        # 中文注释：计算U
        U[k] = OCV[k] - x[0, k] - RO * I.iloc[k]
        # 中文注释：计算e
        e[k] = Ut.iloc[k] - U[k]
        # 中文注释：初始化参数
        rou = 1.2
        beta = 0.15
        gamma = 1.5
        # 中文注释：计算V
        if k == 2:
            V[k] = e[k] * e[k].T
        else:
            V[k] = (rou * V[k - 1] + e[k] * e[k].T) / (1 + rou)
        # 中文注释：计算N
        N = V[k] - beta * R - np.dot(np.dot(C[k, :], Q), (np.dot(C[k, :], Q)).T)
        # 中文注释：计算M
        M = np.dot(np.dot(C[k, :], A), np.dot(np.dot(P, A.T), np.dot(C[k, :], A).T))
        # 中文注释：计算Ek
        Ek = N / M
        # 中文注释：计算lambda_k
        if gamma * Ek > 1 and gamma * Ek < 1.5:
            lambda_k = gamma * Ek
        elif gamma * Ek >= 1.5:
            lambda_k = 1.5
        else:
            lambda_k = 1
        # 中文注释：更新P
        P = lambda_k * (A.dot(P)).dot(A.T) + Q
        # 中文注释：计算K
        K = np.dot(P, C[k, :].T) / (np.dot(C[k, :], np.dot(P, C[k, :].T)) + R)
        # 中文注释：更新x
        x[:, k] = x[:, k] + np.dot(K, e[k])
        # 中文注释：更新P
        P = P - np.dot(np.dot(K, C[k, :]), P)
        # 中文注释：更新xpre
        xpre[:, k + 1] = np.dot(A, x[:, k]) + B * I.iloc[k]
        # 中文注释：更新Spre
        Spre[k + 1] = soc[k] + B[1] * I.iloc[k]
        # 中文注释：更新Upre
        Upre[k + 1] = OCV[k] - xpre[0, k + 1] - RO * I.iloc[k + 1]

        # 中文注释：循环计算每个SOC状态
        for j in range(volt_all.shape[1]):
            # 中文注释：更新xi
            xi[k, j] = xi[k - 1, j]
            # 中文注释：更新Pi
            Pi[k, j] = Pi[k - 1, j] + Qi
            # 中文注释：更新OCVi
            OCVi[k, j] = b[17] * (xi[k, j] + x[1, k]) ** 17 + b[16] * (xi[k, j] + x[1, k]) ** 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 15 + b[14] * (xi[k, j] + x[1, k]) ** 14 + b[13] * (
                                     xi[k, j] + x[1, k]) ** 13 + b[12] * (xi[k, j] + x[1, k]) ** 12 + b[11] * (
                                     xi[k, j] + x[1, k]) ** 11 + b[10] * (xi[k, j] + x[1, k]) ** 10 + b[9] * (
                                     xi[k, j] + x[1, k]) ** 9 + b[8] * (xi[k, j] + x[1, k]) ** 8 + b[7] * (
                                     xi[k, j] + x[1, k]) ** 7 + b[6] * (xi[k, j] + x[1, k]) ** 6 + b[5] * (
                                     xi[k, j] + x[1, k]) ** 5 + b[4] * (xi[k, j] + x[1, k]) ** 4 + b[3] * (
                                     xi[k, j] + x[1, k]) ** 3 + b[2] * (xi[k, j] + x[1, k]) ** 2 + b[1] * (
                                     xi[k, j] + x[1, k]) + b[0] + b[18] * temp[k] + b[19] * temp[k] ** 2 + b[20] * temp[
                             k] ** 3;
            # 中文注释：OCVi限制
            if OCVi[k, j] > OCV[k] + 0.1:
                OCVi[k, j] = OCV[k] + 0.1
            # 中文注释：计算Ci
            Ci = b[17] * (xi[k, j] + x[1, k]) ** 16 * 17 + b[16] * (xi[k, j] + x[1, k]) ** 15 * 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 14 * 15 + b[14] * (xi[k, j] + x[1, k]) ** 13 * 14 + b[13] * (
                             xi[k, j] + x[1, k]) ** 12 * 13 + b[12] * (xi[k, j] + x[1, k]) ** 11 * 12 + b[11] * (
                             xi[k, j] + x[1, k]) ** 10 * 11 + b[10] * (xi[k, j] + x[1, k]) ** 9 * 10 + b[9] * (
                             xi[k, j] + x[1, k]) ** 8 * 9 + b[8] * (xi[k, j] + x[1, k]) ** 7 * 8 + b[7] * 7 * (
                             xi[k, j] + x[1, k]) ** 6 + b[6] * 6 * (xi[k, j] + x[1, k]) ** 5 + b[5] * 5 * (
                             xi[k, j] + x[1, k]) ** 4 + b[4] * 4 * (xi[k, j] + x[1, k]) ** 3 + 3 * b[3] * (
                             xi[k, j] + x[1, k]) ** 2 + 2 * b[2] * (xi[k, j] + x[1, k]) + b[1]
            # 中文注释：计算DUi
            DUi[k, j] = OCVi[k, j] - OCV[k] - I[k] * DRi[j]
            # 中文注释：计算Ui
            Ui[k, j] = U[k] + DUi[k, j]
            # 中文注释：更新Uipre
            Uipre[k + 1, j] = Upre[k + 1] + DUi[k, j]
            # 中文注释：计算ei
            ei[k, j] = DUt[k, j] - DUi[k, j]
            # 中文注释：计算Ki
            Ki = (Pi[k, j] * Ci.T) / ((Ci * Pi[k, j] * Ci.T + Ri))
            # 中文注释：更新xi
            xi[k, j] = xi[k, j] + Ki * ei[k, j]
            # 中文注释：更新Xi
            Xi[k, j] = soc[k].T + xi[k, j]
            # 中文注释：更新xiphi
            xiphi[k + 1, j] = xpre[1, k + 1].T + xi[k, j]
            # 中文注释：更新Pi
            Pi[k, j] = Pi[k, j] - Ki * Ci * Pi[k, j]
            # 中文注释：更新deta1
            # deta1[k,j]=1000*(Ui[k,j]-Utt[k,j])
            # 中文注释：更新deta4
            # deta4[k,j]=100*(Xi[k,j]-SOCi[k,j])
    # 中文注释：更新xi
    xipre = np.zeros((xi.shape[0], xi.shape[1]))
    xipre[1:, :] = xi[:xi.shape[0] - 1, :]
    return xipre, xpre, Spre, Upre, xiphi, x, DUi, Xi


# 中文注释：自定义激活函数
def custom_activation(x):
    return 2.5 + 1.8 * torch.sigmoid(x)


# 中文注释：主成分分析
def PCA(data, l1, l2):
    # 中文注释：数据标准化
    data_mean = np.mean(data, 0)
    data_std = np.std(data, 0)
    data_nor = (data - data_mean) / data_std
    # 中文注释：计算标准化数据的协方差矩阵
    X = np.cov(data_nor.T)
    # 中文注释：计算协方差矩阵的奇异值
    P, v, P_t = np.linalg.svd(X)  # 该函数返回三个值 u s v
    v_ratio = np.cumsum(v) / np.sum(v)
    # 中文注释：查找累积比率大于0.95的特征值索引
    k = np.where(v_ratio > 0.95)[0]
    # 中文注释：新的主成分
    p_k = P[:, :k[0]]
    v_I = np.diag(1 / v[:k[0]])
    # 中文注释：T2统计量阈值计算
    coe = k[0] * (np.shape(data)[0] - 1) * (np.shape(data)[0] + 1) / \
        ((np.shape(data)[0] - k[0]) * np.shape(data)[0])
    T_95_limit = coe * stats.f.ppf(0.95, k[0], (np.shape(data)[0] - k[0]))
    T_99_limit = coe * stats.f.ppf(l1, k[0], (np.shape(data)[0] - k[0]))
    # 中文注释：SPE统计量阈值计算
    O1 = np.sum((v[k[0]:]) ** 1)
    O2 = np.sum((v[k[0]:]) ** 2)
    O3 = np.sum((v[k[0]:]) ** 3)
    h0 = 1 - (2 * O1 * O3) / (3 * (O2 ** 2))
    c_95 = norm.ppf(0.95)
    c_99 = norm.ppf(l2)
    SPE_95_limit = O1 * ((h0 * c_95 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    SPE_99_limit = O1 * ((h0 * c_99 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    return v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor


# 中文注释：SPE统计量
def SPE(data_in, data_mean, data_std, p_k):
    test_data_nor = ((data_in - data_mean) / data_std).reshape(len(data_in), 1)
    I = np.eye(len(data_in))
    Q_count = np.dot(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor).T,
                     np.dot((I - np.dot(p_k, p_k.T)), test_data_nor))
    return Q_count
# 中文注释：控制图
def cont(X_col, data_mean, data_std, X_test, P, num_pc, lamda, T2UCL1):
    X_test = ((X_test - data_mean) / data_std)
    S = np.dot(X_test, P[:, :num_pc])
    r = []
    ee = (T2UCL1 / num_pc)
    for i in range(num_pc):
        aa = S[i] * S[i]
        a = aa / lamda[i, i]
        if a > ee:
            r = np.append(r, i)
    cont = np.zeros((len(r), X_col))
    for i in range(len(r)):
        for j in range(X_col):
            cont[i, j] = np.abs(S[i] / lamda[i, i] * P[j, i] * X_test[j])
    contT = np.zeros(X_col)
    for j in range(X_col):
        contT[j] = np.sum(cont[:, j])
    I = np.eye((np.dot(P, P.T)).shape[0], (np.dot(P, P.T)).shape[1])
    e = np.dot(X_test, (I - np.dot(P, P.T)))
    contQ = np.square(e)
    return contT, contQ
# 中文注释：比率归一化
def Ratio_cu(x):
    sum = np.sum(x)
    for i in range(x.shape[0]):
        x[i] = x[i] / sum
    return x
# 中文注释：滑动平均
def SlidingAverage_list(s, n):
    mean = []
    if len(s) > n:
        for m in range (n):
            mean.append(np.mean(s[:n]))
        # mean = s[:n].tolist()
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean

def SlidingAverage(s, n):
    mean = []
    if len(s) > n:
        for m in range (n):
            mean.append(np.mean(s[:n]))
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean
# 中文注释：诊断特征
def DiagnosisFeature(ERRORU,ERRORX):
    ERRORUm = pd.DataFrame(ERRORU).max(axis=1)
    ERRORXm = pd.DataFrame(ERRORX).max(axis=1)
    meanu = ERRORU.mean(axis=1)
    meanx = ERRORX.mean(axis=1)
    Z_U = (pd.DataFrame(ERRORU).sub(meanu, axis=0).div(ERRORU.std(axis=1), axis=0)).max(axis=1)
    Z_X = (pd.DataFrame(ERRORX).sub(meanx, axis=0).div(ERRORX.std(axis=1), axis=0)).max(axis=1)
    second_largest_ERRORU = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORU))
    max_diff_ERRORU = (ERRORUm - second_largest_ERRORU).div(ERRORU.std(axis=1), axis=0)
    second_largest_ERRORX = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORX))
    max_diff_ERRORX = (ERRORXm - second_largest_ERRORX).div(ERRORX.std(axis=1), axis=0)
    alpha = 0.2
    Z_U_smoothed = ERRORUm.ewm(alpha=alpha).mean()
    Z_X_smoothed = ERRORXm.ewm(alpha=alpha).mean()
    max_diff_ERRORX = pd.Series(SlidingAverage_list(max_diff_ERRORX , 100))
    Z_X = pd.Series(SlidingAverage_list(Z_X , 100))
    Z_X_smoothed = pd.Series(SlidingAverage_list(Z_X_smoothed , 100))
    df_data = pd.concat([max_diff_ERRORU,max_diff_ERRORX,Z_U,Z_X,Z_U_smoothed,Z_X_smoothed], axis=1)
    return df_data

# 中文注释：分类特征
def ClassifyFeature(temp_max,temp_avg,CONTN,insulation_resistance,threshold1,fai):
    reversed_fai = np.flip(fai[:f_time])
    index_array = np.where(reversed_fai < threshold1)[0]
    index = index_array[0] if index_array.size > 0 else None
    original_index = f_time - index - 1
    temp_dif = temp_max - temp_avg
    features_array = np.empty((0, CONTN.shape[1] + 5))
    for ttime in range(original_index + 1, f_time + 1, 1):
        Feature1 = CONTN[ttime, :]
        f1 = max(fai)
        max_erroru_column = np.argmax(ERRORU[ttime, :])
        max_count_U = np.sum(np.argmax(ERRORU[ttime - 3000:ttime, :], axis=1) == max_erroru_column)
        f2 = max_count_U / 3000
        f3 = temp_dif[f_time - 50:f_time].max()
        f4 = insulation_resistance.iloc[ttime-1000:ttime].min()
        f5 = volt_all.iloc[ttime-100:ttime,:].min().min()
        new_features = np.concatenate((Feature1, np.array([f1, f2, f3, f4, f5])))  
        features_array = np.vstack((features_array, new_features))  
    vin_feature = pd.DataFrame(features_array)
    return vin_feature

# 中文注释：准备训练数据
def prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device):
    test_X_df = pd.DataFrame(test_X.cpu().detach().numpy()[:,0,:])
    reframed = series_to_supervised(test_X_df, 1, 1)
    if isinstance(reframed, np.ndarray):
        reframed = pd.DataFrame(reframed)
    reframed.drop(reframed.columns[INPUT_SIZE:INPUT_SIZE * 2 - 2], axis=1, inplace=True)
    train = reframed.values
    train_X, train_y = train[:, :-2], train[:, -2:]
    train_y = train_y.reshape(-1, 2)
    batch_train = int(reframed.shape[0] / TIME_STEP)

    train_X = torch.tensor(train_X)
    train_X = train_X.reshape(batch_train, TIME_STEP, INPUT_SIZE).to(device)
    train_y = torch.tensor(train_y)
    train_y = train_y.reshape(batch_train, TIME_STEP, 2).to(device)
    
    return train_X, train_y
