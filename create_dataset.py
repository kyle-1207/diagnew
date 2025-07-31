import numpy as np

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将多维时序数据转为监督学习格式。
    data: 2D numpy数组，shape=[时间步数, 特征数]
    n_in: 输入步数
    n_out: 输出步数
    dropnan: 是否去除无效行
    返回: 2D numpy数组，shape=[样本数, n_in*特征数 + n_out*特征数]
    """
    n_vars = data.shape[1]
    cols = []
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(np.roll(data, shift=i, axis=0))
    # 输出序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(np.roll(data, shift=-i, axis=0))
    agg = np.concatenate(cols, axis=1)
    if dropnan:
        agg = agg[n_in:-(n_out-1) if n_out > 1 else None]
    return agg 