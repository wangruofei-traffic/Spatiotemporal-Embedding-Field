# -- coding: utf-8 --
from inits import *
import numpy as np
import datetime

def seq2instance(data, P, Q, low_index=0, high_index=100, granularity=15, sites=108, type='train'):
    """
    :param data: 原始数据矩阵
    :param P: 输入时间步长
    :param Q: 输出时间步长
    :param low_index/high_index: 数据范围
    :param granularity: 时间粒度
    :param sites: 节点数
    :param type: train / val / test
    :return: X, DoW, D, H, M, L, XAll
    """
    X, DoW, D, H, M, L, XAll = [], [], [], [], [], [], []
    total_week_len = 60 // granularity * 24 * 7

    feature_dim = data.shape[1] - 5
    while low_index + P + Q < high_index:
        # ---- X 输入 ----
        feature_slice = data[low_index * sites: (low_index + P) * sites, 5:5 + feature_dim]
        X.append(np.reshape(feature_slice, [1, P, sites, feature_dim]))

        # ---- Label 输出（包含flow + emb）----
        label = data[low_index * sites: (low_index + P + Q) * sites, 5:5 + feature_dim]
        label = np.concatenate([label[i * sites: (i + 1) * sites] for i in range(P + Q)], axis=1)
        L.append(np.reshape(label, [1, sites, P + Q, feature_dim]))

        # ---- 其他时间信息不改 ----
        date = data[low_index * sites: (low_index + P + Q) * sites, 1]
        # ---- 时间特征 ----
        DoW.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 1], [1, P + Q, sites]))  # 星期
        D.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 2], [1, P + Q, sites]))  # 天
        H.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 3], [1, P + Q, sites]))  # 小时

        hours_to_minutes = data[low_index * sites: (low_index + P + Q) * sites, 3] * 60
        minutes_index_of_day = np.add(hours_to_minutes, data[low_index * sites: (low_index + P + Q) * sites, 4])
        M.append(np.reshape(minutes_index_of_day // granularity, [1, P + Q, sites]))

        # ---- XAll 同样改为多维 ----
        XAll.append(np.reshape(
            data[(low_index - total_week_len) * sites: (low_index - total_week_len + P + Q) * sites, 5:5 + feature_dim],
            [1, P + Q, sites, feature_dim]
        ))

        low_index += 1

    return (
        np.concatenate(X, axis=0),  # (N, P, sites, 33)
        np.concatenate(DoW, axis=0),
        np.concatenate(D, axis=0),
        np.concatenate(H, axis=0),
        np.concatenate(M, axis=0),
        np.concatenate(L, axis=0),
        np.concatenate(XAll, axis=0)
    )


def loadData(args):
    df = pd.read_csv(args.file_train_s)
    Traffic = df.values
    total_samples = df.shape[0] // args.site_num

    train_low = 60 // args.granularity * 24 * 7
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # --- 数据集划分 ---
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=train_low, high_index=val_low,
        granularity=args.granularity, sites=args.site_num, type='train'
    )
    valX, valDoW, valD, valH, valM, valL, valXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=val_low, high_index=test_low,
        granularity=args.granularity, sites=args.site_num, type='validation'
    )
    testX, testDoW, testD, testH, testM, testL, testXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=test_low, high_index=total_samples,
        granularity=args.granularity, sites=args.site_num, type='test'
    )

    # --- 仅对特征维度的第0维（流量）做标准化 ---
    mean = np.mean(trainX[..., 0])
    std = np.std(trainX[..., 0])

    # 对第0维归一化，其余特征保持原值
    def normalize_flow(X, mean, std):
        X_norm = X.copy()
        X_norm[..., 0] = (X_norm[..., 0] - mean) / std
        return X_norm

    trainX, trainXAll = normalize_flow(trainX, mean, std), normalize_flow(trainXAll, mean, std)
    valX, valXAll = normalize_flow(valX, mean, std), normalize_flow(valXAll, mean, std)
    testX, testXAll = normalize_flow(testX, mean, std), normalize_flow(testXAll, mean, std)

    return (
        trainX, trainDoW, trainM, trainL, trainXAll,
        valX, valDoW, valM, valL, valXAll,
        testX, testDoW, testM, testL, testXAll,
        mean, std
    )
