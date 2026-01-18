# -- coding: utf-8 --
from tf_utils import *
import datetime
import pandas as pd
import numpy as np


def seq2instance(data, P, Q, low_index=0, high_index=100, granularity=15, sites=108, type='train'):
    '''
    将原始时序数据转化为样本：
    输入:
        data: 原始数据 [T * sites, num_features]
        P: 输入时间步
        Q: 输出时间步
    输出:
        X: [N, P, sites, feature_dim]
        DoW, D, H, M: 时间索引信息
        L: [N, sites, Q, feature_dim]
    '''
    X, DoW, D, H, M, L, XAll = [], [], [], [], [], [], []

    feature_start_idx = 5              # 从 flow 开始
    feature_dim = data.shape[1] - feature_start_idx  # flow + emb_0~emb_31 = 33维

    while low_index + P + Q < high_index:
        # ---- 标签（Y）部分 ----
        label = data[low_index * sites:(low_index + P + Q) * sites, feature_start_idx:]
        label = label.reshape(P + Q, sites, feature_dim)
        L.append(label[np.newaxis, ...].transpose(0, 2, 1, 3))  # [sites, Q, feature_dim]

        # ---- 输入（X）部分 ----
        x = data[low_index * sites:(low_index + P) * sites, feature_start_idx:]
        x = x.reshape(P, sites, feature_dim)
        X.append(x[np.newaxis, ...])  # [1, P, sites, feature_dim]

        # ---- 时间信息 ----
        date = data[low_index * sites: (low_index + P + Q) * sites, 1]
        DoW.append(np.reshape([datetime.date(
            int(char.replace('/', '-').split('-')[0]),
            int(char.replace('/', '-').split('-')[1]),
            int(char.replace('/', '-').split('-')[2])
        ).weekday() for char in date], [1, P + Q, sites]))
        D.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 2], [1, P + Q, sites]))
        H.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 3], [1, P + Q, sites]))
        hours_to_minutes = data[low_index * sites: (low_index + P + Q) * sites, 3] * 60
        minutes_index_of_day = np.add(hours_to_minutes,
                                      data[low_index * sites: (low_index + P + Q) * sites, 4])
        M.append(np.reshape(minutes_index_of_day // granularity, [1, P + Q, sites]))

        # ---- 所有X（用于后续特征）----
        # ---- XAll 同样改为 P+Q 时间步 ----
        XAll.append(np.reshape(
            data[low_index * sites: (low_index + P + Q) * sites, feature_start_idx:],  # 包含输入+输出
            [1, P + Q, sites, feature_dim]
        ))

        low_index += 1

    return (
        np.concatenate(X, axis=0),      # [N, P, sites, feature_dim]
        np.concatenate(DoW, axis=0),
        np.concatenate(D, axis=0),
        np.concatenate(H, axis=0),
        np.concatenate(M, axis=0),
        np.concatenate(L, axis=0),      # [N, sites, Q, feature_dim]
        np.concatenate(XAll, axis=0)
    )


def loadData(args):
    # ---- 加载数据 ----
    df = pd.read_csv(args.file_train_f)
    Traffic = df.values
    total_samples = df.shape[0] // args.site_num

    train_low = 0
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # ---- 训练集 ----
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=train_low, high_index=val_low,
        granularity=args.granularity, sites=args.site_num, type='train')
    print('Training dataset has been loaded!')

    # ---- 验证集 ----
    valX, valDoW, valD, valH, valM, valL, valXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=val_low, high_index=test_low,
        granularity=args.granularity, sites=args.site_num, type='validation')
    print('Validation dataset has been loaded!')

    # ---- 测试集 ----
    testX, testDoW, testD, testH, testM, testL, testXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=test_low, high_index=total_samples,
        granularity=args.granularity, sites=args.site_num, type='test')
    print('Testing dataset has been loaded!')

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
        trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
        valX, valDoW, valD, valH, valM, valL, valXAll,
        testX, testDoW, testD, testH, testM, testL, testXAll,
        mean, std
    )


# 调试用例（示例）
# trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, \
# valX, valDoW, valD, valH, valM, valL, valXAll, \
# testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
#
# print("trainX:", trainX.shape, "trainY:", trainL.shape)
