import pandas as pd
import numpy as np
import torch
from config import DATA_EDGE, DATA_FLOW, num_nodes, num_time_slices, device


def load_adjacency_matrix():
    """
    加载并处理道路网络邻接矩阵。
    """
    edges = pd.read_csv(DATA_EDGE)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in edges.iterrows():
        src = int(row['src_FID'])
        dst = int(row['nbr_FID'])
        adj[src, dst] = 1
        adj[dst, src] = 1  # 保证双向性
    np.fill_diagonal(adj, 1)  # 自环

    # 转换为 Tensor
    A_forward = torch.tensor(adj, dtype=torch.float32, device=device)
    A_backward = torch.tensor(adj, dtype=torch.float32, device=device)
    return A_forward, A_backward


def load_flow_data():
    """
    加载流量数据，并将其按 (weekday, time_slot) 索引组织，以便计算周平均损失。
    """
    df = pd.read_csv(DATA_FLOW)
    # 保持原有逻辑：仅使用前 70% 的数据进行预训练
    cutoff = int(len(df) * 0.7)
    df = df.iloc[:cutoff].reset_index(drop=True)
    df = df.astype({'weekday': int, 'time_slot': int, 'week_idx': int, 'station': int, 'flow': float})

    node_list = list(range(num_nodes))
    time_slice_data = {}

    # 按照 7天 * 288个槽位 进行归类
    for weekday in range(1, 8):
        for t in range(num_time_slices):
            df_t = df[(df['weekday'] == weekday) & (df['time_slot'] == t)]
            if df_t.empty:
                continue

            flows = []
            # 聚合所有周在同一时刻的流量
            for _, group in df_t.groupby('week_idx'):
                flow_for_nodes = [group.set_index('station')['flow'].to_dict().get(n, 0.0) for n in node_list]
                flows.append(torch.tensor(flow_for_nodes, dtype=torch.float))

            if len(flows) == 0:
                continue

            # (W, N) 张量，W 是周数，N 是节点数
            flows_tensor = torch.stack(flows).to(device)
            time_slice_data[(weekday, t)] = {'flow': flows_tensor}

    return time_slice_data