import torch
import torch.nn as nn
from config import device, num_time_slices_total, num_time_slices


class SpatioTemporalEmbedding(nn.Module):
    """
    时空嵌入模块 (STE): 将星期、时刻和空间位置融合为 32 维向量。
    """

    def __init__(self, num_nodes, num_weekdays=7, num_time_slices=288, embed_dim=32):
        super().__init__()
        self.weekday_emb = nn.Embedding(num_weekdays + 1, embed_dim)
        self.timeslot_emb = nn.Embedding(num_time_slices, embed_dim)
        self.node_emb = nn.Embedding(num_nodes, embed_dim)

        nn.init.normal_(self.weekday_emb.weight, mean=0, std=0.1)
        nn.init.normal_(self.timeslot_emb.weight, mean=0, std=0.1)
        nn.init.normal_(self.node_emb.weight, mean=0, std=0.1)

    def forward(self, weekday_idx, timeslot_idx, node_idx):
        w = self.weekday_emb(weekday_idx.long()).unsqueeze(1)  # (P, 1, 32)
        t = self.timeslot_emb(timeslot_idx.long()).unsqueeze(1)  # (P, 1, 32)
        n = self.node_emb(node_idx.long())  # (P, N, 32)
        ste = w + t + n
        return torch.nan_to_num(ste, nan=0.0)


class DiffusionConv(nn.Module):
    """
    扩散卷积层: 捕获图结构上的空间依赖。
    """

    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.out_channels = out_channels
        self.W_forward = nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels, out_channels) * 0.1) for _ in range(K + 1)])
        self.W_backward = nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels, out_channels) * 0.1) for _ in range(K + 1)])

    def forward(self, X, A_forward, A_backward):
        out = torch.zeros(X.size(0), self.out_channels, device=X.device)
        X0 = X
        # 前向扩散
        Xk = X0
        for k in range(self.K + 1):
            if k == 0:
                out += Xk @ self.W_forward[k]
            else:
                Xk = A_forward @ Xk
                out += Xk @ self.W_forward[k]
        # 后向扩散
        Xk = X0
        for k in range(self.K + 1):
            if k == 0:
                out += Xk @ self.W_backward[k]
            else:
                Xk = A_backward @ Xk
                out += Xk @ self.W_backward[k]
        return out


class STDiffusionCell(nn.Module):
    """
    单层扩散卷积循环单元。
    """

    def __init__(self, in_channels, hidden_channels, K=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        combined = in_channels + hidden_channels
        self.dc_r = DiffusionConv(combined, hidden_channels, K)
        self.dc_u = DiffusionConv(combined, hidden_channels, K)
        self.dc_c = DiffusionConv(combined, hidden_channels, K)

    def forward(self, X, H_prev, A_forward, A_backward):
        combined = torch.cat([X, H_prev], dim=1)
        r = torch.sigmoid(self.dc_r(combined, A_forward, A_backward))
        u = torch.sigmoid(self.dc_u(combined, A_forward, A_backward))

        combined_c = torch.cat([X, r * H_prev], dim=1)
        c = torch.tanh(self.dc_c(combined_c, A_forward, A_backward))
        return u * H_prev + (1 - u) * c


class STDiffusionNet(nn.Module):
    """
    预训练主模型架构。
    """

    def __init__(self, num_nodes, embed_dim, hidden_dim, K=3, window_size=6):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # 核心：可学习的时空嵌入场 XKE
        self.init_emb = nn.Parameter(torch.randn(num_time_slices_total, num_nodes, embed_dim) * 0.1)
        self.ste = SpatioTemporalEmbedding(num_nodes, embed_dim=embed_dim)
        self.cell = STDiffusionCell(embed_dim, hidden_dim, K)
        self.reconstruct_head = nn.Linear(hidden_dim, window_size)

    def forward(self, A_forward, A_backward, time_slice_data):
        losses, rmses, maes = [], [], []

        # 严格按照原有逻辑在 Forward 中遍历全周期进行预训练计算
        for t_start in range(num_time_slices_total - self.window_size + 1):
            t_idx = torch.arange(t_start, t_start + self.window_size, device=device)
            weekday_idx = (t_idx // num_time_slices) % 7 + 1
            timeslot_idx = t_idx % num_time_slices
            node_idx = torch.arange(self.num_nodes, device=device).unsqueeze(0).expand(self.window_size, -1)

            # 融合 XKE + STE
            X_seq = self.init_emb[t_start:t_start + self.window_size] + self.ste(weekday_idx, timeslot_idx, node_idx)

            # 循环编码
            H = torch.zeros(self.num_nodes, self.hidden_dim, device=device)
            for i in range(self.window_size):
                H = self.cell(X_seq[i], H, A_forward, A_backward)

            # 流量重构
            recon = self.reconstruct_head(H)

            # 内部损失计算逻辑 (对比历史所有周)
            win_loss = win_rmse = win_mae = 0.0
            count = 0
            for i in range(self.window_size):
                t_curr = t_start + i
                key = ((t_curr // num_time_slices) % 7 + 1, t_curr % num_time_slices)
                if key not in time_slice_data: continue

                flow_true = time_slice_data[key]['flow']
                pred_i = recon[:, i].unsqueeze(0)
                diff = pred_i - flow_true
                win_loss += (diff ** 2).mean();
                win_rmse += torch.sqrt((diff ** 2).mean());
                win_mae += diff.abs().mean()
                count += 1

            if count > 0:
                losses.append(win_loss / count);
                rmses.append(win_rmse / count);
                maes.append(win_mae / count)
            else:
                zero = torch.tensor(0.0, device=device)
                losses.append(zero);
                rmses.append(zero);
                maes.append(zero)

        return recon, torch.stack(losses).mean(), torch.stack(rmses).mean(), torch.stack(maes).mean()