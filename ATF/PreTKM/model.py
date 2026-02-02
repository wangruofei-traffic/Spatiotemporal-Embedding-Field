import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSpatialAttention(nn.Module):
    """公式 (2)-(5)：多头空间注意力（节点到节点）"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, N, D)
        x_tail = x[:,-12:,:,:]
        B, T, N, D = x_tail.shape
        # (B, T, N, D) -> (B*T, N, D)
        x_flat = x_tail.reshape(B*T, N, D)

        Q = self.W_q(x_flat).view(B*T, N, self.num_heads, self.d_k).transpose(1,2)  # (B*T, H, N, d_k)
        K = self.W_k(x_flat).view(B*T, N, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x_flat).view(B*T, N, self.num_heads, self.d_k).transpose(1,2)

        # (B*T, H, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                                 # (B*T, H, N, d_k)
        out = out.transpose(1,2).contiguous().view(B*T, N, D)                 # (B*T, N, D)
        out = self.W_o(out)
        out = out.view(B, T, N, D)
        return out  # residual + LN

class MultiHeadDynamicGCN(nn.Module):
    """
    动态多头GCN：
    - 静态邻接矩阵 A_static 决定图结构
    - 动态邻接 A_dynamic 由 z 的 QK 生成 (B,T,N,N)
    - 最终邻接矩阵 = A_dynamic * A_static
    - 用最终邻接矩阵对 x 做图卷积
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, predefined_A=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_q = 96 // num_heads
        assert d_model % num_heads == 0

        # === 用于生成 Q 和 K 的线性层（作用在 z 上） ===
        self.W_q = nn.Linear(96, 96)
        self.W_k = nn.Linear(96, 96)

        # === 对 x 做特征变换（用于 GCN 的输入） ===
        self.W_v = nn.Linear(d_model, d_model)

        # === 输出映射 ===
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # === 静态邻接矩阵 mask ===
        self.register_buffer("A_static", (predefined_A > 0).float())  # (N,N)

    def forward(self, x, z):
        """
        x: (B,T,N,D)   -> 用于图卷积
        z: (B,T,N,D)   -> 用于生成动态邻接矩阵
        """
        x_tail = x[:, -12:, :, :]
        B,T,N,D = x_tail.shape

        # ========= 1. 生成 Q,K =========
        Q = self.W_q(z).view(B,T,N,self.num_heads,self.d_q)    # (B,T,N,H,d_k)
        K = self.W_k(z).view(B,T,N,self.num_heads,self.d_q)

        Q = Q.permute(0,1,3,2,4)   # (B,T,H,N,d_k)
        K = K.permute(0,1,3,2,4)

        # ========= 2. QK^T 得到动态邻接 =========
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_q)  # (B,T,H,N,N)
        A_dyn = torch.softmax(scores, dim=-1)

        # ========= 3. 静态拓扑 mask =========
        # A_static: (N,N) → (1,1,1,N,N)
        # A_static: (N,N) → (1,1,1,N,N)
        A_mask = self.A_static.clone().view(1, 1, 1, N, N)  # 先clone避免修改原来的
        # 将对角线设为1
        eye = torch.eye(N, device=A_mask.device).view(1, 1, 1, N, N)
        # 原来为0的位置加上1，其余保持不变
        A_mask = A_mask + eye * (1 - A_mask)

        A_dyn = A_dyn * A_mask              # 无连接的置 0
        #M = A_dyn.sum(dim=-1, keepdim=True)
        #A_dyn = A_dyn / (M + 1e-6)

        # ========= 4. 对 x 做图卷积 =========
        V = self.W_v(x_tail).view(B,T,N,self.num_heads,self.d_k)
        V = V.permute(0,1,3,2,4)  # (B,T,H,N,d_k)

        # A_dyn: (B,T,H,N,N)
        out = torch.matmul(A_dyn, V)  # (B,T,H,N,d_k)

        # reshape 回 (B,T,N,D)
        out = out.permute(0,1,3,2,4).reshape(B,T,N,D)

        return self.W_o(out)

'''
class MultiHeadGCN(nn.Module):
    """多头 GCN：静态图 + 可学习图（标准多头：d_k=d_model/heads）"""
    def __init__(self, d_model, num_heads=4, dropout=0.1, predefined_A=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0

        self.W = nn.Linear(d_model, d_model)

        # === 输出层：把 heads*d_k = d_model 映射回 d_model ===
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.LayerNorm(d_model)

        # === 1. 静态邻接矩阵 ===
        self.A_static = predefined_A               # shape: (N, N)
        #self.register_buffer('A_mask', (predefined_A > 0).float())

        # === 2. 可学习邻接矩阵 ===
        #self.A_learnable = nn.Parameter(
        #    (predefined_A > 0).float()             # 仅在连接处可学习
        #)

    def forward(self, x):
        x_tail = x[:, -12:, :, :]
        B,T,N,D = x_tail.shape

        # --- 得到多头输入 ---
        h = self.W(x_tail)                     # (B,T,N,64)
        h = h.view(B,T,N,self.num_heads,self.d_k)  # (B,T,N,4,16)

        #A_train = torch.sigmoid(self.A_learnable) * self.A_mask

        static_out = []
        #train_out = []

        for i in range(self.num_heads):
            h_i = h[:,:, :, i, :]     # (B,T,N,16)

            # --- 静态图 ---
            s = torch.matmul(h_i.permute(0,1,3,2), self.A_static)
            s = s.permute(0,1,3,2)
            static_out.append(s)

            # --- 可学习图 ---
            #l = torch.matmul(h_i.permute(0,1,3,2), A_train)
            #l = l.permute(0,1,3,2)
            #train_out.append(l)

        out_s = torch.cat(static_out, dim=-1)  # (B,T,N,64)
        #out_l = torch.cat(train_out, dim=-1)   # (B,T,N,64)

        out = self.W_o(out_s)

        return out
'''

class GRUTemporalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, N, D)
        我们只取最后 12 步
        """
        x_tail = x[:, -12:, :, :]        # (B,12,N,D)
        B, T, N, D = x_tail.shape

        # GRU 输入必须是 (B*N, T, D)
        x_flat = x_tail.reshape(B*N, T, D)

        h, _ = self.gru(x_flat)          # (B*N,12,D)

        # reshape 回原形状
        h = h.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B,12,N,D)

        return h



class MultiHeadTemporalAttention(nn.Module):
    """公式：多头时间注意力（时间步到时间步，同一节点）"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        H = self.num_heads
        d_k = self.d_k

        # 线性映射
        Q = self.W_q(x).view(B, T, N, H, d_k).permute(0, 2, 3, 1, 4)  # (B, N, H, T, d_k)
        K = self.W_k(x).view(B, T, N, H, d_k).permute(0, 2, 3, 1, 4)
        V = self.W_v(x).view(B, T, N, H, d_k).permute(0, 2, 3, 1, 4)

        Q_tail = Q[:, :, :, -12:, :]

        # reshape -> 合并 B*N*H
        Q_tail = Q_tail.reshape(B * N * H, 12, d_k)
        K = K.reshape(B * N * H, T, d_k)
        V = V.reshape(B * N * H, T, d_k)

        # 注意力计算
        scores = torch.matmul(Q_tail, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out_tail = torch.matmul(attn, V)  # (B*N*H, 12, d_k)
        out_tail = out_tail.view(B, N, H, 12, d_k).permute(0, 3, 1, 2, 4)  # (B,12,N,H,d_k)
        out_tail = out_tail.reshape(B, 12, N, D)
        out_tail = self.W_o(out_tail)
        out_tail = self.dropout(out_tail)

        # 残差连接，只用原始输入的最后12步
        return out_tail


class STBlock(nn.Module):
    """一个完整的 ST-Block：空间GCN + 空间Attention + 时间Attention + 两级融合门"""
    def __init__(self, d_model, num_heads_spatial=4, num_heads_temporal=4, num_gcn_heads=4, dropout=0.1, predefined_A=None):
        super().__init__()
        self.spatial_attn = MultiHeadSpatialAttention(d_model, num_heads_spatial, dropout)
        self.gcn = MultiHeadDynamicGCN(d_model, num_gcn_heads, dropout, predefined_A)
        self.temporal_attn = MultiHeadTemporalAttention(d_model, num_heads_temporal, dropout)

        self.gru_encoder = GRUTemporalEncoder(d_model)
        self.temporal_fusion_gate = nn.Sigmoid()
        #self.norm_t = nn.LayerNorm(d_model)

        #self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,z):
        # x: (B, T, N, D)
        # 1. 静态空间（GCN）
        h_ss = self.gcn(x,z)                                            # Hss
        #h_ss = self.gcn(h_ss, z)
        #h_ss = self.gcn(h_ss, z)

        # 2. 动态空间（空间注意力）
        h_ds = self.spatial_attn(x)                                   # Hds

        # 3. 空间融合门
        gate_spatial = torch.sigmoid(h_ds * h_ss)
        h_spatial = gate_spatial * h_ds + (1 - gate_spatial) * h_ss   # HFS
        #h_spatial = self.norm1(h_spatial + x[:,-12:,:,:])

        # 4. 时间注意力
        h_temporal_attn = self.temporal_attn(x)  # (B,12,N,D)
        h_gru = self.gru_encoder(x)  # (B,12,N,D)

        # ---- 时间融合门（你原本只有 Attention）----
        gate_t = self.temporal_fusion_gate(h_gru * h_temporal_attn)
        h_time = gate_t * h_gru + (1 - gate_t) * h_temporal_attn
        #h_time = self.norm_t(h_time + x[:, -12:, :, :])

        # ---- 时空融合门 ----
        gate_st = torch.sigmoid(h_spatial * h_time)
        h_st = gate_st * h_spatial + (1 - gate_st) * h_time

        return self.norm2(h_st + x[:, -12:, :, :])  # 最终残差


class BridgeTransformer(nn.Module):
    """
    真正的生成式推理模块（2024–2025 SOTA 标准版）
    未来 Query vs 全序列 Key/Value
    """
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)   # Query: 只来自未来
        self.W_k = nn.Linear(d_model, d_model)   # Key: 来自全部 24 步
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hist, future):
        """
        hist:   (B, 12, N, 64)   ← 编码器输出
        future: (B, 12, N, 64)   ← 解码器输出（未来辅助特征的时空表征）
        """
        # 1. 拼接成全序列
        all_seq = torch.cat([hist, future], dim=1)   # (B, 24, N, 64)

        B, T_all, N, D = all_seq.shape
        T_fut = future.shape[1]  # 12

        # 2. Query 来自未来，Key/Value 来自全部
        Q = self.W_q(future)                                      # (B, 12, N, 64)
        K = self.W_k(all_seq)                                     # (B, 24, N, 64)
        V = self.W_v(all_seq)                                     # (B, 24, N, 64)

        # 3. 多头分割
        Q = Q.view(B, T_fut, N, self.n_heads, self.d_k).permute(0, 3, 2, 1, 4)  # (B,H,N,12,d_k)
        K = K.view(B, T_all, N, self.n_heads, self.d_k).permute(0, 3, 2, 1, 4)  # (B,H,N,24,d_k)
        V = V.view(B, T_all, N, self.n_heads, self.d_k).permute(0, 3, 2, 1, 4)

        # 4. 注意力计算（未来 vs 全序列）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B,H,N,12,24)
        attn = F.softmax(scores, dim=-1)                                    # 重点关注哪些历史/未来时刻
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                          # (B,H,N,12,d_k)
        out = out.permute(0, 3, 2, 1, 4).contiguous().view(B, T_fut, N, D)   # (B,12,N,64)

        out = self.W_o(out)
        out = self.dropout(out)
        out = self.norm(out + future)  # 残差连接（维度一致，完美！）

        return out  # (B, 12, N, 64)


class PreTKM(nn.Module):
    def __init__(self,
                 num_nodes,
                 predefined_A,           # list of two normalized adj matrices [A, A.T]
                 d_model=64,
                 # 最终特征维度
                 num_st_blocks=3,
                 num_heads=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # 输入映射层（保持你原来的方式）
        self.fc_flow = nn.Linear(1, 32)
        self.fc_time = nn.Linear(1, 32)
        self.fc_time_dec = nn.Linear(1, 32)
        self.conv_expand = nn.Conv2d(32, 64, kernel_size=1)
        self.fc_time_d = nn.Linear(1, 32)

        # 编码器 ST-Blocks
        self.encoder_blocks = nn.ModuleList([
            STBlock(d_model, predefined_A=predefined_A[0]) for _ in range(num_st_blocks)
        ])

        # 解码器也用同样的 ST-Block（共享权重或不共享都可以）
        self.decoder_blocks = nn.ModuleList([
            STBlock(d_model, predefined_A=predefined_A[0]) for _ in range(num_st_blocks)
        ])

        # 生成式推理模块
        self.gen_infer = BridgeTransformer(d_model, num_heads)

        # 最终预测头
        self.pred = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, 32))


    def forward(self, x, ycl=None):
        # x:   (B, 12, N, 34)   历史
        # ycl: (B, 34, N, 12)   未来辅助特征
        #print("x",x.shape)
        #print("ycl",ycl.shape)

        # ==================== 1. 输入编码 ====================
        flow = x[:, 0:1, :, :].permute(0, 2, 3, 1)      # (B,N,12,1)
        middle = x[:, 1:33, :, :]                       # (B,32,N,12)
        timef = x[:, 33:34, :, :].permute(0, 2, 3, 1)
        timed = x[:, 34:35, :, :].permute(0, 2, 3, 1)
        flow_emb = self.fc_flow(flow).permute(0, 3, 1, 2)      # (B,16,N,12)

        time_emb = self.fc_time(timef).permute(0, 3, 1, 2)     # (B,16,N,12)
        time_emb_d = self.fc_time_d(timed).permute(0, 3, 1, 2)  # (B,16,N,12)
        flow_emb = flow_emb + time_emb + time_emb_d
        middle = middle + time_emb + time_emb_d
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(-1)  # (1,N,32,1)
        node_emb = node_emb.permute(0, 2, 1, 3)  # (1,32,N,1)
        B, _, N, T = flow_emb.shape
        node_emb = node_emb.expand(B, 32, N, T)  # (B,32,N,T)
        flow_emb = flow_emb + node_emb
        middle = middle + node_emb
        hist_input = torch.cat([flow_emb, middle], dim=1)  # (B,64,N,12)
        enc_ste = torch.cat([time_emb_d, time_emb , node_emb], dim=1)
        enc_ste = enc_ste.permute(0, 3, 2, 1)

        # (B, T, N, D)
        hist = hist_input.permute(0, 3, 2, 1)   # (B,12,N,64)
        enc = hist

        # ==================== 2. Encoder ====================
        for block in self.encoder_blocks:
            hist = block(hist,enc_ste)   # (B,12,N,64)

        #print("hist shape", hist.shape)

        # ==================== 3. Decoder 输入构造 ====================
        # 未来只用时间 + 属性（完全不看未来真实流量）
        time_future = ycl[:, 33:34, :, :].permute(0, 2, 3, 1)   # (B,N,12,1)
        time_future_d = ycl[:, 34:35, :, :].permute(0, 2, 3, 1)  # (B,N,12,1)
        attr_future = ycl[:, 1:33, :, :]                         # (B,32,N,12)

        time32 = self.fc_time(time_future)                  # (B,N,12,32)
        time_d_32 = self.fc_time_d(time_future_d)
        dec_input_raw = time32 + attr_future.permute(0,2,3,1) + time_d_32
        dec_input_raw = dec_input_raw.permute(0,3,1,2)
        dec_input_raw = dec_input_raw + node_emb
        #print("dec_input_raw", dec_input_raw.shape)
        dec_ste = torch.cat([time_d_32.permute(0,3,1,2), time32.permute(0,3,1,2), node_emb], dim=1)
        dec_ste = dec_ste.permute(0, 3, 2, 1)

        dec_input = self.conv_expand(dec_input_raw)  # (B,64,N,12)
        dec = dec_input.permute(0, 3, 2, 1)                      # (B,12,N,64)
        #print("dec", dec.shape)
        dec = torch.cat([enc, dec], dim= 1)
        for block in self.decoder_blocks:
            dec = block(dec,dec_ste)
        #print("dec", dec.shape)
        #dec = dec[:,-12:,:,:]

        # ==================== 5. 生成式推理 ====================
        final_hidden = self.gen_infer(hist, dec)   # (B,12,N,64)
        #print("final_hidden", final_hidden.shape)
        # ==================== 6. 预测 ====================
        out = self.pred(final_hidden)                             # (B,12,N,1)
        #print("out", out.shape)
        out = out.squeeze(-1).permute(0, 2, 1)


        return out,0