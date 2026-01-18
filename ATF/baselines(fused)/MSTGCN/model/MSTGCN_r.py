# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = cheb_conv(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)

        return x_residual

# ======================
class FlowEmbeddingConcat(nn.Module):
    """将输入的 flow 第0通道映射到32维，并与原始 embedding（32维）拼接成64维"""
    def __init__(self):
        super().__init__()
        self.flow_embedding = nn.Linear(1, 32)  # 1→32维线性映射

    def forward(self, x):
        # x: [B, N, 33, T], 第0通道为 flow, 后32维为 embedding
        flow = x[:, :, 0:1, :].permute(0, 1, 3, 2)       # [B, N, T, 1]
        flow_mapped = self.flow_embedding(flow)          # [B, N, T, 32]
        flow_mapped = flow_mapped.permute(0, 1, 3, 2)    # [B, N, 32, T]
        embedding = x[:, :, 1:, :]                       # [B, N, 32, T]
        x_cat = torch.cat([flow_mapped, embedding], dim=2)  # [B, N, 64, T]
        return x_cat


# ======================
# 输出增强模块（与 ASTGCN 相同）
# ======================
class YFeatureFusion(nn.Module):
    """将 y 的后32维线性映射到64维，并与 MSTGCN 输出逐元素相加"""
    def __init__(self):
        super().__init__()
        self.y_mapping = nn.Linear(32, 64)

    def forward(self, x, y):
        """
        x: [B, N, 64, T]  MSTGCN 主体输出
        y: [B, N, 33, T]  y 特征
        """
        y_tail = y[:, :, 1:, :].permute(0, 1, 3, 2)  # [B, N, T, 32]
        y_mapped = self.y_mapping(y_tail)           # [B, N, T, 64]
        y_mapped = y_mapped.permute(0, 1, 3, 2)     # [B, N, 64, T]
        return x + y_mapped

class MSTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, mean, std):
        super().__init__()
        self.DEVICE = DEVICE
        self.mean = mean
        self.std = std

        # ===== 输入增强部分 =====
        if in_channels == 33:
            self.flow_embedding_concat = FlowEmbeddingConcat()
            first_block_in_channels = 64
        else:
            self.flow_embedding_concat = None
            first_block_in_channels = in_channels

        # ===== 主体 MSTGCN Blocks =====
        self.BlockList = nn.ModuleList([
            MSTGCN_block(first_block_in_channels, K, nb_chev_filter, nb_time_filter,
                         time_strides, cheb_polynomials)
        ])
        self.BlockList.extend([
            MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter,
                         1, cheb_polynomials)
            for _ in range(nb_block - 1)
        ])

        # ===== 输出层 =====
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.y_fusion = YFeatureFusion().to(self.DEVICE)
        self.to(DEVICE)

    def forward(self, x, y_feat=None):


        #print("y_feat", y_feat.shape)
        # x: [B, N, 33, T]
        if self.flow_embedding_concat is not None:
            x = self.flow_embedding_concat(x)  # -> [B, N, 64, T]

        for block in self.BlockList:
            x = block(x)  # -> [B, N, 64, T]
            #print("x", x.shape)



            # 如果有y_feat, 先映射后加
        if y_feat is not None:
            x = self.y_fusion(x, y_feat)
            #print("x", x.shape)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = output * self.std + self.mean
        return output


# ======================
# 构造函数
# ======================
def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, adj_mx, num_for_predict, len_input, num_of_vertices, mean, std):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = MSTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                             time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, mean, std)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model


