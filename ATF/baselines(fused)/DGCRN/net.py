import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time
from layer import *
import sys
from collections import OrderedDict


class DGCRN(nn.Module):
    def __init__(self,
                 gcn_depth,
                 num_nodes,
                 device,
                 predefined_A=None,
                 dropout=0.3,
                 subgraph_size=20,
                 node_dim=40,
                 middle_dim=2,
                 seq_length=12,
                 in_dim=64,
                 out_dim=12,
                 layers=3,
                 list_weight=[0.05, 0.95, 0.95],
                 tanhalpha=3,
                 cl_decay_steps=4000,
                 rnn_size=64,
                 hyperGNN_dim=16):
        super(DGCRN, self).__init__()
        self.output_dim = 1
        self.fc_flow = nn.Linear(1, 16)  # 流量
        self.fc_time = nn.Linear(1, 16)  # 时间属性
        self.fc_flow_dec = nn.Linear(1, 32)
        self.fc_time_dec = nn.Linear(1, 32)
        #self.fc_attr_dec = nn.Linear(32, 64)

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.seq_length = seq_length

        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(self.num_nodes).to(device)

        self.rnn_size = rnn_size
        self.in_dim = in_dim

        hidden_size = self.rnn_size
        self.hidden_size = self.rnn_size

        dims_hyper = [
            self.hidden_size + in_dim, hyperGNN_dim, middle_dim, node_dim
        ]

        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                           'hyper')

        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                           'hyper')

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                              'hyper')

        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                              'hyper')

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                             'hyper')

        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                             'hyper')

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                                'hyper')

        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight,
                                'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        #self.alpha = nn.Parameter(torch.tensor(float(tanhalpha)))
        self.device = device
        self.k = subgraph_size
        dims = [in_dim + self.hidden_size, self.hidden_size]

        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth
        #self.mu = nn.Parameter(torch.tensor(0.5))
        #self.beta = nn.Parameter(torch.tensor(0.5))
        self.alpha =nn.Parameter(torch.tensor(0.5))

    def preprocessing(self, adj, predefined_A):
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def step(self,
             input,
             Hidden_State,
             Cell_State,
             predefined_A,
             type='encoder',
             idx=None,
             i=None):

        x = input

        x = x.transpose(1, 2).contiguous()

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.num_nodes, self.hidden_size)), 2)

        if type == 'encoder':

            filter1 = self.GCN1_tg(hyper_input,
                                   predefined_A[0]) + self.GCN1_tg_1(
                                       hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg(hyper_input,
                                   predefined_A[0]) + self.GCN2_tg_1(
                                       hyper_input, predefined_A[1])

        if type == 'decoder':

            filter1 = self.GCN1_tg_de(hyper_input,
                                      predefined_A[0]) + self.GCN1_tg_de_1(
                                          hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg_de(hyper_input,
                                      predefined_A[0]) + self.GCN2_tg_de_1(
                                          hyper_input, predefined_A[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_A[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_A[1])

        Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size)

        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = F.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = F.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        elif type == 'decoder':
            z = F.sigmoid(
                self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = F.sigmoid(
                self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(
                self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(
            1 - z, Cell_State)

        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(
            -1, self.hidden_size)

    def forward(self,
                input,
                idx=None,
                ycl=None,
                batches_seen=None,
                task_level=12):

        #print("input", input.shape)
        # input: (B, T, N, 34)
        flow = input[:, 0:1, :, :]  # (B, T, N, 1)
        middle = input[:, 1:33, :, :]  # (B, T, N, 32)
        timef = input[:, 33:34, :, :]  # (B, T, N, 1)
        flow_t = flow.permute(0, 2, 3, 1)
        time_t = timef.permute(0, 2, 3, 1)

        # 映射 → (B,N,T,16)
        flow_16 = self.fc_flow(flow_t)
        time_16 = self.fc_time(time_t)

        # 再变回 (B,16,N,T)
        flow_16 = flow_16.permute(0, 3, 1, 2)
        time_16 = time_16.permute(0, 3, 1, 2)


        # middle 已经是 (B,32,N,T)

        # 拼接成 (B,64,N,T)
        input = torch.cat([flow_16, middle, time_16], dim=1)



        #print("input", input.shape)
        #print("ycl", ycl.shape)

        predefined_A = self.predefined_A
        x = input

        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(batch_size * self.num_nodes,
                                                   self.hidden_size)

        outputs = None
        for i in range(self.seq_length):
            Hidden_State, Cell_State = self.step(x[..., i],
                                                 Hidden_State, Cell_State,
                                                 predefined_A, 'encoder', idx,
                                                 i)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        #print("outputs", outputs.shape)

        go_symbol = torch.zeros((batch_size, self.output_dim, self.num_nodes),
                                device=self.device)
        #print("go_symbol", go_symbol.shape)
        timeofday = ycl[:, 33:34, :, :]  # 第 33 维：时间特征
        future_attr = ycl[:, 1:33, :, :]  # 1-32 维
        future_flow = ycl[:, 0:1, :, :]  # 第 0 维：未来流量

        decoder_input = go_symbol  # 初始 (B,1,N)

        outputs_final = []

        for i in range(task_level):

            # ===============================
            # 1. 取未来流量与未来时间
            # ===============================
            flow_t = decoder_input  # (B,1,N)
            time_t = timeofday[..., i]  # (B,1,N)

            # ===============================
            # 2. 提取未来 1-32维属性 (B,32,N)
            # ===============================
            attr = future_attr[..., i]  # (B,32,N)
            #print("attr", attr.shape)

            # ===============================
            # 3. reshape → (B,N,1)
            # ===============================
            flow_t = flow_t.permute(0, 2, 1)  # (B,N,1)
            time_t = time_t.permute(0, 2, 1)  # (B,N,1)
            attr = attr.permute(0, 2, 1)
            flow_t = self.fc_flow_dec(flow_t)
            time_t = self.fc_time_dec(time_t)
            #flow_t = self.beta*time_t + (1-self.beta)*flow_t
            attr = self.alpha*time_t + (1-self.alpha)*attr
            decoder_input = torch.cat([flow_t, attr], dim=-1)
            # 变回 (B,64,N)
            decoder_input = decoder_input.permute(0, 2, 1)

            # ===============================
            # 7. 送入 RNN step
            # ===============================
            Hidden_State, Cell_State = self.step(
                decoder_input, Hidden_State, Cell_State, predefined_A,
                'decoder', idx, None
            )

            decoder_output = self.fc_final(Hidden_State)

            # 下一步使用预测流量作为输入
            decoder_input = decoder_output.view(batch_size, self.num_nodes, self.output_dim).transpose(1, 2)

            outputs_final.append(decoder_output)

            # curriculum learning
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = ycl[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)
        #print("outputs_final.shape",outputs_final.shape)

        outputs_final = outputs_final.view(batch_size, self.num_nodes,
                                           task_level,
                                           self.output_dim).transpose(1, 2)
        #print("outputs_final.shape",outputs_final.shape)

        return outputs_final

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))

            nn.init.orthogonal(Hidden_State)
            nn.init.orthogonal(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
