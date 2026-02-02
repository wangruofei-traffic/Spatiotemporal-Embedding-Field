import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import metrics
from model import HSTKM

class Trainer():
    def __init__(self, args, scaler, supports, edge_indices, edges):
        self.model = HSTKM(
        num_nodes=args.num_nodes,
        predefined_A=supports,
        d_model=64,
        num_st_blocks=3,
        num_heads=4,
        edges = edges
    )
        self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1, verbose=False)
        
        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.use_spatial = args.use_spatial
        self.grad_clip = args.grad_clip

    def train(self, input, real_val, x_speed=None, y_speed=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_spatial:
            output, spatial_loss = self.model(input,real_val,x_speed , y_speed)
            real = real_val[:,:,:,0]
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)-0.3*spatial_loss
        else:
            output, _ = self.model(input, real_val,x_speed , y_speed)
            real = real_val[:,0,:,:]
            #print("output", output.shape)
            #print("real", real.shape)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
        
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        mape = metrics.masked_mape(predict,real,0.0).item()
        rmse = metrics.masked_rmse(predict,real,0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val, x_speed=None, y_speed=None, flag='overall', return_data=True):
        # 引入必要的库 (为了保证函数独立性，写在函数内部)
        import matplotlib.pyplot as plt
        import os
        import time

        if flag == 'overall':
            self.model.eval()
            output, _ = self.model(input, real_val, x_speed, y_speed)
            real = real_val[:, 0, :, :]
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = metrics.masked_mape(predict, real, 0.0).item()
            rmse = metrics.masked_rmse(predict, real, 0.0).item()

            if return_data:
                return loss.item(), mape, rmse, predict.detach().cpu().numpy(), real.detach().cpu().numpy()
            else:
                return loss.item(), mape, rmse

        elif flag == 'horizon':
            self.model.eval()
            output, _ = self.model(input, real_val, x_speed, y_speed)
            real = real_val[:, 0, :, :]
            predict = self.scaler.inverse_transform(output)
            loss = []
            mape = []
            rmse = []
            for i in range(12):
                loss.append(self.loss(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())

            # --- 修改部分：支持返回数据 ---
            if return_data:
                return loss, mape, rmse, predict.detach().cpu().numpy(), real.detach().cpu().numpy()
            else:
                return loss, mape, rmse
            # ---------------------------
