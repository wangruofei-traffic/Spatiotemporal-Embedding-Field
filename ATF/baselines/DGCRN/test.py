import torch
import numpy as np
from util import load_dataset, load_adj, metric
from trainer import Trainer
from net import DGCRN
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/YINCHUAN')
parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_yinchuan.pkl')
parser.add_argument('--model_path', type=str, default='./model.save-1/exp1_0.pth')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seq_in_len', type=int, default=12)
parser.add_argument('--seq_out_len', type=int, default=12)
parser.add_argument('--num_nodes', type=int, default=66)
args = parser.parse_args()

device = torch.device(args.device)


def build_model(predefined_A):
    model = DGCRN(
        gcn_depth=2,
        num_nodes=args.num_nodes,
        device=device,
        predefined_A=predefined_A,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        middle_dim=2,
        seq_length=args.seq_in_len,
        in_dim=2,
        out_dim=args.seq_out_len,
        layers=3,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=3,
        cl_decay_steps=2000,
        rnn_size=64,
        hyperGNN_dim=16
    )
    return model


if __name__ == '__main__':

    # 1. 加载数据
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # 2. 加载邻接矩阵
    predefined_A = load_adj(args.adj_data)
    predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]

    # 3. 构建模型
    model = build_model(predefined_A).to(device)

    # 4. 加载训练好的权重
    print("Loading model:", args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 5. 评估
    outputs = []
    realy = torch.Tensor(dataloader['y_test'].astype(float)).to(device).transpose(1, 3)[:, 0, :, :]

    with torch.no_grad():
        for x, y in dataloader['test_loader'].get_iterator():
            x, y = x.astype(float), y.astype(float)
            x = torch.Tensor(x).to(device).transpose(1, 3)
            y = torch.Tensor(y).to(device).transpose(1, 3)

            preds = model(x, ycl=y)
            preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # 6. 输出测试指标
    print("\n===== TEST RESULTS =====")
    for i in [2, 5, 8, 11]:
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        mae, mape, rmse = metric(pred, real)
        print(f"Horizon {i+1}: MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}")

    amae = []
    amape = []
    armse = []
    print('                MAE\t\tRMSE\t\tMAPE')
    for (l, r) in [(0, 66)]:
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:, l:r, i])
            real = realy[:, l:r, i]
            metrics = metric(pred, real)
            # log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, metrics[0], metrics[2], metrics[1] * 100))
            # print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
        metrics = metric(scaler.inverse_transform(yhat[:, l:r]), realy[:, l:r])
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' % (metrics[0], metrics[2], metrics[1] * 100))
        print('\n')
