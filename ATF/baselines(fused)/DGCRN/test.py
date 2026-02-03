import torch
import numpy as np
import argparse
import os
from util import load_dataset, metric, load_adj
from net import DGCRN

def str_to_bool(x):
    return x.lower() in ['true','1','yes','y']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data', type=str, default='data/YINCHUAN')
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_yinchuan.pkl')
    parser.add_argument('--model_path', type=str, default='./model.save-1/exp1_0.pth')

    parser.add_argument('--batch_size', type=int, default=64)

    # 模型必要参数
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=66)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--subgraph_size', type=int, default=20)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--tanhalpha', type=float, default=3.0)
    parser.add_argument('--cl_decay_steps', type=float, default=2000)
    parser.add_argument('--rnn_size', type=int, default=64)
    parser.add_argument('--hyperGNN_dim', type=int, default=16)

    return parser.parse_args()



def main():
    args = get_args()
    device = torch.device(args.device)

    print("Loading data...")
    dataloader = load_dataset(args.data,
                              args.batch_size,
                              args.batch_size,
                              args.batch_size)
    scaler = dataloader['scaler']

    print("Loading adjacency matrix...")
    predefined_A = load_adj(args.adj_data)
    predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]

    print("Building model...")
    model = DGCRN(
        args.gcn_depth,
        args.num_nodes,
        device,
        predefined_A=predefined_A,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        middle_dim=64,
        seq_length=args.seq_in_len,
        in_dim=args.in_dim,
        out_dim=args.seq_out_len,
        layers=args.layers,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=args.tanhalpha,
        cl_decay_steps=args.cl_decay_steps,
        rnn_size=args.rnn_size,
        hyperGNN_dim=args.hyperGNN_dim
    )

    print("Loading trained weights from:", args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Start testing...")
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]   # shape: [B, N, T]

    with torch.no_grad():
        for x, y in dataloader['test_loader'].get_iterator():
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)

            preds = model(testx, ycl=testy)
            preds = preds.transpose(1, 3)  # [B, 1, N, T]
            outputs.append(preds.squeeze(dim=1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # 计算 4 个 horizon
    mae_list, mape_list, rmse_list = [], [], []
    print("\n===== Test Results =====")
    for i in [2, 5, 8, 11]:
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        m = metric(pred, real)

        print(f"Horizon {i+1:2d}: MAE={m[0]:.4f}, MAPE={m[1]:.4f}, RMSE={m[2]:.4f}")

        mae_list.append(m[0])
        mape_list.append(m[1])
        rmse_list.append(m[2])

    print("\nAverage:")
    print(f"MAE  = {np.mean(mae_list):.4f}")
    print(f"MAPE = {np.mean(mape_list):.4f}")
    print(f"RMSE = {np.mean(rmse_list):.4f}")

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


if __name__ == "__main__":
    main()
