import torch
import numpy as np
import argparse
from util import *
from trainer import Trainer
from net import gtnet
import os

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/YINCHUAN', help='data path')
parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adjacent.npz', help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True)
parser.add_argument('--buildA_true', type=str_to_bool, default=True)
parser.add_argument('--load_static_feature', type=str_to_bool, default=False)
parser.add_argument('--cl', type=str_to_bool, default=True)

parser.add_argument('--gcn_depth', type=int, default=2)
parser.add_argument('--num_nodes', type=int, default=66)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--subgraph_size', type=int, default=20)
parser.add_argument('--node_dim', type=int, default=40)
parser.add_argument('--dilation_exponential', type=int, default=1)
parser.add_argument('--conv_channels', type=int, default=32)
parser.add_argument('--residual_channels', type=int, default=32)
parser.add_argument('--skip_channels', type=int, default=64)
parser.add_argument('--end_channels', type=int, default=128)

parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--seq_in_len', type=int, default=12)
parser.add_argument('--seq_out_len', type=int, default=12)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--clip', type=int, default=5)

parser.add_argument('--propalpha', type=float, default=0.05)
parser.add_argument('--tanhalpha', type=float, default=3)

parser.add_argument('--num_split', type=int, default=1)
parser.add_argument('--save', type=str, default='experiments/', help='model path to load')

args = parser.parse_args()
torch.set_num_threads(3)


def test_only():
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    predefined_A = load_adjacent(args.adj_data)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    # 载入训练好的模型权重
    model_path = args.save + args.data.split('/')[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, 0, args.seq_out_len, scaler, device, args.cl)

    # test data
    outputs = []
    realy = torch.tensor(np.array(dataloader['y_test'], dtype=np.float32)).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.tensor(np.array(x, dtype=np.float32)).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae, mape, rmse = [], [], []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        print(f'Horizon {i + 1}, Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}')
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])

    print('Average across all horizons -> MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(
        np.mean(mae), np.mean(mape), np.mean(rmse)
    ))
    for (l,r) in [(0,66)]:
        for i in range(args.seq_out_len):
            pred = scaler.inverse_transform(yhat[:, l:r, i])
            real = realy[:, l:r, i]
            metrics = metric(pred, real)
            print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, metrics[0], metrics[2], metrics[1] * 100))
            # log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            # print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        metrics = metric(scaler.inverse_transform(yhat[:,l:r]),realy[:,l:r])
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(metrics[0], metrics[2], metrics[1] * 100))
        print('\n')


if __name__ == "__main__":
    test_only()
