import torch
import numpy as np
import argparse
import time
import util
from trainer import Trainer
import os


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default='./data/YINCHUAN',help='data path')
parser.add_argument('--adjdata', type=str, default='./data/YINCHUAN/adj_yinchuan.pkl', help='adj data path')
parser.add_argument('--input_length',type=int,default=12,help='')
parser.add_argument('--output_length',type=int,default=12,help='')
parser.add_argument('--hid_dim',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=64,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=66,help='number of nodes')
parser.add_argument('--num_layers',type=int,default=3,help='number of layers')
parser.add_argument('--tau',type=int,default=0.25,help='temperature coefficient')
parser.add_argument('--random_feature_dim',type=int,default=64,help='random feature dimension')
parser.add_argument('--node_dim',type=int,default=32,help='node embedding dimension')
parser.add_argument('--time_dim',type=int,default=32,help='time embedding dimension')
parser.add_argument('--time_num',type=int,default=288,help='time in day')
parser.add_argument('--week_num',type=int,default=7,help='day in week')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--grad_clip',type=float,default=5,help='gradient cliip')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--milestones',type=list,default=[80, 100],help='optimizer milestones')
parser.add_argument('--patience',type=int,default=20,help='early stopping')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=1000,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default='./garage/yincuan',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

print(args)


def sample_period(x):
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i+12] for i in range(0, history_length, args.time_num)]
    period_feat = [x[:,:,sublist,0] for sublist in period_list]
    period_feat = torch.stack(period_feat)
    period_feat = torch.mean(period_feat, dim=0)
    return period_feat


def main():
    args.device = torch.device(args.device)
    predefined_A = util.load_adj(args.adjdata)
    supports = [torch.tensor(adj).to(args.device) for adj in predefined_A]
    dataloader = util.load_dataset(args)
    scaler = dataloader['scaler']
    edge_indices = torch.nonzero(supports[0] > 0)

    # 初始化 Trainer
    trainer = Trainer(args, scaler, supports, edge_indices)

    # 加载 best_model
    best_path = f'{args.save}/best_model.pth'
    print(f"Loading trained model: {best_path}")
    trainer.model.load_state_dict(torch.load(best_path, map_location=args.device))


    print("Start TEST ONLY ...")

    # 原有指标收集
    test_loss = {str(k): [] for k in range(12)}
    test_mape = {str(k): [] for k in range(12)}
    test_rmse = {str(k): [] for k in range(12)}

    # 新增：非重叠采样用于画图
    all_pred_selected = []  # list of (N, 12) numpy arrays
    all_real_selected = []  # list of (N, 12) numpy arrays

    sample_interval = 12               # 每隔12个样本取一个
    global_sample_count = 0            # 全局样本计数（跨batch）

    t_start = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x, y = x.astype(float), y.astype(float)
        testx = torch.Tensor(x).to(args.device).transpose(1, 3)   # (B, N, T, C)
        testy = torch.Tensor(y).to(args.device).transpose(1, 3)   # (B, N, T, C)


        metrics = trainer.eval(testx[:, :, :, :], testy, flag='horizon', return_data=True)

        # metrics 返回: loss(list), mape(list), rmse(list), predict(np), real(np)
        # predict 和 real 形状: (B, N, 12)，且已经是 numpy（经过 inverse_transform）
        _, _, _, predict, real = metrics

        batch_size_current = predict.shape[0]

        # 非重叠采样逻辑
        for b in range(batch_size_current):
            if global_sample_count % sample_interval == 0:
                # 取出当前样本的 (N, 12)
                all_pred_selected.append(predict[b])      # (N, 12)
                all_real_selected.append(real[b])         # (N, 12)
            global_sample_count += 1

        # 原有指标收集（所有样本都要算）
        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])

    t_end = time.time()
    print(f"Test Inference Time: {t_end - t_start:.4f} secs")

    # 输出指标（保持不变）
    amae, amape, armse = [], [], []
    for k in range(12):
        mae_k = np.mean(test_loss[str(k)])
        mape_k = np.mean(test_mape[str(k)])
        rmse_k = np.mean(test_rmse[str(k)])
        amae.append(mae_k)
        amape.append(mape_k)
        armse.append(rmse_k)
        print(f"Horizon {k+1:02d} MAE: {mae_k:.3f} MAPE: {mape_k:.5f} RMSE: {rmse_k:.3f}")
    print(f"\nAverage over 12 horizons: MAE={np.mean(amae):.3f} MAPE={np.mean(amape):.5f} RMSE={np.mean(armse):.3f}")

    # # ==================== 保存非重叠采样的数据 ====================
    # save_dir = r'D:\python\pycharm\python项目\self_knowledge_net\MY moud\ASTDGCN+SPEED\拟合图'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, 'PreTKM.npz')  # 你可以改名字
    #
    # if len(all_pred_selected) > 0:
    #     np.savez(save_path,
    #              pred=np.stack(all_pred_selected, axis=0),   # (num_selected, N, 12)
    #              real=np.stack(all_real_selected, axis=0))   # (num_selected, N, 12)
    #     print(f"\n非重叠采样数据已保存！")
    #     print(f"路径: {save_path}")
    #     print(f"选出样本数: {len(all_pred_selected)}")
    #     print(f"总点数: {len(all_pred_selected) * args.num_nodes * 12}")
    # else:
    #     print("\n警告：没有采集到任何样本，请检查测试集大小是否足够大")


if __name__ == "__main__":
    main()
