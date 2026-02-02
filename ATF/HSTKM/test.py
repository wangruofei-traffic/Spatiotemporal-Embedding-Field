import torch
import numpy as np
import argparse
import time
import util
from trainer import Trainer
import os
import matplotlib.pyplot as plt  # 新增导入

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

edges = [
    (0,14,55), (0,15,60), (1,17,37), (1,16,59), (2,18,44), (2,19,51),
    (3,20,50), (3,21,53), (4,22,50), (4,23,53), (5,24,48), (5,25,57),
    (6,26,46), (6,27,49), (7,1,34), (7,0,42), (7,2,45), (8,3,38),
    (8,4,43), (9,5,28), (9,6,30), (9,7,32), (10,9,31), (10,8,65),
    (11,10,26), (11,11,29), (12,13,27), (12,12,62), (26,90,25), (26,45,62),
    (27,103,24), (27,70,29), (28,89,24), (28,44,26), (29,102,22), (29,68,30),
    (29,69,32),(30,92,23), (30,51,65), (31,106,22), (31,77,28), (31,78,32), (32,76,35),
    (32,74,40), (32,75,47), (33,91,22), (33,49,28), (33,50,30), (34,47,33),
    (34,46,40), (34,48,47), (35,105,20), (35,72,42), (35,73,45), (36,86,14),(36,37,59),
    (37,59,39), (37,60,41), (38,36,36), (39,98,21), (39,58,43), (40,43,36),
    (41,65,33), (41,67,35), (41,66,47), (42,85,21), (42,35,38), (43,97,20),
    (43,56,34), (43,57,45), (44,84,20), (44,34,34), (44,33,42), (45,96,15),
    (45,55,51), (46,41,33), (46,42,35), (46,40,40), (47,101,19), (47,64,49),
    (48,88,19), (48,39,46), (49,100,18), (49,63,57), (50,83,15), (50,32,44),
    (51,94,16), (51,95,17), (51,54,53), (52,81,16), (52,82,17), (52,31,50),
    (53,93,13), (53,53,54), (54,52,60), (55,29,52), (55,30,56), (56,87,18),
    (56,38,48), (57,62,54), (58,99,14), (58,61,37), (61,80,13), (61,28,55),
    (63,104,25), (63,71,27), (64,107,23), (64,79,31)
]



def main():
    args.device = torch.device(args.device)
    predefined_A = util.load_adj(args.adjdata)
    supports = [torch.tensor(adj).to(args.device) for adj in predefined_A]
    dataloader = util.load_dataset(args)
    scaler = dataloader['scaler']
    edge_indices = torch.nonzero(supports[0] > 0)

    trainer = Trainer(args, scaler, supports, edge_indices, edges)
    best_path = f'{args.save}/best_model.pth'
    print(f"Loading trained model: {best_path}")
    trainer.model.load_state_dict(torch.load(best_path, map_location=args.device))



    print("Start TEST ONLY ...")

    test_loss = {str(k): [] for k in range(12)}
    test_mape = {str(k): [] for k in range(12)}
    test_rmse = {str(k): [] for k in range(12)}

    all_pred_selected = []  # 用于保存的非重叠样本
    all_real_selected = []

    sample_interval = 12
    sample_count = 0

    t_start = time.time()
    for iter, (x, y, x_speed, y_speed) in enumerate(dataloader['test_loader'].get_iterator()):
        x, y, x_speed, y_speed = x.astype(float), y.astype(float), x_speed.astype(float), y_speed.astype(float)
        testx = torch.Tensor(x).to(args.device).transpose(1, 3)
        testy = torch.Tensor(y).to(args.device).transpose(1, 3)
        testx_speed = torch.Tensor(x_speed).to(args.device)
        testy_speed = torch.Tensor(y_speed).to(args.device)


        metrics = trainer.eval(testx[:, :, :, :], testy, testx_speed, testy_speed, flag='horizon', return_data=True)

        _, _, _, predict, real = metrics[:5]  # (1, N, 12)

        # # 非重叠采样收集
        # if sample_count % sample_interval == 0:
        #     all_pred_selected.append(predict.squeeze(0))  # 已经是 numpy，直接用
        #     all_real_selected.append(real.squeeze(0))  # 已经是 numpy，直接用   # (N, 12)
        #
        # sample_count += 1

        # 指标始终基于所有样本
        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])

    t_end = time.time()
    print(f"Test Inference Time: {t_end - t_start:.4f} secs")

    # 输出指标（保持原样）
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

    # ------------------- 保存非重叠采样的预测和真实值 -------------------
    # save_dir = r'D:\python\pycharm\python项目\self_knowledge_net\MY moud\ASTDGCN+SPEED\拟合图'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, 'HSTKM_predictions_non_overlap.npz')
    #
    # np.savez(save_path,
    #          pred=np.stack(all_pred_selected, axis=0),   # (num_selected_samples, N, 12)
    #          real=np.stack(all_real_selected, axis=0))   # (num_selected_samples, N, 12)
    #
    # print(f"非重叠采样预测数据已保存至: {save_path}")
    # print(f"选出样本数: {len(all_pred_selected)}, 总点数: {len(all_pred_selected) * 66 * 12}")

if __name__ == "__main__":
    main()