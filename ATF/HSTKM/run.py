import torch
import numpy as np
import argparse
import time
import util
import copy
from trainer import Trainer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data_path', type=str, default='./data/YINCHUAN', help='data path')
parser.add_argument('--adjdata', type=str, default='./data/YINCHUAN/adj_yinchuan.pkl', help='adj data path')
parser.add_argument('--input_length', type=int, default=12, help='')
parser.add_argument('--output_length', type=int, default=12, help='')
parser.add_argument('--hid_dim', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=64, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=66, help='number of nodes')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--tau', type=int, default=0.25, help='temperature coefficient')
parser.add_argument('--random_feature_dim', type=int, default=64, help='random feature dimension')
parser.add_argument('--node_dim', type=int, default=32, help='node embedding dimension')
parser.add_argument('--time_dim', type=int, default=32, help='time embedding dimension')
parser.add_argument('--time_num', type=int, default=288, help='time in day')
parser.add_argument('--week_num', type=int, default=7, help='day in week')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient cliip')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--milestones', type=list, default=[80, 100], help='optimize r milestones')
parser.add_argument('--patience', type=int, default=25, help='early stopping')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='')
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--save', type=str, default='./garage/yincuan', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()
print(args)


def sample_period(x):
    # trainx (B, N, T, F)
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i + 12] for i in range(0, history_length, args.time_num)]
    period_feat = [x[:, :, sublist, 0] for sublist in period_list]
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

    trainer = Trainer(args, scaler, supports, edge_indices,edges)

    # Default setting: input length 12 output length 12
    # Using long-term features require much more time, it would be faster to pre-compute the features and save them in disk

    print("start training...", flush=True)
    his_loss = []
    test_time = []
    val_time = []
    train_time = []

    wait = 0
    min_val_loss = np.inf

    for i in range(1, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y,x_speed,y_speed) in enumerate(dataloader['train_loader'].get_iterator()):
            x, y ,x_speed,y_speed = x.astype(float), y.astype(float),x_speed.astype(float),y_speed.astype(float)
            # trainx: long history data T=2016 or 864
            trainx = torch.Tensor(x).to(args.device)  # (B, T, N, F)
            trainx = trainx.transpose(1, 3)  # (B, N, T, F)
            trainy = torch.Tensor(y).to(args.device)  # (B, T, N, F)
            trainy = trainy.transpose(1, 3)  # (B, N, T, F)
            trainx_speed = torch.Tensor(x_speed).to(args.device)
            trainy_speed = torch.Tensor(y_speed).to(args.device)


            metrics = trainer.train(trainx[:, :, :, :], trainy,trainx_speed,trainy_speed)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            t2 = time.time()
            train_time.append(t2 - t1)

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        trainer.scheduler.step()

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y,x_speed,y_speed) in enumerate(dataloader['test_loader'].get_iterator()):
            x, y ,x_speed,y_speed = x.astype(float), y.astype(float),x_speed.astype(float),y_speed.astype(float)
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 3)
            testx_speed = torch.Tensor(x_speed).to(args.device)
            testy_speed = torch.Tensor(y_speed).to(args.device)


            metrics = trainer.eval(testx[:, :, :, :], testy,testx_speed,testy_speed)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, test MAE: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

        if mvalid_loss < min_val_loss:
            wait = 0
            min_val_loss = mvalid_loss
            best_epoch = i
            best_state_dict = copy.deepcopy(trainer.model.state_dict())
            save_path = f'{args.save}/best_model.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录（如果已经存在不会报错）
            torch.save(best_state_dict, save_path)
            print("model saved!")

        else:
            wait += 1
            print("wait:", wait)
            if wait >= args.patience:
                print("Early stopping triggered at epoch:", i)
                break

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y,x_speed,y_speed) in enumerate(dataloader['val_loader'].get_iterator()):
            x, y ,x_speed,y_speed = x.astype(float), y.astype(float),x_speed.astype(float),y_speed.astype(float)
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 3)
            testx_speed = torch.Tensor(x_speed).to(args.device)
            testy_speed = torch.Tensor(y_speed).to(args.device)

            metrics = trainer.eval(testx[:, :, :, :], testy,testx_speed,testy_speed)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

    # test
    trainer.model.load_state_dict(best_state_dict)
    test_loss = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [],
                 '11': []}
    test_mape = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [],
                 '11': []}
    test_rmse = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [],
                 '11': []}
    s1 = time.time()
    for iter, (x, y,x_speed,y_speed) in enumerate(dataloader['test_loader'].get_iterator()):
        x, y ,x_speed,y_speed = x.astype(float), y.astype(float),x_speed.astype(float),y_speed.astype(float)
        testx = torch.Tensor(x).to(args.device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(args.device)
        testy = testy.transpose(1, 3)
        testx_speed = torch.Tensor(x_speed).to(args.device)
        testy_speed = torch.Tensor(y_speed).to(args.device)


        metrics = trainer.eval(testx[:, :, :, :], testy, testx_speed,testy_speed,flag='horizon')

        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])
    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(i, (s2 - s1)))
    test_time.append(s2 - s1)
    amae = []
    amape = []
    armse = []
    for k in range(12):
        amae.append(np.mean(test_loss[str(k)]))
        amape.append(np.mean(test_mape[str(k)]))
        armse.append(np.mean(test_rmse[str(k)]))
        log = 'Model performance for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.5f}, Test RMSE: {:.4f}'
        print(log.format(k + 1, amae[-1], amape[-1], armse[-1]))

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.5f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
