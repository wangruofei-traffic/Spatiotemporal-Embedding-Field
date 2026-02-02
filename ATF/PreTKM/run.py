import torch
import numpy as np
import argparse
import time
import util
import copy
from trainer import Trainer
import os


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
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
parser.add_argument('--patience',type=int,default=25,help='early stopping')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=1000,help='')
parser.add_argument('--print_every',type=int,default=200,help='')
parser.add_argument('--save',type=str,default='./garage/yinchuan',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()
print(args)

def sample_period(x):
    # trainx (B, N, T, F)
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

    trainer = Trainer(args, scaler, supports, edge_indices)


    
    print("start training...",flush=True)
    his_loss =[]
    test_time = []
    val_time = []
    train_time = []

    wait = 0
    min_val_loss = np.inf
    
    for i in range(1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x, y = x.astype(float), y.astype(float)

            trainx = torch.Tensor(x).to(args.device) # (B, T, N, F)
            trainx = trainx.transpose(1, 3) # (B, N, T, F)
            trainy = torch.Tensor(y).to(args.device) # (B, T, N, F)
            trainy = trainy.transpose(1, 3) # (B, N, T, F)
            


            metrics = trainer.train(trainx[:,:,:,:], trainy)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            t2 = time.time()
            train_time.append(t2-t1)
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        trainer.scheduler.step()

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            x, y = x.astype(float), y.astype(float)
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 3)


            metrics = trainer.eval(testx[:, :, :, :], testy)

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
        
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x, y = x.astype(float), y.astype(float)
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 3)
            

            metrics = trainer.eval(testx[:,:,:,:], testy)
                
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

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
        
        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

    # test
    trainer.model.load_state_dict(best_state_dict)
    test_loss = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_mape = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_rmse = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x, y = x.astype(float), y.astype(float)
        testx = torch.Tensor(x).to(args.device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(args.device)
        testy = testy.transpose(1, 3)
        

        metrics = trainer.eval(testx[:,:,:,:], testy, flag='horizon')
            
        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])
    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    test_time.append(s2-s1)
    amae = []
    amape = []
    armse = []
    for k in range(12):
        amae.append(np.mean(test_loss[str(k)]))
        amape.append(np.mean(test_mape[str(k)]))
        armse.append(np.mean(test_rmse[str(k)]))
        log = 'Model performance for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.5f}, Test RMSE: {:.4f}'
        print(log.format(k+1, amae[-1], amape[-1], armse[-1]))

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.5f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
