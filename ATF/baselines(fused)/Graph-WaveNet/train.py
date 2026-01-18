import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import datetime

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--data',type=str,default='data/YINCHUAN',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adjacent.npz',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=64,help='')
parser.add_argument('--in_dim',type=int,default=34,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=66,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=128,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=500,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage1/yinchuan',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--runs', type=int, default=1, help='number of runs')

args = parser.parse_args()

import os
if not os.path.exists('./garage/'):
    os.makedirs('./garage/')

def main(runid, patience=20):  # 新增 patience 参数
    device = torch.device(args.device)
    if args.adjdata == 'data/sensor_graph/adjacent.npz':
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_graph_adj(args.adjdata,args.adjtype)
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    adjinit = None if args.randomadj else supports[0]
    if args.aptonly: supports = None

    num_nodes = adj_mx[-1].shape[0]
    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    min_val_loss = float('inf')
    no_improve_count = 0   # 连续没有提升的 epoch 数
    iteration = 0
    start_time = datetime.datetime.now()

    for i in range(1, args.epochs + 1):
        # 训练部分
        train_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            iteration += 1
            x, y = x.astype(float), y.astype(float)
            trainx = torch.from_numpy(x).float().to(device).transpose(1, 3)
            trainy = torch.from_numpy(y).float().to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
        t2 = time.time()
        train_time.append(t2-t1)

        # 验证部分
        valid_loss = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x, y = x.astype(float), y.astype(float)
            testx = torch.from_numpy(x).float().to(device).transpose(1,3)
            testy = torch.from_numpy(y).float().to(device).transpose(1,3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
        s2 = time.time()
        val_time.append(s2-s1)

        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)
        print(f"Epoch {i}, Train Loss: {np.mean(train_loss):.4f}, Val Loss: {mvalid_loss:.4f}")

        # 早停逻辑
        if mvalid_loss < min_val_loss:
            min_val_loss = mvalid_loss
            no_improve_count = 0
            torch.save(engine.model.state_dict(), args.save)
            print(f"Validation loss decreased to {min_val_loss:.4f}, model saved.")
        else:
            no_improve_count += 1
            print(f"Validation loss did not improve for {no_improve_count} epoch(s).")

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {i} epochs.")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save))


    outputs = []
    realy = torch.Tensor(dataloader['y_test'].astype(float)).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x, y = x.astype(float), y.astype(float)
        testx = torch.from_numpy(x).float().to(device).transpose(1, 3)
        testy = torch.from_numpy(y).float().to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx,testy).transpose(1,3)
        outputs.append(preds.squeeze(axis=1))

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    return amae, amape, armse


if __name__ == "__main__":
    # t1 = time.time()
    # main()
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2-t1))
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        # if args.TEST_ONLY:
        #     main(i)
        # else:
        m1, m2, m3 = main(i)
        # vmae.append(vm1)
        # vmape.append(vm2)
        # vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    # print('\n\nResults for 10 runs\n\n')
    # # valid data
    # print('valid\tMAE\tRMSE\tMAPE')
    # log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    # print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    # log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    # print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    # print('\n\n')
    # test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2, 5, 11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))
