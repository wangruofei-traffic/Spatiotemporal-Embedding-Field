import torch
import pandas as pd
from config import *
from data_utils import load_adjacency_matrix, load_flow_data
from model import STDiffusionNet


def export_embedding(model, save_path, final_mode=False):
    """
    将预训练的时空嵌入场导出到 CSV。
    """
    emb = model.init_emb.detach().cpu().numpy()
    data = []
    if not final_mode:
        for t in range(emb.shape[0]):
            for n in range(num_nodes):
                data.append([t, n] + emb[t, n].tolist())
        df = pd.DataFrame(data, columns=["time_idx", "node"] + [f"emb_{i}" for i in range(embed_dim)])
    else:
        for t in range(emb.shape[0]):
            weekday, ts = t // num_time_slices + 1, t % num_time_slices
            for n in range(num_nodes):
                data.append([weekday, ts, n] + emb[t, n].tolist())
        cols = ['weekday', 'time_slot', 'node'] + [f'emb_{i}' for i in range(embed_dim)]
        df = pd.DataFrame(data, columns=cols)
    df.to_csv(save_path, index=False)


def train_process():
    # 1. 初始化数据与邻接矩阵
    A_f, A_b = load_adjacency_matrix()
    time_data = load_flow_data()
    print(f"数据准备完毕，设备: {device}")

    # 2. 实例化模型
    model = STDiffusionNet(num_nodes, embed_dim, hidden_dim, K=2, window_size=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae = float('inf')
    best_state = None
    stop_counter = 0

    # 3. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向计算损失
        _, loss, rmse, mae = model(A_f, A_b, time_data)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

        # 早停与最优模型保存逻辑
        if mae < best_mae:
            best_mae = mae
            best_state = model.state_dict()
            stop_counter = 0
            export_embedding(model, OUTPUT_EMBED_CSV, final_mode=False)
            print(f">>> 最优嵌入已保存至: {OUTPUT_EMBED_CSV}")
        else:
            stop_counter += 1
            print(f"早停计数: {stop_counter}")

        if stop_counter >= patience:
            print(f"触发早停，最佳 MAE: {best_mae:.6f}")
            break

    # 4. 最终导出
    if best_state:
        model.load_state_dict(best_state)
        export_embedding(model, OUTPUT_EMBED_CSV, final_mode=True)
        print("预训练完成，最优时空嵌入已最终导出。")


if __name__ == "__main__":
    train_process()