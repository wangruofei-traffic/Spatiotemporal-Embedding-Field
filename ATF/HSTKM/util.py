import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

class DataLoader(object):
    """DataLoader using pre-defined x and y, supports padding and batching"""

    def __init__(self, x_node, y_node, x_seg, y_seg, batch_size, pad_with_last_sample=True):
        """
        x_node: (样本数, 12

12, N, 34)  → 历史节点输入（含流量+时间）
        y_node: (样本数, 12, N, 1)   → 真实未来流量
        x_seg : (样本数, 12, 108, 1) → 历史路段速度
        y_seg : (样本数, 12, 108, 1) → 未来路段速度
        """
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(x_node) % batch_size)) % batch_size
            if num_padding:
                x_node = np.concatenate([x_node, np.repeat(x_node[-1:], num_padding, axis=0)], axis=0)
                y_node = np.concatenate([y_node, np.repeat(y_node[-1:], num_padding, axis=0)], axis=0)
                x_seg = np.concatenate([x_seg, np.repeat(x_seg[-1:], num_padding, axis=0)], axis=0)
                y_seg = np.concatenate([y_seg, np.repeat(y_seg[-1:], num_padding, axis=0)], axis=0)

        self.size = len(x_node)
        self.num_batch = self.size // batch_size
        self.x_node = x_node
        self.y_node = y_node
        self.x_seg = x_seg
        self.y_seg = y_seg

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.x_node = self.x_node[perm]
        self.y_node = self.y_node[perm]
        self.x_seg = self.x_seg[perm]
        self.y_seg = self.y_seg[perm]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start = self.current_ind * self.batch_size
                end = min(self.size, (self.current_ind + 1) * self.batch_size)
                yield (
                    self.x_node[start:end],  # 历史节点输入
                    self.y_node[start:end],  # 真实未来流量
                    self.x_seg[start:end],  # 历史路段速度
                    self.y_seg[start:end]  # 未来路段速度
                )
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """Standard scaler for feature normalization"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()



def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]


def load_dataset(args):
    data = {}
    for category in ['train', 'val', 'test', 'train_speed', 'val_speed', 'test_speed']:
        cat_data = np.load(os.path.join(args.data_path, category + '.npz'))
        data['x_' + category] = cat_data['x']      # (样本, 12, N, 34)
        data['y_' + category] = cat_data['y']      # (样本, 12, N, 1) 或 (样本, 12, N)

    # 流量标准化
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for cat in ['train', 'val', 'test']:
        data['x_' + cat][..., 0] = scaler.transform(data['x_' + cat][..., 0])

    # 路段速度标准化（关键！）
    scaler_speed = StandardScaler(mean=data['x_train_speed'][..., 0].mean(),
                                  std=data['x_train_speed'][..., 0].std())
    for cat in ['train', 'val', 'test']:
        data['x_' + cat + '_speed'][..., 0] = scaler_speed.transform(data['x_' + cat + '_speed'][..., 0])

    # 构造干净的 ycl（未来辅助特征，不含真实流量）
    def make_ycl(y_data):
        ycl = y_data.copy()
        ycl[..., 0] = 0  # 把流量通道清零
        return ycl

    data['train_loader'] = DataLoader(
        data['x_train'],
        data['y_train'],
        data['x_train_speed'],   # 关键！保留108条路段，只取速度列
        data['y_train_speed'],
        args.batch_size
    )

    data['val_loader'] = DataLoader(
        data['x_val'],
        data['y_val'],
        data['x_val_speed'],
        data['y_val_speed'],
        args.batch_size
    )

    data['test_loader'] = DataLoader(
        data['x_test'],
        data['y_test'],
        data['x_test_speed'],
        data['y_test_speed'],
        args.batch_size
    )

    data['scaler'] = scaler

    print("Dataset loaded successfully!")
    print(f"Train samples: {len(data['x_train'])} | Val: {len(data['x_val'])} | Test: {len(data['x_test'])}")
    return data

