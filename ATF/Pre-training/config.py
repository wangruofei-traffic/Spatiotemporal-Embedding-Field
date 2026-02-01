import torch
import numpy as np
import random
from pathlib import Path

# ==========================================
# 1. 动态路径配置 (基于当前脚本位置)
# ==========================================
# 获取当前 config.py 文件所在的根目录 (即 ATF/Pre-training/)
BASE_DIR = Path(__file__).resolve().parent

# 动态定位 data 目录下的文件
DATA_FLOW = BASE_DIR / "data" / "flow_data_flat_with_weekday.csv"
DATA_EDGE = BASE_DIR / "data" / "adjacent.csv"

# 输出路径：同样放在 data 目录下
OUTPUT_EMBED_CSV = BASE_DIR / "data" / "Spatiotemporal Embedding Field(flow).csv"

# ==========================================
# 2. 硬件与环境参数
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """全局随机种子设置，保证实验可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 3. 模型超参数
# ==========================================
num_nodes = 66
embed_dim = 32
hidden_dim = 64
num_time_slices = 288
num_time_slices_total = 7 * num_time_slices  # 2016步 [cite: 176, 177]
window_size = 6
lr = 0.001
epochs = 2000
patience = 70