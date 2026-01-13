import unittest
import numpy as np
import torch
from lib import metrics  # 假设你已经把 masked_mse_tf 换成 PyTorch 版本

class MyTestCase(unittest.TestCase):
    def test_masked_mape_np(self):
        preds = np.array([[1, 2, 2], [3, 4, 5]], dtype=np.float32)
        labels = np.array([[1, 2, 2], [3, 4, 4]], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 24.0, mape, delta=1e-5)

    def test_masked_mape_np2(self):
        preds = np.array([[1, 2, 2], [3, 4, 5]], dtype=np.float32)
        labels = np.array([[1, 2, 2], [3, 4, 4]], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels, null_val=4)
        self.assertEqual(0., mape)

    def test_masked_rmse_torch(self):
        preds = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        labels = torch.tensor([[1, 0], [3, 3]], dtype=torch.float32)
        rmse = metrics.masked_rmse_torch(preds, labels, null_val=0)  # 假设你实现了 PyTorch 版本
        self.assertAlmostEqual(np.sqrt(1 / 3.), rmse.item(), delta=1e-5)

    def test_masked_mse_torch_nan(self):
        preds = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        labels = torch.tensor([[1, 2], [3, float('nan')]], dtype=torch.float32)
        mse = metrics.masked_mse_torch(preds, labels)  # PyTorch 版本
        self.assertEqual(0., mse.item())

if __name__ == '__main__':
    unittest.main()
