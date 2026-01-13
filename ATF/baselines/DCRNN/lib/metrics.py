import torch
import numpy as np

# -------------------- Masked Loss Functions --------------------
def masked_mse_torch(preds, labels, null_val=np.nan):
    """MSE with masking for PyTorch tensors"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= mask.mean()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_mae_torch(preds, labels, null_val=np.nan):
    """MAE with masking for PyTorch tensors"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= mask.mean()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_rmse_torch(preds, labels, null_val=np.nan):
    """RMSE with masking for PyTorch tensors"""
    return torch.sqrt(masked_mse_torch(preds, labels, null_val))

# -------------------- Numpy Metrics --------------------
def masked_mse_np(preds, labels, null_val=np.nan):
    mask = ~np.isnan(labels) if np.isnan(null_val) else labels != null_val
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mse = np.square(preds - labels).astype('float32')
    mse = np.nan_to_num(mask * mse)
    return np.mean(mse)

def masked_mae_np(preds, labels, null_val=np.nan):
    mask = ~np.isnan(labels) if np.isnan(null_val) else labels != null_val
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mae = np.abs(preds - labels).astype('float32')
    mae = np.nan_to_num(mask * mae)
    return np.mean(mae)

def masked_mape_np(preds, labels, null_val=np.nan):
    mask = ~np.isnan(labels) if np.isnan(null_val) else labels != null_val
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mape = np.abs((preds - labels).astype('float32') / labels)
    mape = np.nan_to_num(mask * mape)
    return np.mean(mape)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds, labels, null_val))

# -------------------- Loss Builder --------------------
def masked_mse_loss(scaler=None, null_val=np.nan):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_torch(preds, labels, null_val)
    return loss

def masked_rmse_loss(scaler=None, null_val=np.nan):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_torch(preds, labels, null_val)
    return loss

def masked_mae_loss(scaler=None, null_val=np.nan):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mae_torch(preds, labels, null_val)
    return loss

# -------------------- Metric Calculation --------------------
def calculate_metrics(df_pred, df_test, null_val=np.nan):
    """
    Calculate MAE, MAPE, RMSE using numpy
    :param df_pred: pandas DataFrame
    :param df_test: pandas DataFrame
    :param null_val: value to mask
    :return: mae, mape, rmse
    """
    preds = df_pred.to_numpy()
    labels = df_test.to_numpy()
    mae = masked_mae_np(preds, labels, null_val)
    mape = masked_mape_np(preds, labels, null_val)
    rmse = masked_rmse_np(preds, labels, null_val)
    return mae, mape, rmse
