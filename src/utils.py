# src/utils.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import xarray as xr
from . import config

# --- 文件与目录工具 ---
def ensure_dir(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# --- 模型加载/保存工具 ---
def save_checkpoint(epoch, model, optimizer, loss, filepath):
    """保存模型checkpoint"""
    ensure_dir(os.path.dirname(filepath))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint 已保存至: {filepath}")

def load_checkpoint(filepath, model, device):
    """加载模型checkpoint到指定设备"""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型权重已从 {filepath} 加载 (Epoch {checkpoint.get('epoch', 'N/A')}, Loss {checkpoint.get('loss', 'N/A'):.4f})")
        return model
    except FileNotFoundError:
        print(f"错误: Checkpoint文件 {filepath} 未找到。")
        raise
    except Exception as e:
        print(f"加载checkpoint {filepath} 失败: {e}")
        raise

# --- 数据转换工具 ---
def denormalize_sst(normalized_sst_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """将归一化到 [-1, 1] 的SST数据反归一化回原始范围。"""
    if max_val == min_val:
        return torch.full_like(normalized_sst_tensor, min_val)
    original_sst = (normalized_sst_tensor + 1.0) * (max_val - min_val) / 2.0 + min_val
    return original_sst

def reconstruct_image_from_patches(
    patches_dict: dict, 
    image_dims: tuple,
    patch_height: int,
    patch_width: int
) -> torch.Tensor:
    """从patches字典中重建完整图像，重叠区域取平均。"""
    H, W = image_dims
    reconstructed_image = torch.zeros((H, W), dtype=torch.float32)
    count_map = torch.zeros((H, W), dtype=torch.float32)
    for (r, c), patch_tensor in patches_dict.items():
        reconstructed_image[r:r+patch_height, c:c+patch_width] += patch_tensor
        count_map[r:r+patch_height, c:c+patch_width] += 1
    reconstructed_image = torch.where(count_map > 0, reconstructed_image / count_map, 0.0)
    return reconstructed_image

# --- 评估与绘图工具 ---
def calculate_metrics(predicted_np:np.ndarray, target_np:np.ndarray):
    """计算predict和target的MSE,RMSE"""
    mse = mean_squared_error(predicted_np, target_np)
    rmse = np.sqrt(mse)
    return mse, rmse

def plot_loss_curve(epoch_losses_list: list, filepath: str):
    """绘制并保存loss曲线图。"""
    ensure_dir(os.path.dirname(filepath))
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses_list, label='Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"损失曲线图已保存至: {filepath}")


def save_img_comparison(
        predicted_np: np.ndarray,
        target_np: np.ndarray,
        land_sea_mask_np: np.ndarray,
        filepath: str,
        title_prefix: str = ""
):
    """保存预测、目标与差值图像（带色条）"""

    def ensure_dir(directory: str):
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    ensure_dir(os.path.dirname(filepath))
    pred_masked = np.where(land_sea_mask_np, predicted_np, np.nan)

    if target_np is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(pred_masked, cmap='coolwarm')
        ax.set_title(f"{title_prefix} Predicted SST")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  
    else:
        target_masked = np.where(land_sea_mask_np, target_np, np.nan)
        diff_masked = np.where(land_sea_mask_np, predicted_np - target_np, np.nan)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(pred_masked, cmap='coolwarm',origin='upper')
        axes[0].set_title(f"{title_prefix} Predicted SST")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(target_masked, cmap='coolwarm',origin='upper')
        axes[1].set_title(f"{title_prefix} Target SST")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff_masked, cmap='RdBu_r',origin='upper')
        axes[2].set_title(f"{title_prefix} Difference (Predicted - Target)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()
    print(f"图像对比图已保存到: {filepath}")

def check_date_contiguity(date_str_list:list,date_formate:str='%Y-%m-%d') -> bool:
    """
    检查日期列表是否连续。
    """
    if len(date_str_list) < 2:
        return True
    try:
        dates = [datetime.strptime(date_str, date_formate) for date_str in date_str_list]
    except:
        print("日期格式错误，请检查输入的日期字符串。")
        return False
    
    # 日期间隔
    one_day_delta = timedelta(days=1)

    for i in range(1, len(dates)):
        if dates[i] - dates[i - 1] != one_day_delta:
            print(f"日期 {dates[i]} 和 {dates[i - 1]} 不连续。")
            return False
    return True

# --- 结果保存工具 ---
def save_as_netcdf(date_np:np.ndarray,coords_dict:dict,filepath:str,var_name:str='sst'):
    """将np数组保存为nc文件"""
    ensure_dir(filepath)
    dims = ('latitude','longtitude')
    data_xr = xr.DataArray(data=date_np,dims=dims,coords=coords_dict,name=var_name)
    data_xr.to_netcdf(filepath)
    print(f"预测结果已保存为NetCDF文件: {filepath}")

def setup_train_directory():
    """为单次训练运行创建并返回一个带时间戳的专属目录。"""
    now = datetime.now()
    # 目录名格式: YYYYMMDD_HH
    train_run_dir = os.path.join(config.CHECKPOINT_PATH, now.strftime("%Y%m%d_%H"))
    ensure_dir(train_run_dir) 
    print(f"本次训练产出将保存在: {train_run_dir}")
    return train_run_dir

def setup_inference_directory(checkpoint_relative_path: str):
    """根据所使用的checkpoint路径，为单次推理运行创建专属结果目录。"""
    # 例如, checkpoint_relative_path = "20250609_21/model_final.pt"
    # 我们希望生成 "20250609_21_model_final"
    dir_name = checkpoint_relative_path.replace(os.sep, "_").replace(".pt", "").replace(".pth", "")
    
    inference_run_dir = os.path.join(config.RESULTS_PATH, dir_name)
    ensure_dir(inference_run_dir)
    print(f"本次推理结果将保存在: {inference_run_dir}")
    return inference_run_dir