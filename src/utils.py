# src/utils.py

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import xarray as xr
from . import config
import inspect
import json

# --- 文件与目录工具 ---
def ensure_dir(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# --- 加载/保存工具 ---
def load_single_nc_file(file_path):
    """
    加载单个 NetCDF 文件，返回 SST 数据和经纬度坐标。
    """
    ds = xr.open_dataset(file_path)
    sst_data = ds[config.SST_VARIABLE_NAME]
    if 'time' in sst_data.dims and len(sst_data.time) == 1:
        sst_data = sst_data.squeeze('time')
    # 确保维度顺序和坐标升序
    if sst_data.dims != ('latitude', 'longitude'):
        try:
            sst_data = sst_data.transpose('latitude', 'longitude')
        except ValueError:
            print(f"警告: 无法将 {file_path} 的维度转置为 ('latitude', 'longitude')。")
    if sst_data['latitude'].values[0] > sst_data['latitude'].values[-1]:
        sst_data = sst_data.reindex(latitude=list(reversed(sst_data['latitude'].values)))
    return sst_data, ds.get(config.LON_VARIABLE_NAME), ds.get(config.LAT_VARIABLE_NAME)

def save_checkpoint(epoch, model, optimizer, loss, filepath):
    """保存模型checkpoint"""
    ensure_dir(os.path.dirname(filepath))

    # 检查模型是否被DataParallel包装，如果是，则获取其内部的.module
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint 已保存至: {filepath}")

def load_checkpoint(filepath, model, device):
    """加载模型checkpoint到指定设备"""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"模型权重已从 {filepath} 加载 (Epoch {checkpoint.get('epoch', 'N/A')}, Loss {checkpoint.get('loss', 'N/A'):.4f})")
        return model
    except FileNotFoundError:
        print(f"错误: Checkpoint文件 {filepath} 未找到。")
        raise
    except Exception as e:
        print(f"加载checkpoint {filepath} 失败: {e}")
        raise

def save_as_netcdf(date_np:np.ndarray,filepath:str,lat_np:np.ndarray,lon_np:np.ndarray,
                   lat_name:str='latitude',lon_name:str='longitude',var_name:str='sst'):
    """将np数组保存为nc文件"""

    # # 检查维度是否匹配
    # if date_np.shape != (lat_np.size, lon_np.size):
    #     raise ValueError(f"数据 shape 与坐标不匹配: data shape = {date_np.shape}, lat = {lat_np.shape}, lon = {lon_np.shape}")
    # print(f"data shape = {date_np.shape}, lat = {lat_np.shape}, lon = {lon_np.shape}")

    coords_dict = {
        lat_name:lat_np,
        lon_name:lon_np
    }
    dims = (lat_name,lon_name)

    data_xr = xr.DataArray(data=date_np,dims=dims,coords=coords_dict,name=var_name)
    data_xr.to_netcdf(filepath)
    print(f"预测结果已保存为NetCDF文件: {filepath}")

def save_config(file_name):
    """ 保存config中所有的大写变量 """
    config_dict = {}
    for key , value in inspect.getmembers(config):
        if key.isupper():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
    with open(file_name, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"配置已经成功保存至：{file_name}")

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
def calculate_metrics(predicted_np:np.ndarray, target_np:np.ndarray,land_sea_mask_np:np.ndarray):
    """计算predict和target的MSE,RMSE"""

    # 选择海洋点
    ocean_pred = predicted_np[land_sea_mask_np]
    ocean_target = target_np[land_sea_mask_np]

    # print(f"pred has nan?{np.isnan(ocean_pred).any()}. target has nan?{np.isnan(ocean_target).any()}")
    mse = mean_squared_error(ocean_pred, ocean_target)
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
    print(f"损失曲线图已保存至: {filepath}")


def save_img(
        pred_np: np.ndarray,
        target_np: np.ndarray,
        filepath: str,
        title_prefix: str = ""
):
    """
    保存图像
    target_np = None => 保存预测图像
    保存预测、目标与差值图像
    """

    def ensure_dir(directory: str):
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    ensure_dir(os.path.dirname(filepath))

    if target_np is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(pred_np, cmap='coolwarm')
        ax.set_title(f"{title_prefix} Predicted SST")
        fig.colorbar(im, ax=ax)  
        output_info = f"预测图象已保存到: {filepath}"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(pred_np, cmap='coolwarm',origin='lower')
        axes[0].set_title(f"{title_prefix} Predicted SST")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(target_np, cmap='coolwarm',origin='lower')
        axes[1].set_title(f"{title_prefix} Target SST")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(pred_np - target_np, cmap='RdBu_r',origin='lower')
        axes[2].set_title(f"{title_prefix} Difference (Predicted - Target)")
        fig.colorbar(im2, ax=axes[2])

        output_info = f"对比图像已保存到: {filepath}"

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)
    print(output_info)

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

def setup_train_directory(run_name=None):
    """为单次训练运行创建并返回一个带时间戳的专属目录。"""
    now = datetime.now()
    # 目录名格式: YYYYMMDD_HH
    if run_name is None:
        train_run_dir = os.path.join(config.CHECKPOINT_PATH, now.strftime("%Y%m%d_%H"))
    else:
        train_run_dir = os.path.join(config.CHECKPOINT_PATH, run_name)
    ensure_dir(train_run_dir) 
    print(f"本次训练产出将保存在: {train_run_dir}")
    return train_run_dir

def setup_inference_directory(checkpoint_relative_path: str):
    """根据所使用的checkpoint路径，为单次推理运行创建专属结果目录。"""
    # 例如, checkpoint_relative_path = "checkpoints/20250609_21/model_final.pth"
    # 我们希望生成 "20250609_21_model_final"
    dir_name = checkpoint_relative_path.replace("\\",'/').split("checkpoints/")[-1]
    dir_name = dir_name.replace("/", "_").replace(".pth", "")
    inference_run_dir = os.path.join(config.RESULTS_PATH, dir_name)
    
    inference_run_dir = os.path.join(config.RESULTS_PATH, dir_name)
    inference_images_path = os.path.join(inference_run_dir, "images")
    inference_data_path = os.path.join(inference_run_dir, "predicted_data")
    for path in [inference_images_path, inference_data_path]:
        ensure_dir(path)

    print(f"本次推理结果将保存在: {inference_run_dir}")
    return {
        "base": inference_run_dir,
        "images": inference_images_path,
        "data": inference_data_path
    }