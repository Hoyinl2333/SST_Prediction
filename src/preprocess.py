# src/preprocess.py

import os
import glob
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from . import config
from .utils import ensure_dir,load_single_nc_file


def crop_image(data_array: xr.DataArray, target_height: int, target_width: int) -> xr.DataArray:
    """
    裁剪 xarray.DataArray 到目标尺寸。
    假设输入 data_array 的维度是 (lat, lon)。
    """
    if data_array.shape[0] == 721 and target_height == 720:
        data_array = data_array[:-1, :]
    if data_array.shape[0] < target_height or data_array.shape[1] < target_width:
        raise ValueError("原始图像小于目标尺寸。")
    return data_array[:target_height, :target_width]



def fill_land(sst_array: xr.DataArray, method: str = None ) -> xr.DataArray:
    """
    对SST数据中的NaN值（陆地）进行插值，采用 pandas DataFrame 的插值方法，然后用0填充。
    sst_array: 原始SST数据，xarray.DataArray，维度应为 (lat, lon)
    method: pandas.DataFrame.interpolate 使用的插值方法 ('linear', 'nearest', 'quadratic', 'cubic', etc.)
    返回: 插值后的SST数据 (NaNs 被替换或填充为0)，xarray.DataArray
    """
    # 确保输入是 xarray.DataArray
    if not isinstance(sst_array, xr.DataArray):
        raise TypeError("输入必须是 xarray.DataArray 类型。")
        
    sst_values_original = sst_array.values.copy() # 获取numpy数组副本

    # 转换为pandas DataFrame
    df = pd.DataFrame(sst_values_original)

    # 如果指定了method，则使用插值，否则直接填充
    if method :
        # 1. 先按列插值 (axis=0)
        # limit_direction='both' 会尝试向前和向后填充
        df.interpolate(method=method, axis=0, limit_direction='both', inplace=True)
    
        # 2. 再按行插值 (axis=1)
        df.interpolate(method=method, axis=1, limit_direction='both', inplace=True)
    
    # 3. 将剩余的所有NaN值（如果插值未能完全覆盖）填充为0
    sst_values_filled = df.fillna(0.0).values

    return xr.DataArray(
        sst_values_filled.astype(np.float32), 
        coords=sst_array.coords, 
        dims=sst_array.dims, 
        name=sst_array.name 
    )

def get_normalization_stats(file_paths, crop_dims):
    """ 计算给定文件列表中SST数据的归一化统计参数 (最小值和最大值)。"""
    
    print(f"开始计算归一化统计参数，使用 {len(file_paths)} 个训练文件...")
    
    all_sst_values = []
    for file_path in tqdm(file_paths, desc="读取训练文件计算统计参数"):
        sst_array, _, _ = load_single_nc_file(file_path)
        sst_cropped = crop_image(sst_array, crop_dims[0], crop_dims[1])
        valid_values = sst_cropped.values
        all_sst_values.append(valid_values[~np.isnan(valid_values)])
    all_sst_values_np = np.concatenate(all_sst_values)
    min_sst, max_sst = np.nanmin(all_sst_values_np), np.nanmax(all_sst_values_np)
    
    print(f"归一化统计参数计算完成：Min_SST = {min_sst:.4f}, Max_SST = {max_sst:.4f}")
    return float(min_sst), float(max_sst)


def normalize_sst(sst_array: xr.DataArray, min_val: float, max_val: float) -> xr.DataArray:
    """
    将SST数据归一化到 [-1, 1] 范围。
    """
    if max_val == min_val: # 避免除以零
        raise ValueError("SST数据的最小值和最大值相同。归一化结果将全部为0。")

    normalized_values = 2 * (sst_array.values - min_val) / (max_val - min_val) - 1
    normalized_values = np.clip(normalized_values, -1.0, 1.0)
    return xr.DataArray(normalized_values.astype(np.float32), coords=sst_array.coords, dims=sst_array.dims)

def create_land_sea_mask(raw_sst_array: xr.DataArray) -> torch.Tensor:
    """
    根据原始SST数据创建海洋陆地掩码。
    海洋为True (或1), 陆地为False (或0)。
    raw_sst_array: 未经插值的SST数据 (通常是裁剪后的)。
    """
    # NaN 值对应陆地，非NaN值对应海洋
    mask_np = ~np.isnan(raw_sst_array.values)
    return torch.from_numpy(mask_np.astype(bool))

def create_and_save_patches(
    daily_sst_normalized_array: xr.DataArray, 
    date_str: str, 
    patch_height:int,
    patch_width:int,
    stride: int, 
    output_base_dir: str
):
    """
    从单日已归一化的SST图像中提取patches并保存。每日patches保存在一个文件中。包括原始信息和位置信息。
    """
    height,width = daily_sst_normalized_array.shape
    sst_np = daily_sst_normalized_array.values
    ensure_dir(output_base_dir)
    patches_one_day = [] # 存储单日的所有patches信息
    for r in range(0, height - patch_width + 1, stride):
        for c in range(0, width - patch_height + 1, stride):
            patch_np = sst_np[r:r + patch_width, c:c + patch_height]
            patch_data = {
                'sst_patch': torch.from_numpy(patch_np.astype(np.float32)),
                'coords':(r,c)
            }
            patches_one_day.append(patch_data)
    if patches_one_day:
        daily_patches_filename = f"{date_str}_patches.pt"
        filepath = os.path.join(output_base_dir, daily_patches_filename)
        torch.save(patches_one_day, filepath)
    return len(patches_one_day)  # 返回单日patches的数量

def run_preprocessing():
    """
    执行完整的预处理流程：
    1. 确定文件列表和日期范围。
    2. (如果不存在) 计算并保存归一化统计参数 (仅基于训练期数据)。
    3. (如果不存在) 创建并保存陆地海洋掩码。
    4. 逐日处理数据：加载、裁剪、插值、归一化。
    5. 调用 create_and_save_patches 将每日所有patches保存到单个汇总文件中。
    """
    ensure_dir(config.DATA_PROCESSED_PATH) # 确保预处理数据目录存在
    ensure_dir(config.PATCHES_PATH) # 确保patches目录存在
    ensure_dir(config.RESULTS_PATH) 
    ensure_dir(config.CHECKPOINT_PATH)

    # 1. 所有可用文件和对应日期
    start_date = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(config.DATA_END_DATE, "%Y-%m-%d")
    train_end_date = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d")

    all_files_map = {}
    current_date = start_date
    while current_date <= end_date:
        date_str_nodash = current_date.strftime("%Y%m%d")
        filepath = os.path.join(config.DATA_RAW_PATH, config.FILENAME_TEMPLATE.format(date_str=date_str_nodash))
        if os.path.exists(filepath):
            all_files_map[current_date] = filepath
        current_date += timedelta(days=1)
    
    train_files = [fp for dt, fp in all_files_map.items() if dt <= train_end_date]
    
    if not os.path.exists(config.NORMALIZATION_STATS_PATH):
        min_sst, max_sst = get_normalization_stats(train_files, (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH))
        torch.save({'min_sst': min_sst, 'max_sst': max_sst}, config.NORMALIZATION_STATS_PATH)
    else:
        stats = torch.load(config.NORMALIZATION_STATS_PATH)
        min_sst, max_sst = stats['min_sst'], stats['max_sst']
        print(f"已从文件加载归一化参数。min_sst: {min_sst}, max_sst: {max_sst}")

    if not os.path.exists(config.LAND_SEA_MASK_PATH):
        sample_sst_raw, _, _ = load_single_nc_file(next(iter(all_files_map.values())))
        sample_sst_cropped = crop_image(sample_sst_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
        torch.save(create_land_sea_mask(sample_sst_cropped),config.LAND_SEA_MASK_PATH)
        print("陆海掩码已创建并保存。")
    
    print("开始处理每日数据并生成每日汇总patches...")
    pattches_num = 0
    for dt, fp in tqdm(all_files_map.items(), desc="预处理每日数据"):
        date_str_iso = dt.strftime("%Y-%m-%d")
        if os.path.exists(os.path.join(config.PATCHES_PATH, f"{date_str_iso}_daily_patches.pt")):
            continue
        sst_raw, _, _ = load_single_nc_file(fp)
        sst_cropped = crop_image(sst_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
        sst_filled = fill_land(sst_cropped,method=None)
        sst_normalized = normalize_sst(sst_filled, min_sst, max_sst)
        pattches_num = create_and_save_patches(sst_normalized, date_str_iso, patch_height=config.PATCH_HEIGHT,patch_width=config.PATCH_WIDTH, stride=config.STRIDE, output_base_dir=config.PATCHES_PATH)
    print(f"已处理 {len(all_files_map)} 天数据，每天生成了 {pattches_num} 个patches。")
    pred_num_of_daily_patches = ((config.IMAGE_TARGET_HEIGHT - config.PATCH_HEIGHT)/config.STRIDE + 1) * ((config.IMAGE_TARGET_WIDTH - config.PATCH_WIDTH)/config.STRIDE +1)
    if pattches_num != pred_num_of_daily_patches:
        raise ValueError(f"选取的patch参数不能合适裁剪！理论裁剪pathes数：{pred_num_of_daily_patches}")
    print("数据预处理流程全部完成。")

# 执行预处理流程
if __name__ == "__main__":
    # 使用python -m src.preprocess 命令运行
    run_preprocessing()
