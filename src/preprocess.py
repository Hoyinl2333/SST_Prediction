# src/preprocess.py

import os
import glob
from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import torch
from scipy.interpolate import griddata
from tqdm import tqdm # 用于显示处理进度
import pandas as pd # 用于处理日期序列

# 从当前包（src）中导入配置
from src import config

def ensure_dir(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    # print(f"目录已确保存在: {directory_path}") # 调试信息

def load_single_nc_file(file_path):
    """
    加载单个NetCDF文件，并提取SST、经度和纬度数据。
    假设SST数据维度为 (lat, lon) 或 (lon, lat)。
    """
    try:
        ds = xr.open_dataset(file_path)
        sst_data = ds[config.SST_VARIABLE_NAME]
        lon = ds.get(config.LON_VARIABLE_NAME, ds.get('lon', None))
        lat = ds.get(config.LAT_VARIABLE_NAME, ds.get('lat', None))

        if lon is None or lat is None:
            raise ValueError("NetCDF 文件中未找到经度或纬度变量。")

        # 确保SST数据的维度顺序是 (latitude, longitude)
        if sst_data.dims[0] == lon.name and sst_data.dims[1] == lat.name:
            sst_data = sst_data.transpose(lat.name, lon.name)
        elif sst_data.dims[0] != lat.name or sst_data.dims[1] != lon.name:
            if len(sst_data.dims) == 2:
                if ds[lat.name].size == sst_data.shape[0] and ds[lon.name].size == sst_data.shape[1]:
                    sst_data = sst_data.rename({sst_data.dims[0]: lat.name, sst_data.dims[1]: lon.name})
                elif ds[lon.name].size == sst_data.shape[0] and ds[lat.name].size == sst_data.shape[1]:
                     sst_data = sst_data.rename({sst_data.dims[0]: lon.name, sst_data.dims[1]: lat.name})
                     sst_data = sst_data.transpose(lat.name, lon.name)
                else:
                    raise ValueError(
                        f"SST数据维度 {sst_data.dims} 与经纬度名称和大小不匹配。"
                    )
            else:
                raise ValueError(f"SST数据具有意外的维度数量: {len(sst_data.dims)}")

        if 'time' in sst_data.dims and len(sst_data.time) == 1:
            sst_data = sst_data.squeeze('time')

        return sst_data, lon, lat
    except Exception as e:
        print(f"加载或处理文件 {file_path} 时出错: {e}")
        raise

def crop_image(data_array: xr.DataArray, target_height: int, target_width: int) -> xr.DataArray:
    """
    裁剪 xarray.DataArray 到目标尺寸。
    假设输入 data_array 的维度是 (lat, lon)。
    """
    current_height, current_width = data_array.shape
    
    if current_height == 721 and target_height == 720:
         data_array = data_array[:-1, :] # 去掉最后一行以匹配720的高度
         current_height = data_array.shape[0] # 更新当前高度
    elif current_height == target_height and current_width == target_width:
        return data_array
    
    # 再次检查尺寸，如果仍不匹配目标（且不是上述721->720的情况），则尝试直接裁剪
    if current_height == target_height and current_width == target_width:
        return data_array
    elif current_height < target_height or current_width < target_width:
        raise ValueError(f"原始图像 ({current_height},{current_width}) 小于目标尺寸 ({target_height},{target_width})。无法裁剪。")
    
    # 执行普适性的左上角裁剪
    cropped_data = data_array[:target_height, :target_width]
    if cropped_data.shape != (target_height, target_width):
         raise ValueError(f"裁剪后的形状为 {cropped_data.shape}, 期望为 ({target_height},{target_width})。请检查原始数据维度和裁剪逻辑。")
    return cropped_data



def interpolate_land(sst_array: xr.DataArray, method: str = 'linear') -> xr.DataArray:
    """
    对SST数据中的NaN值（陆地）进行插值，采用 pandas DataFrame 的插值方法，然后用0填充。

    sst_array: 原始SST数据，xarray.DataArray，维度应为 (lat, lon)
    method: pandas.DataFrame.interpolate 使用的插值方法 ('linear', 'nearest', 'quadratic', 'cubic', etc.)
            注意：pandas的 'linear' 对于时间序列是一维的，对于二维数据，我们会分别按行和列做。
    返回: 插值后的SST数据 (NaNs 被填充为0)，xarray.DataArray
    """
    sst_values_original = sst_array.values.copy() # 获取numpy数组副本

    # 转换为pandas DataFrame
    df = pd.DataFrame(sst_values_original)

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

def get_normalization_stats(file_paths_for_stats, crop_dims, interpolation_method='linear'):
    """
    计算SST数据的全局最小值和最大值，用于归一化。
    仅使用指定文件列表（通常是训练集文件）的数据进行计算。
    file_paths_for_stats: 用于计算统计数据的文件路径列表。
    crop_dims: (target_height, target_width) 裁剪尺寸。
    interpolation_method: 陆地插值方法。
    """
    print(f"开始计算归一化统计参数，使用 {len(file_paths_for_stats)} 个文件...")
    
    all_sst_values = []
    
    for file_path in tqdm(file_paths_for_stats, desc="处理文件以计算统计参数"):
        try:
            sst_array, _, _ = load_single_nc_file(file_path)
            sst_cropped = crop_image(sst_array, crop_dims[0], crop_dims[1])
            sst_interpolated = interpolate_land(sst_cropped, method=interpolation_method)
            all_sst_values.append(sst_interpolated.values.flatten()) #展平并收集所有值
        except Exception as e:
            print(f"处理文件 {file_path} 时跳过（统计计算阶段）: {e}")
            continue
            
    if not all_sst_values:
        raise ValueError("未能从任何文件中收集到SST数据以计算归一化统计参数。")

    all_sst_values_np = np.concatenate(all_sst_values)
    min_sst = np.nanmin(all_sst_values_np) # 使用nanmin以防插值后仍意外存在nan
    max_sst = np.nanmax(all_sst_values_np)
    
    print(f"归一化统计参数计算完成：Min_SST = {min_sst}, Max_SST = {max_sst}")
    return float(min_sst), float(max_sst)

def normalize_sst(sst_array: xr.DataArray, min_val: float, max_val: float) -> xr.DataArray:
    """
    将SST数据归一化到 [-1, 1] 范围。
    """
    if max_val == min_val: # 避免除以零
        raise ValueError("SST数据的最小值和最大值相同。归一化结果将全部为0。")

    normalized_values = 2 * (sst_array.values - min_val) / (max_val - min_val) - 1
    # TODO:将超出[-1,1]范围的值裁剪到该范围，以防测试数据超出训练数据的min/max范围
    # normalized_values = np.clip(normalized_values, -1.0, 1.0) 
    return xr.DataArray(normalized_values.astype(np.float32), coords=sst_array.coords, dims=sst_array.dims, name=sst_array.name + "_normalized")

def create_land_sea_mask(raw_sst_array: xr.DataArray) -> torch.Tensor:
    """
    根据原始SST数据创建海洋陆地掩码。
    海洋为True (或1), 陆地为False (或0)。
    raw_sst_array: 未经插值的SST数据 (通常是裁剪后的)。
    """
    # NaN 值对应陆地，非NaN值对应海洋
    mask_np = ~np.isnan(raw_sst_array.values)
    return torch.from_numpy(mask_np.astype(np.bool_)) # 保存为布尔张量

def create_and_save_patches(
    daily_sst_normalized_array: xr.DataArray, 
    date_str: str, 
    patch_size: int, 
    stride: int, 
    output_base_dir: str
):
    """
    从单日已归一化的SST图像中提取patches并保存。
    每个patch保存为一个 .pt 文件，包含patch数据、日期和原始坐标。
    daily_sst_normalized_array: 维度为 (H, W) 的已归一化SST数据。
    date_str: 'YYYY-MM-DD' 格式的日期字符串。
    patch_size: patch的边长。
    stride: 提取patch的步长。
    output_base_dir: patches的根输出目录 (例如 config.PATCHES_PATH)。
    """
    height, width = daily_sst_normalized_array.shape
    sst_np = daily_sst_normalized_array.values # 转换为numpy数组

    patch_idx = 0
    for r in range(0, height - patch_size + 1, stride):
        for c in range(0, width - patch_size + 1, stride):
            patch = sst_np[r:r+patch_size, c:c+patch_size]
            
            patch_data = {
                'sst_patch': torch.from_numpy(patch.astype(np.float32)), # 保存为 (patch_size, patch_size)
                'date': date_str, # YYYY-MM-DD
                'coords': (r, c)  # (row_start, col_start) 对应在 (H,W) 全局图像中的左上角坐标
            }
                 
            # 文件名包含日期和patch索引，确保唯一性
            patch_filename = f"{date_str}_patch_r{r:03d}_c{c:03d}.pt"
            filepath = os.path.join(output_base_dir, patch_filename)
            
            torch.save(patch_data, filepath)
            patch_idx += 1
    print(f"为日期 {date_str} 创建并保存了 {patch_idx} 个patches。")

def run_preprocessing():
    """
    执行完整的预处理流程。
    """
    print("开始执行数据预处理流程...")
    ensure_dir(config.DATA_PROCESSED_PATH)
    ensure_dir(config.PATCHES_PATH)
    ensure_dir(config.RESULTS_PATH)
    ensure_dir(config.CHECKPOINT_PATH)
    ensure_dir(config.FIGURES_PATH)
    ensure_dir(config.PREDICTIONS_PATH)

    # 1. 生成所有可用文件的列表和对应的日期对象
    start_date_dt = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d")
    end_date_dt = datetime.strptime(config.DATA_END_DATE, "%Y-%m-%d")
    train_period_end_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d") # 新增

    all_file_paths = []
    date_objects_for_processing = [] 

    current_dt = start_date_dt
    while current_dt <= end_date_dt:
        date_str_nodash = current_dt.strftime("%Y%m%d")
        filename = config.FILENAME_TEMPLATE.format(date_str=date_str_nodash)
        file_path = os.path.join(config.DATA_RAW_PATH, filename)
        if os.path.exists(file_path):
            all_file_paths.append(file_path)
            date_objects_for_processing.append(current_dt)
        else:
            print(f"警告: 文件 {file_path} 未找到，将跳过此日期 {current_dt.strftime('%Y-%m-%d')}。")
        current_dt += timedelta(days=1)

    if not all_file_paths:
        print("错误: 在指定日期范围内 ({config.DATA_START_DATE} 至 {config.DATA_END_DATE}) 未找到任何数据文件。请检查config.py中的路径和日期设置。")
        return

    # 2. 筛选出训练期的文件用于计算归一化统计参数
    file_paths_for_norm_stats = []
    for i, dt_obj in enumerate(date_objects_for_processing):
        if dt_obj <= train_period_end_dt:
            file_paths_for_norm_stats.append(all_file_paths[i])
            
    if not file_paths_for_norm_stats:
        raise ValueError(f"在指定的训练期 (截止到 {config.TRAIN_PERIOD_END_DATE}) 内未找到任何数据文件用于计算归一化统计参数。")
    
    print(f"将使用截止到 {config.TRAIN_PERIOD_END_DATE} 的 {len(file_paths_for_norm_stats)} 个文件计算归一化统计。")

    # 3. 计算并保存归一化统计参数 (如果尚未存在)
    crop_dimensions = (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
    if not os.path.exists(config.NORMALIZATION_STATS_PATH):
        min_sst, max_sst = get_normalization_stats(file_paths_for_norm_stats, crop_dimensions) # 使用筛选后的文件列表
        torch.save({'min_sst': min_sst, 'max_sst': max_sst}, config.NORMALIZATION_STATS_PATH)
        print(f"归一化参数已计算并保存到: {config.NORMALIZATION_STATS_PATH}")
    else:
        stats = torch.load(config.NORMALIZATION_STATS_PATH)
        min_sst, max_sst = stats['min_sst'], stats['max_sst']
        print(f"已从文件加载归一化参数: Min_SST={min_sst}, Max_SST={max_sst} (这些参数应基于训练数据计算得到)")

    # 4. 创建并保存陆地海洋掩码 (如果尚未存在) - 使用整个数据集中的第一个可用文件
    if not os.path.exists(config.LAND_SEA_MASK_PATH):
        if not all_file_paths:
             print("错误：没有文件可用于创建陆地海洋掩码。")
             return
        print(f"使用文件 {all_file_paths[0]} 创建陆地海洋掩码...")
        sample_sst_raw, _, _ = load_single_nc_file(all_file_paths[0])
        sample_sst_cropped_raw = crop_image(sample_sst_raw, crop_dimensions[0], crop_dimensions[1])
        land_sea_mask = create_land_sea_mask(sample_sst_cropped_raw)
        torch.save(land_sea_mask, config.LAND_SEA_MASK_PATH)
        print(f"陆地海洋掩码已保存到: {config.LAND_SEA_MASK_PATH}")
    else:
        print(f"陆地海洋掩码文件已存在: {config.LAND_SEA_MASK_PATH}")
        
    # 5. 处理每日数据（覆盖整个 DATA_START_DATE 到 DATA_END_DATE 范围）：
    #    加载、裁剪、插值、使用(基于训练集计算的)归一化参数进行归一化、分块并保存
    print(f"\n开始处理每日数据并生成patches (覆盖从 {config.DATA_START_DATE} 至 {config.DATA_END_DATE} 的所有数据)...")
    for i, file_path in enumerate(tqdm(all_file_paths, desc="处理每日数据并生成Patches")):
        current_dt_obj = date_objects_for_processing[i]
        current_date_str_iso = current_dt_obj.strftime("%Y-%m-%d")
            
        try:
            sst_array, _, _ = load_single_nc_file(file_path)
            sst_cropped = crop_image(sst_array, crop_dimensions[0], crop_dimensions[1])
            sst_interpolated = interpolate_land(sst_cropped)
            sst_normalized = normalize_sst(sst_interpolated, min_sst, max_sst) # 使用基于训练集计算的min_max
            
            create_and_save_patches(
                sst_normalized, 
                current_date_str_iso, 
                config.PATCH_SIZE, 
                config.STRIDE, 
                config.PATCHES_PATH
            )
        except Exception as e:
            print(f"处理文件 {file_path} (日期 {current_date_str_iso}) 时发生错误，跳过此文件: {e}")
            continue
            
    print("数据预处理流程全部完成。所有日期的数据都已处理并生成patches。")
    print(f"归一化参数是基于 {config.DATA_START_DATE} 至 {config.TRAIN_PERIOD_END_DATE} 的数据计算的。")

if __name__ == "__main__":
    run_preprocessing()
