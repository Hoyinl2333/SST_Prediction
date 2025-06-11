import os
import torch
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
from diffusers import DDPMScheduler
from . import config
from .utils import *
from .models import get_diffusion_model
from .features import get_time_features, get_spatial_features
from .preprocess import crop_image


def _run_inference_logic(model,noise_scheduler,current_history_all_coords,valid_coords,
                         inference_start_date_dt,min_sst,max_sst,ref_year,land_sea_mask_np,
                         lat_coords_np,lon_coords_np,inference_paths):
    """推理主要逻辑函数"""
    evalution_metrics = defaultdict(list)  # 用于存储评估指标
    with torch.no_grad():
        for i in range(config.AUTOREGRESSIVE_PREDICT_DAYS):
            target_dt = inference_start_date_dt + timedelta(days=i)
            target_date_str = target_dt.strftime("%Y-%m-%d")
            lead_time_key = f"T+{i+1}"  # 用于存储评估结果的键
            print(f"\n正在预测 ({config.PREDICTION_TARGET}) {target_date_str}({lead_time_key}) 的数据...")

            predicted_patches ={}
            for coords in tqdm(list(valid_coords), desc=f"预测Patches:{target_date_str}"):
                history_tensor = torch.stack(current_history_all_coords[coords]).unsqueeze(0).to(config.DEVICE)  # (1, HISTORY_DAYS, H, W)
                time_feat = get_time_features(target_date_str, reference_year=ref_year).unsqueeze(0).to(config.DEVICE)
                spatial_feat = get_spatial_features(coords, (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH), config.PATCH_HEIGHT, config.PATCH_WIDTH).unsqueeze(0).to(config.DEVICE)
                
                noisy_sample = torch.randn(1, config.UNET_OUT_CHANNELS, config.PATCH_HEIGHT, config.PATCH_WIDTH, device=config.DEVICE)  # (1, C, H, W)
                for t in noise_scheduler.timesteps:
                    pred_noise = model(history_tensor, noisy_sample, torch.tensor([t], device=config.DEVICE).long(), time_feat, spatial_feat)
                    noisy_sample = noise_scheduler.step(pred_noise, t, noisy_sample).prev_sample
                predicted_patch_norm  = noisy_sample.squeeze(0).squeeze(0)  # (H, W)
            
                # 根据不同模式获得目标的绝对值
                if config.PREDICTION_TARGET == 'absolute':
                    predicted_patch_absolute_norm = predicted_patch_norm
                elif config.PREDICTION_TARGET == 'difference':
                    predicted_patch_absolute_norm = current_history_all_coords[coords][-1] + predicted_patch_norm
                else:
                    raise ValueError(f"不存在{config.PREDICTION_TARGET}! ")
            
                predicted_patches[coords] = predicted_patch_absolute_norm.cpu()
                # 自回归使用新的预测结果更新历史
                if i < config.AUTOREGRESSIVE_PREDICT_DAYS - 1:
                    current_history_all_coords[coords] =[predicted_patch_norm.detach()] + current_history_all_coords[coords][1:]  # 更新历史数据，保留最新的预测结果
            # 重建、评估、可视化
            _process_daily_results(predicted_patches,target_dt,lead_time_key,min_sst,max_sst,evalution_metrics,land_sea_mask_np,lat_coords_np,lon_coords_np,inference_paths)

    _print_summary_metrics(evalution_metrics)


def  _process_daily_results(predicted_patches_cpu, target_dt, lead_time_key, min_sst, max_sst, evaluation_metrics,land_sea_mask_np,lat_coords_np,lon_coords_np,inference_paths):
    """重建、反归一化、评估、可视化"""
    print(f"  重建图像...")
    full_pred_norm = reconstruct_image_from_patches(
        predicted_patches_cpu,
        (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
        config.PATCH_HEIGHT, config.PATCH_WIDTH
    )
    full_pred_denorm = denormalize_sst(full_pred_norm, min_sst, max_sst).numpy()
    pred_masked = np.where(land_sea_mask_np,full_pred_denorm,np.nan)
    output_image_filepath = os.path.join(inference_paths['images'], f"prediction_{target_dt.strftime('%Y-%m-%d')}_{lead_time_key}.png")
    save_img(pred_masked, None ,output_image_filepath, f"{target_dt.strftime('%Y-%m-%d')} ({lead_time_key}) ")

    # 加载真实数据进行评估
    target_np = None
    target_filepath = os.path.join(config.DATA_RAW_PATH, config.FILENAME_TEMPLATE.format(date_str=target_dt.strftime("%Y%m%d")))
    if os.path.exists(target_filepath):
        try:
            gt_raw, _, _ = load_single_nc_file(target_filepath)
            gt_cropped = crop_image(gt_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
            target_np = gt_cropped.values

            mse,rmse = calculate_metrics(pred_masked, target_np,land_sea_mask_np)
            print(f"  评估结果 ({lead_time_key}): RMSE = {rmse:.4f}, MSE = {mse:.4f}")
            evaluation_metrics[lead_time_key].append({'date': target_dt.strftime('%Y-%m-%d'), 'rmse': rmse, 'mse': mse})
        except Exception as e:
            raise ValueError(f"  加载真实数据或评估时出错: {e}")
    else:
        print(f"  警告: 真实数据文件 {target_filepath} 未找到，无法评估。")
    
    # 保存可视化与原文件
    output_comparsion_image_filepath = os.path.join(inference_paths['images'], f"comparison_prediction_{target_dt.strftime('%Y-%m-%d')}_{lead_time_key}.png")
    save_img(pred_masked, target_np,output_comparsion_image_filepath,f"{target_dt.strftime('%Y-%m-%d')} ({lead_time_key}) ")
    nc_save_path = os.path.join(inference_paths['data'],f"pred_sst_{target_dt.strftime('%Y-%m-%d')}.nc")
    save_as_netcdf(pred_masked,nc_save_path,lat_coords_np,lon_coords_np,
                   lat_name=config.LAT_VARIABLE_NAME,lon_name=config.LON_VARIABLE_NAME,var_name=config.SST_VARIABLE_NAME)

def _print_summary_metrics(evaluation_metrics):
    """打印评估结果mse"""
    print("\n" + "="*50)
    print("--- 最终评估指标总结 (针对此次推理序列) ---")
    print("="*50)
    for lead_time, metrics_list in evaluation_metrics.items():
        if metrics_list:
            avg_mse = np.nanmean([m['mse'] for m in metrics_list])
            print(f"  {lead_time}: 平均 MSE = {avg_mse:.4f}")
        else:
            print(f"  {lead_time}: 无评估数据")


def run_inference(checkpoint_relative_path):
    """
    总推理入口

    Args:
        checkpoint_relative_path (str): 从 'checkpoints/' 目录开始的模型文件相对路径。
                                         例如: 'run_20250611_11/model_final.pt'
    """
    # 结果保存目录
    inference_paths = setup_inference_directory(checkpoint_relative_path)

    # 基本配置
    device = config.DEVICE
    stats = torch.load(config.NORMALIZATION_STATS_PATH)
    min_sst,max_sst = stats['min_sst'], stats['max_sst']

    model = get_diffusion_model()
    checkpoint_full_path = os.path.join(config.CHECKPOINT_PATH, checkpoint_relative_path)
    model = load_checkpoint(os.path.join(config.CHECKPOINT_PATH,checkpoint_full_path),model, device) 
   
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.DDPM_NUM_TRAIN_TIMESTEPS, beta_schedule=config.DDPM_BETA_SCHEDULE)
    noise_scheduler.set_timesteps(config.DDPM_NUM_INFERENCE_STEPS) 

    # 历史数据加载
    inference_start_date_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE,"%Y-%m-%d") + timedelta(days=1)
    print(f"将从日期 {inference_start_date_dt.strftime('%Y-%m-%d')} 开始进行 {config.AUTOREGRESSIVE_PREDICT_DAYS} 天的自回归预测。")
    
    history_start_dt = inference_start_date_dt - timedelta(days=config.HISTORY_DAYS)
    current_history_all_coords = defaultdict(list)

    date_range = pd.date_range(history_start_dt,inference_start_date_dt-timedelta(days=1))
    for dt in tqdm(date_range,desc="加载初始历史数据"):
        file_path = os.path.join(config.PATCHES_PATH,f"{dt.strftime('%Y-%m-%d')}_patches.pt")
        if os.path.exists(file_path):
            for patch in torch.load(file_path,map_location=device):
                current_history_all_coords[tuple(patch['coords'])].append(patch['sst_patch'])
    
    valid_coords = {c for c,h in current_history_all_coords.items() if len(h) == config.HISTORY_DAYS}
    if not valid_coords:
        raise ValueError("没有可用于预测的完整初始历史数据。")

    land_sea_mask_np = torch.load(config.LAND_SEA_MASK_PATH).numpy()
    ref_year = datetime.strptime(config.DATA_START_DATE,"%Y-%m-%d").year

    # 从样本文件中加载地理坐标信息
    print("加载地理坐标信息用于保存.nc文件...")
    lat_coords_np, lon_coords_np = None, None

    any_raw_file = glob.glob(os.path.join(config.DATA_RAW_PATH, "*.nc"))[0]
    sample_sst_raw, _, _ = load_single_nc_file(any_raw_file)
    sample_sst_cropped = crop_image(sample_sst_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
        
    # 从裁剪后的 xarray 对象中提取出坐标的Numpy数组
    lat_coords_np = sample_sst_cropped[config.LAT_VARIABLE_NAME].values
    lon_coords_np = sample_sst_cropped[config.LON_VARIABLE_NAME].values

    # 调用统一的核心逻辑函数
    _run_inference_logic(model, noise_scheduler, current_history_all_coords, valid_coords, 
                         inference_start_date_dt, min_sst, max_sst, ref_year, land_sea_mask_np, 
                         lat_coords_np,lon_coords_np,inference_paths)

