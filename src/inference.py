import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from diffusers import DDPMScheduler
from . import config
from .utils import ensure_dir,load_checkpoint,denormalize_sst, reconstruct_image_from_patches, calculate_metrics,save_img_comparison
from .models import get_diffusion_model
from .features import get_time_features, get_spatial_features
from .preprocess import load_single_nc_file, crop_image


def run_inference(checkpoint_filename = "model_fianl.pt",inference_start_date_str=None):
    print("开始运行推理流程...")
    
    # 1.基本设置
    ensure_dir(config.RESULTS_PATH) 

    land_sea_mask = torch.load(config.LAND_SEA_MASK_PATH)
    land_sea_mask_np = land_sea_mask.numpy()  # 转换为numpy数组以便后续处理
    print(f"陆海掩码已加载，形状: {land_sea_mask.shape}。")

    device = torch.device(config.DEVICE) 
    stats = torch.load(config.NORMALIZATION_STATS_PATH)
    min_sst, max_sst = stats['min_sst'], stats['max_sst']
    print(f"归一化参数： Min_SST = {min_sst:.4f}, Max_SST = {max_sst:.4f}")
    
    model = get_diffusion_model()
    model.to(device)  # 确保模型在正确的设备上
    model = load_checkpoint(os.path.join(config.CHECKPOINT_PATH, checkpoint_filename),model,device)  # 加载模型checkpoint
    model.eval()  
    print(f"模型已加载并设置为评估模式。使用设备: {device}")
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.DDPM_NUM_TRAIN_TIMESTEPS, 
        beta_schedule=config.DDPM_BETA_SCHEDULE
    )
    noise_scheduler.set_timesteps(config.DDPM_NUM_INFERENCE_STEPS)  # 设置推理步数
    print(f"DDPM噪声调度器已设置为 {config.DDPM_NUM_INFERENCE_STEPS} 步推理。")
    
    if inference_start_date_str is None:
        inference_start_date_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d") + timedelta(days=1)
    else:
        inference_start_date_dt = datetime.strptime(inference_start_date_str, "%Y-%m-%d")
    print(f"将从日期 {inference_start_date_dt.strftime('%Y-%m-%d')} 开始进行 {config.AUTOREGRESSIVE_PREDICT_DAYS} 天的自回归预测。")
    

    # 2.加载历史数据
    print("加载初始历史数据...")
    
    current_history_all_coords = defaultdict(list)
    history_start_dt = inference_start_date_dt - timedelta(days=config.HISTORY_DAYS)

    for day_offset in range(config.HISTORY_DAYS):
        dt = history_start_dt + timedelta(days=day_offset)
        file_path = os.path.join(config.PATCHES_PATH, f"{dt.strftime('%Y-%m-%d')}_daily_patches.pt")
        if os.path.exists(file_path):
            for patch_data in torch.load(file_path, map_location=device):
                current_history_all_coords[tuple(patch_data['coords'])].append(patch_data['sst_patch'])
    
    valid_coords = {c for c, h in current_history_all_coords.items() if len(h) == config.HISTORY_DAYS}
    if not valid_coords: 
        raise ValueError("没有可用于预测的完整初始历史数据。")
    
    ref_year = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d").year
    evalution_metrics = defaultdict(list)  # 用于存储评估指标

    # 3. 推理循环
    with torch.no_grad():
        for day in range(config.AUTOREGRESSIVE_PREDICT_DAYS):
            target_pred_dt = inference_start_date_dt + timedelta(days=day)
            target_date_str = target_pred_dt.strftime("%Y-%m-%d")
            lead_time_key = f"T+{day+1}"  # 用于存储评估结果的键
            print(f"\n正在预测 {target_date_str} 的数据...")

            predicted_patches = {}
            for coords in tqdm(list(valid_coords), desc=f"预测Patches:{target_date_str}"):
                # 准备历史数据
                history_sequence_list = current_history_all_coords[coords]
                history_tensor = torch.stack(history_sequence_list).unsqueeze(0).to(device)  # (1, HISTORY_DAYS, H, W)

                # time and spatial features
                time_feat = get_time_features(target_date_str, reference_year=ref_year).unsqueeze(0).to(device)
                spatial_feat = get_spatial_features(coords, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH).unsqueeze(0).to(device)

                noisy_patch_sample = torch.randn(1,config.UNET_OUT_CHANNELS,config.PATCHES_WIDTH,config.PATCHES_WIDTH, device=device)  # (1, C, H, W)

                for t in noise_scheduler.timesteps:
                    pred_noise = model(history_tensor, noisy_sample, torch.tensor([t], device=device).long(), time_feat, spatial_feat)
                    noisy_sample = noise_scheduler.step(pred_noise, t, noisy_sample).prev_sample
                

                pred_path_normalized = noisy_patch_sample.squeeze(0).squeeze(0) 
                predicted_patches[coords] = pred_path_normalized

                # 自回归使用新的预测结果更新历史
                if day < config.AUTOREGRESSIVE_PREDICT_DAYS - 1:
                    current_history_all_coords[coords] = history_sequence_list[1:] + [pred_path_normalized.detach()]
            
            # 重构预测图像
            print(f"重建图像...")
            full_pred_img_norm = reconstruct_image_from_patches(
                predicted_patches,
                img_dims=(config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
                patch_height=config.PATCH_HEIGHT,
                patch_width=config.PATCHES_WIDTH
            )
            full_pred_img_denorm = denormalize_sst(full_pred_img_norm, min_sst, max_sst).numpy()
            
            # 评估
            target_np  = None
            target_img_path = os.path.join(config.DATA_RAW_PATH, config.FILENAME_TEMPLATE.format(date_str=target_pred_dt.strftime("%Y-%m-%d")))
            if os.path.exists(target_img_path):
                target_sst_raw,_,_ = load_single_nc_file(target_img_path)
                target_cropped = crop_image(target_sst_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
                target_np  = target_cropped.values

                mse,rmse = calculate_metrics(full_pred_img_denorm, target_np, land_sea_mask_np)
                print(f"  评估结果 ({lead_time_key}, {target_date_str}): RMSE = {rmse:.4f}, MSE = {mse:.4f}")
                evalution_metrics[lead_time_key].append({'date': target_date_str, 'rmse': rmse, 'mse': mse})
            else:
                print(f"警告: 目标图像文件 {target_img_path} 不存在，无法计算评估指标。")

            save_img_comparison(
                full_pred_img_denorm, 
                target_np, 
                land_sea_mask_np, 
                os.path.join(config.PREDICTIONS_PATH, f"prediction_{target_date_str}_{lead_time_key}.png"), 
                f"{target_date_str} ({lead_time_key}) "
            )

    # 5. 打印最终的评估指标总结
    print("="*50)
    print("--- 最终评估指标总结 (针对此次推理序列) ---")
    print("="*50)
    for lead_time, metrics_list in evalution_metrics.items():
        if not metrics_list:
            print(f"{lead_time}: 无评估数据")
            continue
        avg_rmse = np.nanmean([m['rmse'] for m in metrics_list])
        avg_mse = np.nanmean([m['mse'] for m in metrics_list])
        print(f"{lead_time} - 平均 RMSE: {avg_rmse:.4f}, 平均 MSE: {avg_mse:.4f}")

if __name__ == "__main__":
    # python -m src.inference
    run_inference(checkpoint_filename="model_final.pt", inference_start_date_str=None)
    # 可以根据需要修改checkpoint_filename和inference_start_date_str参数
    # inference_start_date_str可以设置为特定日期字符串，如"2023-10-01"，如果不设置则默认从训练结束后的下一天开始推理。