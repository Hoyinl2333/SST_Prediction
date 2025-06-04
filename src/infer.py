# src/inference.py

import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error # 用于计算MSE
import glob
from diffusers import DDPMScheduler

# 从当前包（src）中导入
from . import config
from .models import get_diffusion_model # DiffusionUNetWithMLP
from .features import get_time_features, get_spatial_features
# 从 preprocess.py 中导入原始数据加载和裁剪函数，用于加载真实数据进行比较
from .preprocess import load_single_nc_file as load_raw_sst_data 
from .preprocess import crop_image as crop_raw_sst_data

def ensure_dir(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_checkpoint(filepath, model, device):
    """加载模型checkpoint到指定设备"""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型权重已从 {filepath} 加载。Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
        return model
    except FileNotFoundError:
        print(f"错误: Checkpoint文件 {filepath} 未找到。请先运行训练。")
        raise
    except Exception as e:
        print(f"加载checkpoint {filepath} 失败: {e}")
        raise

def denormalize_sst(normalized_sst_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    将归一化到 [-1, 1] 的SST数据反归一化回原始范围。
    """
    if max_val == min_val:
        return torch.full_like(normalized_sst_tensor, min_val)
    original_sst = (normalized_sst_tensor + 1.0) * (max_val - min_val) / 2.0 + min_val
    return original_sst

def reconstruct_image_from_patches(
    patches_dict: dict, 
    image_dims: tuple,
    patch_size: int,
    stride: int
) -> torch.Tensor:
    """
    从patches字典中重建完整图像，重叠区域取平均。
    patches_dict: key=(r,c) tuple, value=torch.Tensor (patch_size, patch_size) [CPU tensor]
    image_dims: (H, W)
    返回: 重建的图像 (H,W) [CPU tensor]
    """
    H, W = image_dims
    reconstructed_image = torch.zeros((H, W), dtype=torch.float32)
    count_map = torch.zeros((H, W), dtype=torch.float32)

    for (r, c), patch_tensor in patches_dict.items():
        reconstructed_image[r:r+patch_size, c:c+patch_size] += patch_tensor
        count_map[r:r+patch_size, c:c+patch_size] += 1
    
    reconstructed_image = torch.where(
        count_map > 0, 
        reconstructed_image / count_map, 
        0.0 
    ) 
    return reconstructed_image

def save_image_comparison(
    predicted_np: np.ndarray, 
    ground_truth_np: np.ndarray, 
    land_sea_mask_np: np.ndarray, 
    filepath: str,
    title_prefix: str = ""
):
    """保存预测图像、真实图像（如果提供）和差异图的可视化比较。"""
    ensure_dir(os.path.dirname(filepath))
    
    predicted_masked = np.where(land_sea_mask_np, predicted_np, np.nan)
    
    num_plots = 1
    if ground_truth_np is not None:
        num_plots = 3
        ground_truth_masked = np.where(land_sea_mask_np, ground_truth_np, np.nan)
        diff_masked = np.where(land_sea_mask_np, predicted_np - ground_truth_np, np.nan)
        
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5), squeeze=False) # squeeze=False确保axes总是2D
    axes = axes.flatten() # 将axes展平为1D，方便索引

    im0 = axes[0].imshow(predicted_masked, cmap='coolwarm', origin='upper', vmin=config.SST_PLOT_VMIN, vmax=config.SST_PLOT_VMAX) # 使用vmin, vmax
    axes[0].set_title(f"{title_prefix}预测SST")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if ground_truth_np is not None:
        im1 = axes[1].imshow(ground_truth_masked, cmap='coolwarm', origin='upper', vmin=config.SST_PLOT_VMIN, vmax=config.SST_PLOT_VMAX)
        axes[1].set_title(f"{title_prefix}真实SST")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 计算差异图的对称色条范围
        abs_max_diff = np.nanmax(np.abs(diff_masked)) if not np.all(np.isnan(diff_masked)) else 1.0
        im2 = axes[2].imshow(diff_masked, cmap='RdBu_r', origin='upper', vmin=-abs_max_diff, vmax=abs_max_diff)
        axes[2].set_title(f"{title_prefix}差异 (预测 - 真实)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)
    print(f"图像比较已保存至: {filepath}")


def calculate_metrics(predicted_np: np.ndarray, ground_truth_np: np.ndarray, land_sea_mask_np: np.ndarray):
    """在海洋区域计算预测值与真实值之间的RMSE和MSE。"""
    if predicted_np.shape != ground_truth_np.shape or predicted_np.shape != land_sea_mask_np.shape:
        raise ValueError("输入数组的形状必须一致。")

    ocean_pixels_pred = predicted_np[land_sea_mask_np]
    ocean_pixels_gt = ground_truth_np[land_sea_mask_np]

    if ocean_pixels_pred.size == 0 or ocean_pixels_gt.size == 0:
        print("警告: 海洋像素为空，无法计算指标。")
        return np.nan, np.nan

    mse = mean_squared_error(ocean_pixels_gt, ocean_pixels_pred)
    rmse = np.sqrt(mse)
    return rmse, mse

def run_inference(
    checkpoint_filename: str = "model_final.pt", 
    inference_start_date_str: str = None 
):
    print("开始推理与评估流程...")
    device = torch.device(config.DEVICE)

    try:
        stats = torch.load(config.NORMALIZATION_STATS_PATH, map_location='cpu')
        min_sst, max_sst = stats['min_sst'], stats['max_sst']
        land_sea_mask_tensor = torch.load(config.LAND_SEA_MASK_PATH, map_location='cpu')
        land_sea_mask_np = land_sea_mask_tensor.numpy()
    except FileNotFoundError as e:
        print(f"错误: 必需的预处理文件未找到 ({e})。请运行preprocess.py。")
        return
    
    print(f"加载归一化参数: Min={min_sst:.2f}, Max={max_sst:.2f}")

    model = get_diffusion_model()
    model = load_checkpoint(os.path.join(config.CHECKPOINT_PATH, checkpoint_filename), model, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.DDPM_NUM_TRAIN_TIMESTEPS,
        beta_schedule=config.DDPM_BETA_SCHEDULE
    )
    noise_scheduler.set_timesteps(config.DDPM_NUM_INFERENCE_STEPS)
    print(f"DDPM采样器已准备就绪，采样步数: {config.DDPM_NUM_INFERENCE_STEPS}")

    if inference_start_date_str is None:
        train_end_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d")
        inference_start_date_dt = train_end_dt + timedelta(days=1)
        inference_start_date_str = inference_start_date_dt.strftime("%Y-%m-%d")
    else:
        inference_start_date_dt = datetime.strptime(inference_start_date_str, "%Y-%m-%d")
    
    print(f"将从日期 {inference_start_date_str} 开始进行 {config.AUTOREGRESSIVE_PREDICT_DAYS} 天的自回归预测。")

    current_history_all_coords = defaultdict(list)
    all_spatial_coords = set()
    history_start_dt = inference_start_date_dt - timedelta(days=config.HISTORY_DAYS)
    
    print("加载初始历史数据...")
    for day_idx in range(config.HISTORY_DAYS):
        current_scan_dt = history_start_dt + timedelta(days=day_idx)
        date_str_scan = current_scan_dt.strftime("%Y-%m-%d")
        patch_files_today = glob.glob(os.path.join(config.PATCHES_PATH, f"{date_str_scan}_patch_*.pt"))
        if not patch_files_today:
            print(f"警告: 日期 {date_str_scan} 的patch文件缺失。") # 仅警告，后续会检查完整性
            continue
        for patch_file in patch_files_today:
            basename = os.path.basename(patch_file)
            try:
                parts = basename.replace(".pt", "").split('_patch_')
                coord_str_parts = parts[1].split('_')
                r, c = int(coord_str_parts[0][1:]), int(coord_str_parts[1][1:])
                coords = (r,c)
                all_spatial_coords.add(coords)
                patch_content = torch.load(patch_file, map_location=device)
                current_history_all_coords[coords].append(patch_content['sst_patch'])
            except Exception as e:
                print(f"解析或加载patch {basename} 失败: {e}")
    
    valid_initial_coords = set()
    for coords in list(all_spatial_coords): # 使用list进行安全迭代
        if len(current_history_all_coords[coords]) == config.HISTORY_DAYS:
            valid_initial_coords.add(coords)
        else:
            print(f"警告: 坐标 {coords} 初始历史不完整 (仅 {len(current_history_all_coords[coords])} 天)，跳过此坐标。")
            del current_history_all_coords[coords]
            
    if not valid_initial_coords:
        print("错误: 没有坐标具有完整的初始历史数据。推理中止。")
        return
    all_spatial_coords = valid_initial_coords

    ref_year = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d").year
    evaluation_metrics = {f"Tplus{i+1}": [] for i in range(config.AUTOREGRESSIVE_PREDICT_DAYS)}

    with torch.no_grad():
        for day_offset in range(config.AUTOREGRESSIVE_PREDICT_DAYS):
            target_predict_dt = inference_start_date_dt + timedelta(days=day_offset)
            target_predict_date_str = target_predict_dt.strftime("%Y-%m-%d")
            lead_time_key = f"Tplus{day_offset+1}"
            print(f"\n正在预测与评估日期: {target_predict_date_str} ({lead_time_key})")
            predicted_patches_today = {}

            for coords in tqdm(list(all_spatial_coords), desc=f"预测Patches @ {target_predict_date_str}"):
                history_sequence_list = current_history_all_coords[coords]
                history_tensor = torch.stack(history_sequence_list).unsqueeze(0).to(device)
                
                time_feat = get_time_features(target_predict_date_str, reference_year=ref_year).unsqueeze(0).to(device)
                spatial_feat = get_spatial_features(
                    coords, 
                    (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH), 
                    config.PATCH_SIZE
                ).unsqueeze(0).to(device)

                noisy_patch_sample = torch.randn(
                    1, config.UNET_OUT_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE, device=device
                )
                
                for t_step in noise_scheduler.timesteps:
                    timestep_tensor = torch.tensor([t_step], device=device).long()
                    model_output_noise = model(
                        history_tensor, noisy_patch_sample, timestep_tensor, 
                        time_feat, spatial_feat
                    )
                    noisy_patch_sample = noise_scheduler.step(model_output_noise, t_step, noisy_patch_sample).prev_sample
                
                predicted_patch_normalized = noisy_patch_sample.squeeze(0).squeeze(0) # (H,W) on device
                predicted_patches_today[coords] = predicted_patch_normalized.cpu()

                if day_offset < config.AUTOREGRESSIVE_PREDICT_DAYS - 1:
                    new_history = history_sequence_list[1:] + [predicted_patch_normalized.detach().clone()]
                    current_history_all_coords[coords] = new_history
            
            full_predicted_image_normalized = reconstruct_image_from_patches(
                predicted_patches_today,
                (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
                config.PATCH_SIZE,
                config.STRIDE
            )
            full_predicted_image_denorm = denormalize_sst(full_predicted_image_normalized, min_sst, max_sst)
            
            ground_truth_image_denorm_np = None
            gt_filename = config.FILENAME_TEMPLATE.format(date_str=target_predict_dt.strftime("%Y%m%d"))
            gt_filepath = os.path.join(config.DATA_RAW_PATH, gt_filename)
            if os.path.exists(gt_filepath):
                try:
                    gt_sst_raw, _, _ = load_raw_sst_data(gt_filepath)
                    gt_sst_cropped = crop_raw_sst_data(gt_sst_raw, config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
                    ground_truth_image_denorm_np = gt_sst_cropped.values
                    
                    rmse, mse = calculate_metrics(
                        full_predicted_image_denorm.cpu().numpy(), # 确保在CPU上
                        ground_truth_image_denorm_np, 
                        land_sea_mask_np
                    )
                    print(f"  评估结果 ({lead_time_key}, {target_predict_date_str}): RMSE = {rmse:.4f}, MSE = {mse:.4f}")
                    evaluation_metrics[lead_time_key].append({'date': target_predict_date_str, 'rmse': rmse, 'mse': mse})
                except Exception as e:
                    print(f"  加载真实数据 {gt_filepath} 或计算指标失败: {e}")
            else:
                print(f"  警告: 未找到真实数据文件 {gt_filepath}，无法进行评估。")

            ensure_dir(config.PREDICTIONS_PATH)
            output_image_filepath = os.path.join(
                config.PREDICTIONS_PATH, 
                f"eval_pred_{target_predict_date_str}_{lead_time_key}.png"
            )
            # 添加SST绘图范围到config.py中，例如：
            # SST_PLOT_VMIN = -2 # 摄氏度 (或开尔文调整后)
            # SST_PLOT_VMAX = 35
            if not hasattr(config, 'SST_PLOT_VMIN'): # 临时添加默认值，如果config中没有
                config.SST_PLOT_VMIN = -2 
                config.SST_PLOT_VMAX = 35
                print(f"警告: config.py中未定义SST_PLOT_VMIN/VMAX，使用默认值 {config.SST_PLOT_VMIN} 至 {config.SST_PLOT_VMAX}。")


            save_image_comparison(
                full_predicted_image_denorm.cpu().numpy(), 
                ground_truth_image_denorm_np,
                land_sea_mask_np,
                output_image_filepath,
                title_prefix=f"{target_predict_date_str} ({lead_time_key}) "
            )
    
    print("\n--- 最终评估指标总结 (针对此次推理序列) ---")
    for lead_time, metrics_list in evaluation_metrics.items():
        if metrics_list:
            avg_rmse = np.nanmean([m['rmse'] for m in metrics_list])
            avg_mse = np.nanmean([m['mse'] for m in metrics_list])
            print(f"  {lead_time}: 平均 RMSE = {avg_rmse:.4f}, 平均 MSE = {avg_mse:.4f} (基于 {len(metrics_list)} 天的预测)")
            for m in metrics_list:
                print(f"    - {m['date']}: RMSE={m['rmse']:.4f}, MSE={m['mse']:.4f}")
        else:
            print(f"  {lead_time}: 未收集到评估指标。")

    print("推理与评估流程完成。")

if __name__ == "__main__":
    # 你可能需要修改checkpoint文件名和开始日期以进行测试
    # run_inference(checkpoint_filename="model_epoch_X.pt", inference_start_date_str="2020-03-YY")
    run_inference()