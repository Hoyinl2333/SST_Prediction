# SSTPredictionProject/src/train.py

import sys
import os
import time
import random

# --- sys.path 修改，确保可以导入项目模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 结束 sys.path 修改 ---

import torch
import torch.optim as optim
import torch.nn.functional as F
from diffusers import DDPMScheduler
# from diffusers.optimization import get_scheduler # 可选，用于学习率调度
from tqdm.auto import tqdm # 导入 tqdm

from configs import training_config as cfg
from src import data_utils
from src import model as model_utils

# --- 辅助函数 ---
def setup_seed(seed):
    """设置随机种子以保证可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) # numpy需要在之前导入
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建。"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录已创建: {dir_path}")

# --- 核心训练函数 ---
def train_epoch(epoch_num, model, dataloader, scheduler, optimizer, device, ocean_mask_for_loss, grad_clip_value=None):
    """执行一个训练 epoch。"""
    model.train()
    total_loss = 0.0
    
    broadcastable_ocean_mask = ocean_mask_for_loss.unsqueeze(0).unsqueeze(0).to(device)

    # 使用 tqdm 包裹 dataloader
    # leave=True 会在迭代完成后保留进度条，leave=False 则会清除
    # disable=None 会在非 TTY 环境下（如某些 CI 系统）自动禁用进度条
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch_num+1}/{cfg.NUM_EPOCHS} Training", 
        total=len(dataloader),
        leave=True, # Epoch 内的批处理进度条在 Epoch 结束后也保留
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    for batch_idx, batch in enumerate(progress_bar):
        condition_frames = batch["condition"].to(device)
        target_frames_x0 = batch["target"].to(device)

        optimizer.zero_grad()
        batch_size = target_frames_x0.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        noise = torch.randn_like(target_frames_x0)
        noisy_target_frames = scheduler.add_noise(target_frames_x0, noise, timesteps)
        unet_input = torch.cat((noisy_target_frames, condition_frames), dim=1)
        predicted_noise = model(unet_input, timesteps).sample

        squared_error = (predicted_noise - noise) ** 2
        masked_squared_error = squared_error * broadcastable_ocean_mask
        loss = masked_squared_error.sum() / broadcastable_ocean_mask.sum().clamp(min=1e-5)

        loss.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        total_loss += loss.item()
        
        # 更新 tqdm 进度条的后缀信息，显示当前批次的损失
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        # LOG_INTERVAL 的打印可以保留，tqdm 会处理好多行输出
        if (batch_idx + 1) % cfg.LOG_INTERVAL == 0 or (batch_idx + 1) == len(dataloader):
            # 使用 tqdm.write 来打印，避免扰乱进度条
            tqdm.write(f"  Epoch [{epoch_num+1}/{cfg.NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Inst. Loss: {loss.item():.6f}")

    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss

@torch.no_grad()
def sample_loop(model, scheduler, condition_frames, target_shape, device, num_inference_steps=None):
    """生成采样过程。"""
    model.eval()

    if num_inference_steps is None:
        num_inference_steps = scheduler.config.num_train_timesteps

    noisy_sample = torch.randn(target_shape, device=device)
    condition_frames = condition_frames.to(device)

    scheduler.set_timesteps(num_inference_steps)

    # 为采样过程也添加 tqdm 进度条
    # desc 可以更具体，例如 "Sampling"
    sampling_progress_bar = tqdm(
        scheduler.timesteps, 
        desc="  Sampling/Denoising",
        total=len(scheduler.timesteps),
        leave=False, # 通常采样完成后不保留此进度条
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    for t in sampling_progress_bar:
        unet_input = torch.cat((noisy_sample, condition_frames), dim=1)
        timestep_tensor = torch.tensor([t] * unet_input.shape[0], device=device).long()
        noise_pred = model(unet_input, timestep_tensor).sample
        scheduler_output = scheduler.step(noise_pred, t, noisy_sample)
        noisy_sample = scheduler_output.prev_sample
    
    return noisy_sample


def main_train():
    """主训练流程。"""
    setup_seed(cfg.RANDOM_SEED)
    ensure_dir(cfg.CHECKPOINT_DIR)

    print(f"使用的设备: {cfg.DEVICE}")
    device = torch.device(cfg.DEVICE)

    train_dataloader, test_dataloader, ocean_mask_torch, norm_min, norm_max = \
        data_utils.get_dataloaders()
    
    unet_model = model_utils.create_unet_model().to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
        beta_schedule=cfg.BETA_SCHEDULE,
    )

    optimizer = optim.AdamW(
        unet_model.parameters(), 
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.OPTIMIZER_WEIGHT_DECAY
    )

    print("开始训练...")
    start_time = time.time()

    # 使用 tqdm 包裹 epoch 循环
    epoch_progress_bar = tqdm(
        range(cfg.NUM_EPOCHS), 
        desc="Overall Training Progress",
        # initial=0, # 如果是断点续训，可以设置 initial
        # total=cfg.NUM_EPOCHS, # range() 已经提供了长度信息
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    for epoch in epoch_progress_bar:
        # train_epoch 内部的 desc 会显示当前 epoch
        avg_train_loss = train_epoch(
            epoch, unet_model, train_dataloader, noise_scheduler, optimizer, device, ocean_mask_torch, grad_clip_value=1.0
        )
        # 更新 epoch 进度条的后缀信息
        epoch_progress_bar.set_postfix(avg_epoch_loss=f"{avg_train_loss:.6f}")
        
        # 使用 tqdm.write 打印 Epoch 结束信息，保持输出整洁
        tqdm.write(f"--- Epoch {epoch+1}/{cfg.NUM_EPOCHS} 结束 --- 平均训练损失: {avg_train_loss:.6f}")


        if (epoch + 1) % 10 == 0 or epoch == cfg.NUM_EPOCHS - 1:
            tqdm.write(f"\n在 Epoch {epoch+1} 结束时进行采样评估...") # 使用 tqdm.write
            if len(test_dataloader) > 0:
                eval_batch = next(iter(test_dataloader))
                eval_condition = eval_batch["condition"][:1].to(device)
                eval_target_actual = eval_batch["target"][:1].to(device)
                sample_target_shape = (1, cfg.H_FUTURE, cfg.PROCESSED_IMG_HEIGHT, cfg.PROCESSED_IMG_WIDTH)
                
                generated_frames = sample_loop(
                    unet_model, noise_scheduler, eval_condition, sample_target_shape, device, cfg.NUM_INFERENCE_STEPS
                )
                
                eval_loss_sq_err = (generated_frames - eval_target_actual)**2
                eval_broadcast_mask = ocean_mask_torch.unsqueeze(0).unsqueeze(0).to(device)
                eval_masked_loss = (eval_loss_sq_err * eval_broadcast_mask).sum() / eval_broadcast_mask.sum().clamp(min=1e-5)
                tqdm.write(f"  采样评估: 单个测试样本上的预测与真实目标之间的Masked MSE: {eval_masked_loss.item():.6f}")
            else:
                tqdm.write("  测试数据加载器为空，跳过采样评估。")

        if (epoch + 1) % 20 == 0 or epoch == cfg.NUM_EPOCHS - 1:
            checkpoint_name = f"sst_diffusion_T{cfg.T_PAST}_H{cfg.H_FUTURE}_epoch{epoch+1}.pth"
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, checkpoint_name)
            torch.save(unet_model.state_dict(), checkpoint_path)
            tqdm.write(f"模型已保存到: {checkpoint_path}") # 使用 tqdm.write

    end_time = time.time()
    final_message = f"\n训练完成！总耗时: {(end_time - start_time)/60:.2f} 分钟"
    print(final_message) # 最后的总结信息可以用 print


if __name__ == '__main__':
    import numpy as np # 确保 numpy 已导入
    main_train()