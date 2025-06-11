from torch.utils.data import DataLoader # 确保显式导入
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler
from torch.optim import AdamW
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import config
from .utils import save_checkpoint, plot_loss_curve,setup_train_directory
from .dataset import get_dataset
from .models import get_diffusion_model
from tqdm import tqdm

def run_training(run_name=None):
    """训练函数的主体逻辑"""
    print("开始训练流程...")
    train_run_dir  = setup_train_directory(run_name)

    # 1. 准备数据 (SSTDataset 在单元格9定义)
    print("初始化训练数据集...")
    train_dataset = get_dataset(mode='train') 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True if config.DEVICE == "cuda" else False 
    )
    print(f"训练数据加载器准备完毕，每个epoch包含 {len(train_dataloader)} 个批次。")

    # 2. 初始化模型 (get_diffusion_model 在单元格10定义)
    print("初始化模型...")
    model = get_diffusion_model() 
    device = config.DEVICE
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    if config.DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)  # 多GPU
    print(f"GPU = {model.device_ids if isinstance(model, nn.DataParallel) else 'Single GPU'}")
    model.to(device)

    # 3. 初始化 DDPM 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.DDPM_NUM_TRAIN_TIMESTEPS, # 来自单元格1
        beta_schedule=config.DDPM_BETA_SCHEDULE          # 来自单元格1
    )
    print(f"DDPM噪声调度器 ({config.DDPM_BETA_SCHEDULE} schedule, {config.DDPM_NUM_TRAIN_TIMESTEPS} 步) 已初始化。")

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE) # LEARNING_RATE 来自单元格1
    print(f"优化器 AdamW 已初始化，学习率: {config.LEARNING_RATE}")

    # 5. 初始化学习率调度器
    num_training_steps_per_epoch = len(train_dataloader)
    num_total_training_steps = num_training_steps_per_epoch * config.NUM_EPOCHS # NUM_EPOCHS 来自单元格1

    lr_scheduler = get_lr_scheduler(
        name="linear", 
        optimizer=optimizer,
        num_warmup_steps=0, 
        num_training_steps=num_total_training_steps
    )
    print(f"学习率调度器 (linear) 已初始化，总训练步数: {num_total_training_steps}")

    # 6. 训练循环12
    print(f"开始训练，共 {config.NUM_EPOCHS} 轮。")
    epoch_losses_history = [] # 用于存储每个epoch的平均损失
    for epoch in range(config.NUM_EPOCHS):
        model.train() 
        running_loss_this_epoch = 0.0 # 重命名以清晰表示是当前epoch的累计损失
        progress_bar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS}"
        )
        for step, batch in progress_bar:
            history_patches, target_patch, raw_time_features, raw_spatial_features = batch
            
            history_patches = history_patches.to(device)     
            target_patch = target_patch.to(device)          
            raw_time_features = raw_time_features.to(device) 
            raw_spatial_features = raw_spatial_features.to(device) 

            noise = torch.randn_like(target_patch)
            batch_size = target_patch.shape[0] # 获取当前批次大小
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (batch_size,), 
                device=device 
            ).long()
            noisy_target_patch = noise_scheduler.add_noise(target_patch, noise, timesteps)
            
            optimizer.zero_grad()
            predicted_noise = model(
                history_patches, 
                noisy_target_patch, 
                timesteps, 
                raw_time_features, 
                raw_spatial_features
            )
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss_this_epoch += loss.item()
            progress_bar.set_postfix({
                "损失": f"{loss.item():.4f}", 
                "学习率": f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })

        avg_epoch_loss_value = running_loss_this_epoch / len(train_dataloader)
        epoch_losses_history.append(avg_epoch_loss_value)
        print(f"轮次 {epoch+1} 完成。平均训练损失: {avg_epoch_loss_value:.4f}, 当前学习率: {lr_scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0 :
            checkpoint_filepath = os.path.join(train_run_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(epoch + 1, model, optimizer, avg_epoch_loss_value, checkpoint_filepath)

    # 最终保存一次模型和loss图 (确保最后的状态被保存)
    final_model_path = os.path.join(train_run_dir, "model_final.pt")
    final_loss_value = epoch_losses_history[-1] if epoch_losses_history else float('inf')
    save_checkpoint(config.NUM_EPOCHS, model, optimizer, final_loss_value, final_model_path)

    final_loss_curve_path = os.path.join(train_run_dir , "training_loss_curve_final.png")
    plot_loss_curve(epoch_losses_history, final_loss_curve_path)
    
    print("训练完成。")
