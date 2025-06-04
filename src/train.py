# src/train.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler # 区分开，避免与DDPMScheduler混淆
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np # 用于随机种子

# 从当前包（src）中导入
from . import config
from .dataset import SSTDataset # 假设SSTDataset在dataset.py中定义
from .models import get_diffusion_model # 假设get_diffusion_model在models.py中定义

def ensure_dir(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

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

def plot_loss_curve(epoch_losses, filepath):
    """绘制并保存loss曲线图"""
    ensure_dir(os.path.dirname(filepath))
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='训练损失 (MSE)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('平均损失')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"损失曲线图已保存至: {filepath}")

def train():
    """
    主训练函数
    """
    print("开始训练流程...")

    # 0. 确保输出目录存在 (config.py 中已定义)
    ensure_dir(config.CHECKPOINT_PATH)
    ensure_dir(config.FIGURES_PATH)

    # 1. 设置随机种子 (为了可复现性)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True # 可能影响性能，但增强复现性
        torch.backends.cudnn.benchmark = False

    # 2. 准备数据
    print("初始化训练数据集...")
    train_dataset = SSTDataset(mode='train')
    if len(train_dataset) == 0:
        print("错误：训练数据集为空，无法开始训练。请检查数据预处理和Dataset实现。")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # 训练时打乱数据
        num_workers=4, # 根据CPU核心数调整
        pin_memory=True # 如果使用GPU，可以加速数据传输
    )
    print(f"训练数据加载器准备完毕，每个epoch包含 {len(train_dataloader)} 个批次。")

    # 3. 初始化模型
    print("初始化模型...")
    device = torch.device(config.DEVICE)
    model = get_diffusion_model() # 这会返回 DiffusionUNetWithMLP 实例
    model.to(device)
    print(f"模型已移动到设备: {device}")

    # 4. 初始化 DDPM 噪声调度器
    # beta_schedule 来自 config，例如 "linear", "squaredcos_cap_v2" 等
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.DDPM_NUM_TRAIN_TIMESTEPS,
        beta_schedule=config.DDPM_BETA_SCHEDULE
    )
    print(f"DDPM噪声调度器 ({config.DDPM_BETA_SCHEDULE} schedule, {config.DDPM_NUM_TRAIN_TIMESTEPS} 步) 已初始化。")

    # 5. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    print(f"优化器 AdamW 已初始化，学习率: {config.LEARNING_RATE}")

    # 6. 初始化学习率调度器 (可选，但推荐)
    num_training_steps_per_epoch = len(train_dataloader)
    num_total_training_steps = num_training_steps_per_epoch * config.NUM_EPOCHS
    
    lr_scheduler = get_lr_scheduler(
        name="linear", # 例如: linear, cosine, constant
        optimizer=optimizer,
        num_warmup_steps=0, # 可以设置预热步数
        num_training_steps=num_total_training_steps
    )
    print(f"学习率调度器 (linear) 已初始化，总训练步数: {num_total_training_steps}")

    # 7. 训练循环
    print(f"开始训练，共 {config.NUM_EPOCHS} 轮。")
    epoch_losses = []

    for epoch in range(config.NUM_EPOCHS):
        model.train() # 设置模型为训练模式
        running_loss_epoch = 0.0
        
        progress_bar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS}"
        )

        for step, batch in progress_bar:
            # 从batch中获取数据
            history_patches, target_patch, raw_time_features, raw_spatial_features = batch
            
            # 将数据移动到指定设备
            history_patches = history_patches.to(device)     # (B, 30, H, W)
            target_patch = target_patch.to(device)           # (B, 1, H, W) - 这是干净的目标图像 x0
            raw_time_features = raw_time_features.to(device) # (B, num_raw_time_feat)
            raw_spatial_features = raw_spatial_features.to(device) # (B, num_raw_space_feat)
            
            current_batch_size = target_patch.shape[0]

            # 1. 采样高斯噪声 epsilon
            noise = torch.randn_like(target_patch) # 与 target_patch 形状相同

            # 2. 采样随机时间步 t
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (current_batch_size,), 
                device=device
            ).long()

            # 3. 向干净图像中添加噪声，得到带噪声的图像 x_t (noisy_target_patch)
            # target_patch 是 x0
            noisy_target_patch = noise_scheduler.add_noise(target_patch, noise, timesteps)
            # noisy_target_patch 形状 (B, 1, H, W)

            # 清零梯度
            optimizer.zero_grad()

            # 4. 模型前向传播，预测噪声
            # 模型输入: history_patches, noisy_target_patch, timesteps, raw_time_features, raw_spatial_features
            predicted_noise = model(
                history_patches, 
                noisy_target_patch, 
                timesteps, 
                raw_time_features, 
                raw_spatial_features
            ) # predicted_noise 形状 (B, 1, H, W)

            # 5. 计算损失 (预测噪声与真实噪声之间的MSE)
            loss = F.mse_loss(predicted_noise, noise)

            # 6. 反向传播和优化
            loss.backward()
            optimizer.step()
            lr_scheduler.step() # 更新学习率

            running_loss_epoch += loss.item()
            progress_bar.set_postfix({"当前批次损失": f"{loss.item():.4f}", "学习率": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

            # (可选) 定期打印批次损失
            if step % config.LOG_INTERVAL == 0 and step > 0:
                 print(f"    [轮次 {epoch+1}, 批次 {step}/{len(train_dataloader)}] 平均批次损失 (最近{config.LOG_INTERVAL}批): {running_loss_epoch/(step+1):.4f}")


        avg_epoch_loss = running_loss_epoch / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"轮次 {epoch+1} 完成。平均训练损失: {avg_epoch_loss:.4f}, 当前学习率: {lr_scheduler.get_last_lr()[0]:.2e}")

        # (可选) 定期保存模型checkpoint
        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0 or (epoch + 1) == config.NUM_EPOCHS:
            checkpoint_filepath = os.path.join(config.CHECKPOINT_PATH, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(epoch + 1, model, optimizer, avg_epoch_loss, checkpoint_filepath)
            
            # 同时更新并保存loss曲线图
            loss_curve_filepath = os.path.join(config.FIGURES_PATH, "training_loss_curve.png")
            plot_loss_curve(epoch_losses, loss_curve_filepath)


    print("训练完成。")
    # 最终再保存一次模型和loss图 (如果上面已保存，可能重复，但确保最后状态被保存)
    final_model_path = os.path.join(config.CHECKPOINT_PATH, "model_final.pt")
    save_checkpoint(config.NUM_EPOCHS, model, optimizer, epoch_losses[-1] if epoch_losses else float('inf'), final_model_path)
    
    final_loss_curve_path = os.path.join(config.FIGURES_PATH, "training_loss_curve_final.png")
    plot_loss_curve(epoch_losses, final_loss_curve_path)


if __name__ == "__main__":
    #  python -m src.train 运行
    train()