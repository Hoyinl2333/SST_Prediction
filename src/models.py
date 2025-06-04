# src/models.py

import torch
import torch.nn as nn
from diffusers import UNet2DModel # 使用经典U-Net结构

# 从当前包（src）中导入配置和特征模块
from src import config
from src.features import get_conditional_mlp # 用于获取已配置的MLP实例

class DiffusionUNetWithMLP(nn.Module):
    """
    封装了条件MLP和U-Net的扩散模型。
    """
    def __init__(self):
        super().__init__()

        # 1. 初始化条件MLP
        # get_conditional_mlp() 会根据config.py中的设置返回一个ConditionalMLP实例
        self.conditional_mlp = get_conditional_mlp() 

        # 2. 初始化U-Net (UNet2DModel)
        self.unet = UNet2DModel(
            sample_size=(config.PATCH_SIZE, config.PATCH_SIZE), # 目标patch的尺寸
            in_channels=config.UNET_IN_CHANNELS,        # 输入通道: 30历史SST + 1带噪声目标SST
            out_channels=config.UNET_OUT_CHANNELS,      # 输出通道: 预测的噪声 (对应目标SST)
            block_out_channels=config.UNET_BLOCK_OUT_CHANNELS,
            down_block_types=config.UNET_DOWN_BLOCK_TYPES,
            up_block_types=config.UNET_UP_BLOCK_TYPES,
            # 条件相关的参数 for UNet2DModel
            class_embed_type=config.UNET_CLASS_EMBED_TYPE, # 例如 'projection'
            # projection_class_embeddings_input_dim 应该是MLP的输出维度 config.CONDITION_EMBED_DIM
            projection_class_embeddings_input_dim=config.CONDITION_EMBED_DIM, 
            norm_num_groups=config.NORMAL_NUM_GROUPS, 
            dropout = config.DROPOUT
        )

    def forward(self, 
                history_patches: torch.Tensor,    # (B, HISTORY_DAYS, H, W)
                noisy_target_patch: torch.Tensor, # (B, 1, H, W) - 带噪声的目标patch
                timestep: torch.Tensor,           # (B,) - 扩散时间步
                raw_time_features: torch.Tensor,  # (B, num_raw_time_features)
                raw_spatial_features: torch.Tensor # (B, num_raw_space_features)
               ) -> torch.Tensor:
        """
        模型的前向传播。

        返回:
        - torch.Tensor: U-Net预测的噪声，形状 (B, UNET_OUT_CHANNELS, H, W)
        """
        # 1. 通过MLP生成条件嵌入 C_cond
        # conditional_mlp 输入: (B, num_time_feat), (B, num_space_feat)
        # conditional_mlp 输出: (B, CONDITION_EMBED_DIM)
        condition_embedding_C_cond = self.conditional_mlp(raw_time_features, raw_spatial_features)

        # 2. 准备U-Net的输入图像数据
        # 将历史patches和带噪声的目标patch在通道维度上拼接
        # history_patches: (B, 30, H, W)
        # noisy_target_patch: (B, 1, H, W)
        # unet_input: (B, 31, H, W)
        unet_input = torch.cat((history_patches, noisy_target_patch), dim=1)

        # 3. U-Net前向传播
        # UNet2DModel 输入: 
        #   - sample (B, UNET_IN_CHANNELS, H, W)
        #   - timestep (B,)
        #   - class_labels (B, UNET_PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM) - 即我们的 C_cond
        # UNet2DModel 输出: .sample 属性是 (B, UNET_OUT_CHANNELS, H, W)
        predicted_noise = self.unet(
            sample=unet_input,
            timestep=timestep,
            class_labels=condition_embedding_C_cond 
        ).sample # 获取预测的噪声样本

        return predicted_noise

# --- 辅助函数：根据config实例化并返回主模型 ---
def get_diffusion_model():
    """
    根据 config.py 中的设置 (MODEL_ARCHITECTURE) 创建并返回主模型实例。
    目前只支持 "classic_unet"。
    """
    if config.MODEL_ARCHITECTURE == "classic_unet":
        print(f"初始化模型: DiffusionUNetWithMLP (基于 UNet2DModel)")
        model = DiffusionUNetWithMLP()
    # elif config.MODEL_ARCHITECTURE == "another_custom_model":
    #     model = AnotherCustomModel() # 未来可以扩展
    else:
        raise ValueError(f"未知的模型架构在config中定义: {config.MODEL_ARCHITECTURE}")
    return model