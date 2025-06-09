import torch
import torch.nn as nn
from diffusers import UNet2DModel
from . import config
from .features import get_conditional_mlp

class DiffusionUNetWithMLP(nn.Module):
    """
    封装了条件MLP和U-Net的扩散模型。
    """
    def __init__(self):
        super().__init__()

        # 1. 初始化条件MLP (从单元格8定义的函数获取)
        self.conditional_mlp = get_conditional_mlp() 

        # 2. 初始化U-Net (UNet2DModel)
        # 参数从单元格1定义的全局配置中读取
        self.unet = UNet2DModel(
            sample_size=(config.PATCH_HEIGHT, config.PATCH_WIDTH),
            in_channels=config.UNET_IN_CHANNELS,
            out_channels=config.UNET_OUT_CHANNELS,
            block_out_channels=config.UNET_BLOCK_OUT_CHANNELS,
            down_block_types=config.UNET_DOWN_BLOCK_TYPES,
            up_block_types=config.UNET_UP_BLOCK_TYPES,
            class_embed_type=config.UNET_CLASS_EMBED_TYPE,
            num_class_embeds=config.UNET_NUM_CLASS_EMBEDS,
            # 根据需要可以添加其他 UNet2DModel 支持的参数，例如:
            # norm_num_groups=32, 
            # dropout=0.0
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
        返回: U-Net预测的噪声，形状 (B, UNET_OUT_CHANNELS, H, W)
        """
        # 1. 通过MLP生成条件嵌入 C_cond
        condition_embedding_C_cond = self.conditional_mlp(raw_time_features, raw_spatial_features)

        # 2. 准备U-Net的输入图像数据
        unet_input = torch.cat((history_patches, noisy_target_patch), dim=1) # (B, 31, H, W)

        # 3. U-Net前向传播
        # 当 class_embed_type="identity", class_labels 参数接收的是实际的嵌入向量
        predicted_noise = self.unet(
            sample=unet_input,
            timestep=timestep,
            class_labels=condition_embedding_C_cond 
        ).sample 

        return predicted_noise
    
def get_diffusion_model():
    if config.MODEL_ARCHITECTURE == "classic_unet":
        return DiffusionUNetWithMLP()
    else:
        raise ValueError(f"不支持的模型架构: {config.MODEL_ARCHITECTURE}. 目前仅支持 'classic_unet'.")