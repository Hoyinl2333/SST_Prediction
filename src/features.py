# src/features.py

import math
import torch
import torch.nn as nn
from datetime import datetime
import calendar # 用于获取月份天数
from src import config

# 1. time_embedding
def get_time_features(date_str: str, reference_year: int = 2020):
    """
    为给定的日期字符串生成时间特征。
    包括：归一化的年份、月份和日期的周期性编码（正弦/余弦）。
    返回: torch.Tensor: 包含时间特征的一维张量 (5个特征)。
    """
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year_norm = (dt_obj.year - reference_year) / 10.0  # 假设2020年为参考年
    month_sin = np.sin(2 * np.pi * dt_obj.month / 12.0)
    month_cos = np.cos(2 * np.pi * dt_obj.month / 12.0)
    days_in_month = calendar.monthrange(dt_obj.year, dt_obj.month)[1]  # 获取当月天数
    day_sin = np.sin(2 * np.pi * dt_obj.day / days_in_month)
    day_cos = np.cos(2 * np.pi * dt_obj.day / days_in_month)
    return torch.tensor([year_norm, month_sin, month_cos, day_sin, day_cos], dtype=torch.float32)

# 2.spatail_embedding
def get_spatial_features(patch_coords: tuple, full_image_dims: tuple, patch_size: int):
    """
    为给定的patch坐标生成归一化的空间特征 (左上角坐标归一化)。
    返回: torch.Tensor: 包含归一化空间特征的一维张量 (2个特征)。
    """
    row, col = patch_coords
    height, width = full_image_dims
    normalized_row = row / (height - patch_size)
    normalized_col = col / (width - patch_size)
    return torch.tensor([normalized_row, normalized_col], dtype=torch.float32)

# 3. 条件MLP
class ConditionalMLP(nn.Module):
    """
    一个MLP，将编码后的时间和空间特征映射为条件嵌入向量 C_cond。
    """
    def __init__(self,num_time_features:int,num_space_features:int,hidden_dims:list,output_dims:int):
        super().__init__()
        input_dim = num_time_features + num_space_features
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.SiLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dims))
        self.mlp = nn.Sequential(*layers)

    def forward(self,time_features:torch.Tensor,spatial_features:torch.Tensor) -> torch.Tensor:
        """
        前向传播：将时间和空间特征拼接后通过MLP。
        time_features: torch.Tensor, shape (batch_size, num_time_features)
        spatial_features: torch.Tensor, shape (batch_size, num_space_features)
        返回: torch.Tensor, shape (batch_size, output_dims)
        """
        combined_features = torch.cat((time_features, spatial_features), dim=1)
        return self.mlp(combined_features)
    
# 实例化MLP
def get_conditional_mlp():
    """根据config中的参数实例化条件MLP"""
    num_time_features = 5  # 5个时间特征
    num_sparital_features = 2  # 2个空间特征

    return ConditionalMLP(
        num_time_features=num_time_features,
        num_space_features=num_sparital_features,
        hidden_dims=config.MLP_HIDDEN_DIMS,  # 从配置中获取隐藏层维度
        output_dims=config.CONDITION_EMBED_DIM     # 输出维度
    )