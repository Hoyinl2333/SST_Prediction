# src/features.py

import math
import torch
import torch.nn as nn
from datetime import datetime
import calendar # 用于获取月份天数

# 从当前包（src）中导入配置
from src import config

# --- 1. 时间编码 ---
def get_time_features(date_str: str, reference_year: int = 2020):
    """
    为给定的日期字符串生成时间特征。
    包括：归一化的年份、月份和日期的周期性编码（正弦/余弦）。

    参数:
    - date_str (str): 'YYYY-MM-DD' 格式的日期字符串。
    - reference_year (int): 用于年份归一化的参考年份。

    返回:
    - torch.Tensor: 包含时间特征的一维张量 (5个特征)。
                      [year_norm, month_sin, month_cos, day_sin, day_cos]
    """
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # 年份特征 
    year_norm = (dt_obj.year - reference_year) / 10.0

    # 月份的周期性编码 (1-12月)
    month = dt_obj.month
    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)

    # 日的周期性编码 (相对于当月总天数)
    # day = dt_obj.day
    # days_in_month = calendar.monthrange(dt_obj.year, dt_obj.month)[1]
    # day_sin = math.sin(2 * math.pi * day / days_in_month)
    # day_cos = math.cos(2 * math.pi * day / days_in_month)
    
    # 也可以考虑使用一年中的第几天 (day of year) 进行编码，如果季节性更重要
    doy = dt_obj.timetuple().tm_yday
    days_in_year = 366 if calendar.isleap(dt_obj.year) else 365
    doy_sin = math.sin(2 * math.pi * doy / days_in_year)
    doy_cos = math.cos(2 * math.pi * doy / days_in_year)

    return torch.tensor([year_norm, month_sin, month_cos, doy_sin, doy_cos], dtype=torch.float32)

# --- 2. 空间编码 ---
def get_spatial_features(patch_coords: tuple, full_image_dims: tuple, patch_size: int):
    """
    为给定的patch坐标生成归一化的空间特征。
    使用patch左上角坐标进行归一化。

    参数:
    - patch_coords (tuple): (row_start, col_start) patch在完整图像中的左上角坐标。
    - full_image_dims (tuple): (IMG_H, IMG_W) 完整图像的高度和宽度。
    - patch_size (int): patch的边长。

    返回:
    - torch.Tensor: 包含归一化空间特征的一维张量 (2个特征)。
                      [norm_row_start, norm_col_start]
    """
    row_start, col_start = patch_coords
    img_h, img_w = full_image_dims

    # 分母是可放置patch的最大起始位置，确保归一化范围大致在[0,1]
    # 如果图像高度为H，patch大小为P，则最后一个patch的起始行是 H-P
    # 所以有效的起始行范围是 [0, H-P]
    max_start_row = img_h - patch_size
    max_start_col = img_w - patch_size

    norm_row = row_start / max_start_row 
    norm_col = col_start / max_start_col 
    
    # 将归一化范围调整到 [-1, 1] 以匹配通常的激活函数对称中心（可选）
    # norm_row = norm_row * 2.0 - 1.0
    # norm_col = norm_col * 2.0 - 1.0

    return torch.tensor([norm_row, norm_col], dtype=torch.float32)


# --- 3. 条件MLP模型 ---
class ConditionalMLP(nn.Module):
    """
    一个MLP，将编码后的时间和空间特征映射为条件嵌入向量 C_cond。
    """
    def __init__(self, num_time_features: int, num_space_features: int, hidden_dims: list, output_dim: int):
        """
        初始化 ConditionalMLP.

        参数:
        - num_time_features (int): 时间特征的数量 (来自 get_time_features 的输出维度)。
        - num_space_features (int): 空间特征的数量 (来自 get_spatial_features 的输出维度)。
        - hidden_dims (list): MLP隐藏层的维度列表。
        - output_dim (int): 输出条件嵌入 C_cond 的维度。
        """
        super().__init__()
        
        input_dim = num_time_features + num_space_features
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.SiLU()) # SiLU (Swish) 是一种常用的激活函数，也可以用ReLU
            # layers.append(nn.ReLU())
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, time_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        MLP的前向传播。

        参数:
        - time_features (torch.Tensor): (batch_size, num_time_features)
        - spatial_features (torch.Tensor): (batch_size, num_space_features)

        返回:
        - torch.Tensor: (batch_size, output_dim) 条件嵌入 C_cond。
        """
        # 确保输入是二维的 (batch_size, num_features)
        if time_features.ndim == 1:
            time_features = time_features.unsqueeze(0)
        if spatial_features.ndim == 1:
            spatial_features = spatial_features.unsqueeze(0)
            
        # 拼接时间和空间特征
        combined_features = torch.cat((time_features, spatial_features), dim=1)
        
        return self.mlp(combined_features)

# --- 辅助函数：根据config实例化MLP (可选，方便调用) ---
def get_conditional_mlp():
    """
    根据 config.py 中的设置创建并返回 ConditionalMLP 实例。
    """
    # 从 get_time_features 和 get_spatial_features 的实现确定维度
    # get_time_features 返回5个特征
    # get_spatial_features 返回2个特征
    num_raw_time_features = 5 
    num_raw_space_features = 2
    
    return ConditionalMLP(
        num_time_features=num_raw_time_features,
        num_space_features=num_raw_space_features,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        output_dim=config.CONDITION_EMBED_DIM
    )