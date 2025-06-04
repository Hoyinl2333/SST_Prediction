# src/config.py

import torch
import os # 导入os模块方便路径拼接

# --- 项目根目录设定 (根据你的实际项目结构调整) ---
# 假设此config.py文件位于 "sst_diffusion/src/" 目录下
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# --- 数据路径 ---
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data/raw/ERA5")  # 原始NetCDF数据存放路径
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed/ERA5") # 预处理后的数据输出路径
PATCHES_PATH = os.path.join(DATA_PROCESSED_PATH, "patches")
LAND_SEA_MASK_PATH = os.path.join(DATA_PROCESSED_PATH, "land_sea_mask.pt")
NORMALIZATION_STATS_PATH = os.path.join(DATA_PROCESSED_PATH, "normalization_stats.pt")

# --- 数据参数 ---
SST_VARIABLE_NAME = 'sst'
LON_VARIABLE_NAME = 'longitude'
LAT_VARIABLE_NAME = 'latitude'
FILENAME_TEMPLATE = "{date_str}_ERA5_daily_mean_sst.nc"

# 图像参数
IMAGE_TARGET_HEIGHT = 720    # 裁剪后的图像目标高度
IMAGE_TARGET_WIDTH = 1440    # 裁剪后的图像目标宽度

# --- 预处理参数 ---
PATCH_SIZE = 64              # Patch 的边长 (假设为正方形)
STRIDE = 32                  # Patch 的步长 (PATCH_SIZE - STRIDE = 重叠区域)

# 数据集划分 (没有划分验证集)
DATA_START_DATE = "2020-01-01" 
DATA_END_DATE = "2020-03-31"  
# 我们将2020年1月和2月的数据作为训练集，3月的数据作为测试集。
TRAIN_PERIOD_END_DATE = "2020-02-29" # 训练集结束日期 (注意月份天数，2020是闰年2月有29天)

# --- 特征工程参数 (条件嵌入) ---
TIME_EMBED_DIM = 64          # 时间编码原始嵌入维度 (送入MLP前)
SPACE_EMBED_DIM = 2          # 空间编码原始嵌入维度 (通常是归一化的x,y坐标，送入MLP前)
MLP_HIDDEN_DIMS = [128, 128] # 条件MLP的隐藏层维度
CONDITION_EMBED_DIM = 256    # MLP输出的最终条件嵌入维度 C_cond (重要：需与U-Net配置匹配)

# --- 模型参数 (Classic UNet using UNet2DModel) ---
MODEL_ARCHITECTURE = "classic_unet" # 用于标识模型类型
HISTORY_DAYS = 30            # 使用过去多少天的SST数据
TARGET_DAYS = 1              # 预测未来1天 (自回归基础)

# U-Net (UNet2DModel) 配置
UNET_IN_CHANNELS = HISTORY_DAYS + TARGET_DAYS # 输入通道: 30历史SST + 1带噪声目标SST
UNET_OUT_CHANNELS = TARGET_DAYS             # 输出通道: 预测的噪声 (对应目标SST的1个通道)
UNET_BLOCK_OUT_CHANNELS = (64, 128, 256, 256) # U-Net各层通道数 (示例，可调整深度/宽度)

UNET_DOWN_BLOCK_TYPES = tuple(["DownBlock2D"] * len(UNET_BLOCK_OUT_CHANNELS))
UNET_UP_BLOCK_TYPES = tuple(["UpBlock2D"] * len(UNET_BLOCK_OUT_CHANNELS))

# 条件化 UNet2DModel (通过 class_labels)
UNET_CLASS_EMBED_TYPE = "identity" # 当 class_embed_type="identity" 时, num_class_embeds 是 class_labels 向量的实际维度
UNET_NUM_CLASS_EMBEDS = CONDITION_EMBED_DIM
NORMAL_NUM_GROUPS = 32 # 做归一化时，把通道维拆成几个 group,通常32
DROPOUT = 0.0 

# --- 训练参数 ---
BATCH_SIZE = 4               # 批量大小 (根据GPU显存调整)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10              # 示例训练轮次
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DDPM (Denoising Diffusion Probabilistic Models) 参数
DDPM_NUM_TRAIN_TIMESTEPS = 1000 # 训练时总扩散步数
DDPM_BETA_SCHEDULE = "linear"   # beta调度策略 (linear, cosine, etc.)

# --- 推理参数 ---
DDPM_NUM_INFERENCE_STEPS = 50 # 推理采样步数 (通常少于训练步数)
AUTOREGRESSIVE_PREDICT_DAYS = 3 # 自回归连续预测的天数 (T=3)

# --- 结果保存路径 ---
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_PATH, "checkpoints")
FIGURES_PATH = os.path.join(RESULTS_PATH, "figures")
PREDICTIONS_PATH = os.path.join(RESULTS_PATH, "predictions")

# --- 其他 ---
RANDOM_SEED = 42             # 随机种子，用于可复现性
LOG_INTERVAL = 10            # 每多少个batch打印一次日志
SAVE_EPOCH_INTERVAL = 10     # 每多少个epoch保存一次模型