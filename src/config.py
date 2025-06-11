# config.py

import os
import torch

# --- 项目根目录设定 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# --- 数据路径 ---
# DATA_RAW_PATH = "/mnt/sata_16t/yaweiwang/ERA5"
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data/raw/ERA5") # 原始数据目录
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed/ERA5") 
PATCHES_PATH = os.path.join(DATA_PROCESSED_PATH, "patches")
LAND_SEA_MASK_PATH = os.path.join(DATA_PROCESSED_PATH, "land_sea_mask.pt")
NORMALIZATION_STATS_PATH = os.path.join(DATA_PROCESSED_PATH, "normalization_stats.pt")

# --- 数据参数 ---
SST_VARIABLE_NAME = 'sst'
LON_VARIABLE_NAME = 'longitude'
LAT_VARIABLE_NAME = 'latitude'
FILENAME_TEMPLATE = "{date_str}_ERA5_daily_mean_sst.nc"

IMAGE_TARGET_HEIGHT = 720
IMAGE_TARGET_WIDTH = 1440

# --- 预处理参数 ---
PATCH_HEIGHT = 48
PATCH_WIDTH = 48
STRIDE = 24

# 每天生成的 #patches = ((IMAGE_TARGET_HEIGHT - PATCH_HEIGHT)/STRIDE + 1) * ((IMAGE_TARGET_WIDTH - PATCH_WIDTH)/STRIDE +1)

# TODO:最后实验使用完整的时间
DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2020-03-31"
TRAIN_PERIOD_END_DATE = "2020-02-29"

# --- 特征工程参数 ---
MLP_HIDDEN_DIMS = [128] # MLP的隐藏层维度 
CONDITION_EMBED_DIM = 256 # MLP的输出维度，也是U-Net期望的条件向量维度

# --- 模型预测目标模式 ---
# 可选值: "absolute" (预测绝对SST值 )
#         "difference" (预测SST差值 )
PREDICTION_TARGET = "difference" 

# --- 模型参数 ---
MODEL_ARCHITECTURE = "classic_unet"
HISTORY_DAYS = 30
TARGET_DAYS = 1

UNET_IN_CHANNELS = HISTORY_DAYS + TARGET_DAYS
UNET_OUT_CHANNELS = TARGET_DAYS
UNET_BLOCK_OUT_CHANNELS = (64, 128, 256, 256)
UNET_DOWN_BLOCK_TYPES = tuple(["DownBlock2D"] * len(UNET_BLOCK_OUT_CHANNELS))
UNET_UP_BLOCK_TYPES = tuple(["UpBlock2D"] * len(UNET_BLOCK_OUT_CHANNELS))
UNET_CLASS_EMBED_TYPE = "identity"
UNET_NUM_CLASS_EMBEDS  = CONDITION_EMBED_DIM

# --- 训练参数 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_IDS =  [0,1] # 只有多卡时用这个才有效
NUM_WORKERS = 0  # DataLoader的工作线程数

DDPM_NUM_TRAIN_TIMESTEPS = 1000
DDPM_BETA_SCHEDULE = "linear"

# --- 推理参数 ---
DDPM_NUM_INFERENCE_STEPS = 1
AUTOREGRESSIVE_PREDICT_DAYS = 3

# --- 结果保存路径 ---
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")
# 更详细的逻辑在utils文件结果相关函数中

# --- 其他 ---
RANDOM_SEED = 42
LOG_INTERVAL = 1000 # 训练时日志打印间隔
SAVE_EPOCH_INTERVAL = 10 # 训练时模型保存间隔 

print("="*10+" config.py 关键信息打印 "+"="*10)
print(f"项目根目录设置为: {PROJECT_ROOT}")
print(f"原始数据路径: {DATA_RAW_PATH}")
print(f"预处理数据路径: {DATA_PROCESSED_PATH}")
print(f"设备设置为: {DEVICE}")
print(f"训练轮数 (NUM_EPOCHS) 设置为: {NUM_EPOCHS} ")
print(f"DDPM训练时间步数 (DDPM_NUM_TRAIN_TIMESTEPS) 设置为: {DDPM_NUM_TRAIN_TIMESTEPS}")
print(f"DDPM推理时间步数 (DDPM_NUM_INFERENCE_STEPS) 设置为: {DDPM_NUM_INFERENCE_STEPS}")
print(f"保存检查点间隔 (SAVE_EPOCH_INTERVAL) 设置为: {SAVE_EPOCH_INTERVAL} ")
print("="*50)