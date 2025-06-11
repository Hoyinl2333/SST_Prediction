import os
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from . import config
from .features import get_time_features, get_spatial_features
from .utils import check_date_contiguity

class SSTDatasetBase(Dataset):
    """
    基础SST数据集类，提供通用的加载和样本构建方法。
    """
    def __init__(self,mode='train'):
        super().__init__()
        self.mode = mode
        self.history_days = config.HISTORY_DAYS
        self.train_period_end_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d")
        if self.mode == 'train':
            self.start_date_dt = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d")
            self.end_date_dt = self.train_period_end_dt
        elif self.mode == 'test':
            self.start_date_dt = self.train_period_end_dt + timedelta(days=1)
            self.end_date_dt = datetime.strptime(config.DATA_END_DATE, "%Y-%m-%d")
        else:
            raise ValueError(f"未知的模式: {self.mode}。mode 请使用 'train' 或 'test'。")
        print(f"为 '{self.mode}' 模式初始化 {self.__class__.__name__}: {self.start_date_dt.strftime('%Y-%m-%d')} 到 {self.end_date_dt.strftime('%Y-%m-%d')}")

        # 加载所有patches到内存
        self. all_loaded_patches = self._load_patches_into_memory()

        # 构建样本序列
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f"警告: 未能为 '{self.mode}' 模式 ({self.__class__.__name__}) 构建任何有效的输入序列。")
        else:
            print(f"为 '{self.mode}' 模式 ({self.__class__.__name__}) 构建了 {len(self.samples)} 个样本。")


    def _load_patches_into_memory(self):
        """ 加载所有每日patch汇总文件到内存 """
        patches_dict = defaultdict(dict)  # [date_str][coords] = sst_tensor
        date_iter = self.start_date_dt
        date_range = []
        while date_iter <= self.end_date_dt:
            date_range.append(date_iter)
            date_iter += timedelta(days=1)
        
        for dt in tqdm(date_range, desc=f"加载 '{self.mode}' 模式的每日patches"):
            date_str = dt.strftime("%Y-%m-%d")
            file_path = os.path.join(config.PATCHES_PATH, f"{date_str}_patches.pt")
            if os.path.exists(file_path):
                patches = torch.load(file_path)
                for patch in patches:
                    patches_dict[date_str][tuple(patch['coords'])] = patch['sst_patch']
        return patches_dict

    def _build_samples(self):
        print("开始构造样本序列...")
        samples = []
        if not self.all_loaded_patches: 
            raise ValueError("没有加载到任何patch数据，请检查数据预处理是否正确。")
        
        # 获取一个坐标集合
        first_available_date = sorted(self.all_loaded_patches.keys())[0]
        coords_set = set(self.all_loaded_patches[first_available_date].keys())

        available_dates = sorted(self.all_loaded_patches.keys())

        for coords in tqdm(coords_set, desc=f"为'{self.mode}'模式构建样本序列(Difference)"):
            for i in range(len(available_dates) - self.history_days):
                window_of_dates = available_dates[i:i + self.history_days + 1] # 这里我们载入self.history_days + 1天的数据
                if check_date_contiguity(window_of_dates):
                    samples.append({
                        'window_of_dates': window_of_dates,
                        'coords': coords
                    })
        print(f"成功构造了 {len(samples)} 个样本序列。")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        """抽象方法：获取单个样本"""
        return NotImplementedError("子类必须实现 __getitem__ 方法以获取单个样本。")
    
class SSTDatasetAbsolute(SSTDatasetBase):
    """
    用于预测SST绝对值的Dataset
    """
    def __getitem__(self, idx):
        sample = self.samples[idx]
        coords = sample['coords']
        window_of_dates = sample['window_of_dates']

        history_dates = window_of_dates[:-1]
        target_date = window_of_dates[-1]

        hist_stack = torch.stack([self.all_loaded_patches[d][coords] for d in history_dates])  # shape: [HISTORY_DAYS, 1, patch_h, patch_w]

        target_patch = self.all_loaded_patches[target_date][coords]
        target = target_patch.unsqueeze(0)  # shape: [1, patch_h, patch_w]

        time_feat = get_time_features(
            history_dates,
            reference_year=datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d").year
        ) # shape:[5]

        spatial_feat = get_spatial_features(
            coords,
            full_image_dims=(config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
            patch_height=config.PATCH_HEIGHT,
            patch_width=config.PATCH_WIDTH
        ) # shape:[2]

        return hist_stack, target, time_feat, spatial_feat
    

class SSTDatasetDifference(SSTDatasetBase):
    """
    用于预测SST差值的Dataset
    """

    def __getitem__(self, idx):
        sample = self.samples[idx]
        coords = sample['coords']
        window_of_dates = sample['window_of_dates']

        history_dates = window_of_dates[:-1]
        history_stack = torch.stack([self.all_loaded_patches[d][coords] for d in history_dates])

        target = self.all_loaded_patches[window_of_dates[-1]][coords] - self.all_loaded_patches[window_of_dates[-2]][coords]
        target = target.unsqueeze(0)

        ref_year = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d").year
        time_feat = get_time_features(
            window_of_dates[-1],
            reference_year=ref_year
        )
        spatial_feat = get_spatial_features(
            coords,
            full_image_dims=(config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH),
            patch_height=config.PATCH_HEIGHT,
            patch_width=config.PATCH_WIDTH
        )
        return history_stack, target, time_feat, spatial_feat

# 工厂函数
def get_dataset(mode='train'):
    """
    根据模式和数据集类型获取对应的SST数据集实例。
    
    参数:
        mode (str): 'train' 或 'test'
        dataset_type (str): 'absolute' 或 'difference'
    
    返回:
        SSTDatasetBase: 对应的数据集实例
    """
    if config.PREDICTION_TARGET == 'absolute':
        return SSTDatasetAbsolute(mode=mode)
    elif config.PREDICTION_TARGET == 'difference':
        return SSTDatasetDifference(mode=mode)
    else:
        raise ValueError(f"未知的数据集类型: {config.PREDICTION_TARGET}. 目前仅支持 'absolute' 和 'difference'.")