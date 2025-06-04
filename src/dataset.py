# src/dataset.py

import os
import glob
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd 

# 从当前包（src）中导入配置和特征工程函数
from src import config
from src.features import get_time_features, get_spatial_features 

class SSTDataset(Dataset):
    def __init__(self, mode='train'):
        """
        初始化SST数据集。

        参数:
        - mode (str): 'train' 或 'test'，决定加载哪些数据。
        """
        super().__init__()
        self.mode = mode
        self.history_days = config.HISTORY_DAYS

        # 1. 确定日期范围
        self.train_period_end_dt = datetime.strptime(config.TRAIN_PERIOD_END_DATE, "%Y-%m-%d")
        
        if self.mode == 'train':
            self.start_date_dt = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d")
            self.end_date_dt = self.train_period_end_dt
        elif self.mode == 'test':
            self.start_date_dt = self.train_period_end_dt + timedelta(days=1)
            self.end_date_dt = datetime.strptime(config.DATA_END_DATE, "%Y-%m-%d")
        else:
            raise ValueError(f"未知的模式: {mode}. Mode 必须是 'train' 或 'test'.")

        print(f"为 '{self.mode}' 模式初始化SSTDataset, 日期范围: {self.start_date_dt.strftime('%Y-%m-%d')} 到 {self.end_date_dt.strftime('%Y-%m-%d')}")

        # 2. 扫描所有patch文件并按 (坐标, 日期) 组织
        # 文件名格式示例: "2020-01-01_patch_r000_c000.pt"
        all_patch_files = glob.glob(os.path.join(config.PATCHES_PATH, "*.pt"))
        
        # patches_by_coord_date[coords_tuple][date_str] = filepath
        self.patches_by_coord_date = defaultdict(dict) 
        
        for patch_file in all_patch_files:
            basename = os.path.basename(patch_file)
            try:
                # 从文件名解析日期和坐标
                parts = basename.replace(".pt", "").split('_patch_') # [date_str, rXXX_cYYY]
                date_str = parts[0] # YYYY-MM-DD
                coord_str_parts = parts[1].split('_') # [rXXX, cYYY]
                r = int(coord_str_parts[0][1:]) # 提取 rXXX 中的 XXX
                c = int(coord_str_parts[1][1:]) # 提取 cYYY 中的 YYY
                coords = (r, c)
                
                patch_date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                # 根据当前模式(train/test)筛选日期范围内的patch
                if self.start_date_dt <= patch_date_dt <= self.end_date_dt:
                    self.patches_by_coord_date[coords][date_str] = patch_file
            except Exception as e:
                print(f"警告: 解析文件名 {basename} 失败或日期转换失败，跳过。错误: {e}")
                continue
        
        if not self.patches_by_coord_date:
            print(f"警告: 在 '{self.mode}' 模式的日期范围内未找到任何patch文件。请检查 {config.PATCHES_PATH} 目录和日期配置。")
            self.samples = []
            return

        # 3. 构建样本序列 (list_of_history_filepaths, target_filepath, target_date_str, target_coords)
        self.samples = []
        # 遍历每个空间坐标位置
        for coords, date_to_file_map in self.patches_by_coord_date.items():
            # 将该坐标下的日期字符串排序，以便构建时序序列
            sorted_dates_str = sorted(date_to_file_map.keys())
            
            # 至少需要 history_days + 1 天的数据才能构成一个样本
            if len(sorted_dates_str) < self.history_days + 1:
                continue

            # 滑动窗口构建样本
            for i in range(self.history_days, len(sorted_dates_str)):
                target_date_str = sorted_dates_str[i]
                target_filepath = date_to_file_map[target_date_str]
                
                history_filepaths = []
                possible_sample = True
                for j in range(1, self.history_days + 1):
                    history_date_idx = i - j
                    history_date_str_expected = sorted_dates_str[history_date_idx] # 从已排序的列表中取
                    
                    # 检查期望的历史日期是否存在于map中 (通常应该存在，因为我们是基于map的key排序的)
                    # 并且，更重要的是，检查日期间隔是否为1天 (对于严格连续性)
                    # 但由于我们是基于已存在文件构建，所以只要能找到前 history_days 个文件即可
                    # 这里我们假设 sorted_dates_str 已经是连续的，如果不是，预处理或数据源有问题
                    history_filepaths.append(date_to_file_map[history_date_str_expected])

                if len(history_filepaths) == self.history_days:
                    self.samples.append({
                        'history_paths': list(reversed(history_filepaths)), # 历史文件是倒序append的，所以需要反转一下，使其按时间顺序 (从最早到最近)
                        'target_path': target_filepath,
                        'target_date_str': target_date_str,
                        'target_coords': coords
                    })
        
        if not self.samples:
            print(f"警告: 未能为 '{self.mode}' 模式构建任何有效的输入序列。")
        else:
            print(f"为 '{self.mode}' 模式构建了 {len(self.samples)} 个样本。")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 1. 加载历史patches并堆叠
        history_patches_list = []
        for history_path in sample_info['history_paths']:
            patch_data = torch.load(history_path)
            # sst_patch 形状是 (patch_size, patch_size)
            history_patches_list.append(patch_data['sst_patch']) 
        
        # 堆叠成 (HISTORY_DAYS, patch_size, patch_size)
        history_patches_stacked = torch.stack(history_patches_list, dim=0) 

        # 2. 加载目标patch
        target_patch_data = torch.load(sample_info['target_path'])
        target_patch = target_patch_data['sst_patch'] # 形状 (patch_size, patch_size)
        # 增加一个通道维度 -> (1, patch_size, patch_size)
        target_patch_with_channel = target_patch.unsqueeze(0) 

        # 3. 获取目标的时间和空间原始特征
        target_date_str = sample_info['target_date_str']
        target_coords = sample_info['target_coords']
        
        # 从config获取参考年份和图像尺寸
        # 对于参考年份，可以使用数据起始年份
        ref_year = datetime.strptime(config.DATA_START_DATE, "%Y-%m-%d").year
        time_feat = get_time_features(target_date_str, reference_year=ref_year)
        
        spatial_feat = get_spatial_features(
            target_coords, 
            (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH), 
            config.PATCH_SIZE
        )
        
        return history_patches_stacked, target_patch_with_channel, time_feat, spatial_feat