# main.py (位于项目根目录, 与 src/ 同级)

import argparse
import torch
import numpy as np
import os
import sys

# 为了能够从根目录的 main.py 正确导入 src 包中的模块，
# 我们需要确保 src 的父目录（即项目根目录）在 Python 的搜索路径中。
# 如果你是从项目根目录运行 `python main.py ...`，这通常会自动处理好。
# 如果遇到导入问题，可以取消下面几行的注释来手动添加路径：
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir) 
# 或者如果 main.py 在根目录，src 是子目录：
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# 从 src 包导入各个模块的功能
# 注意：如果 main.py 和 src 在同一级别，导入方式应为 from src.preprocess import run_preprocessing
try:
    from src import config # 用于设置随机种子和获取一些路径（如果需要）
    from src.preprocess import run_preprocessing
    from src.train import train as run_training # 别名以防与argparse的train冲突
    from src.inference import run_inference
except ModuleNotFoundError:
    print("错误：无法导入 src 包中的模块。")
    print("请确保你从项目的根目录运行此脚本 (例如 `python main.py --train`)，")
    print("并且 main.py 与 src/ 文件夹在同一级别。")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SST时空序列预测模型的控制脚本")
    
    # 添加互斥组，确保一次只执行一个主要操作
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--preprocess", 
        action="store_true", 
        help="执行数据预处理流程。"
    )
    action_group.add_argument(
        "--train", 
        action="store_true", 
        help="执行模型训练流程。"
    )
    action_group.add_argument(
        "--inference", 
        action="store_true", 
        help="执行模型推理与评估流程。"
    )

    # 为推理添加可选参数
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="model_final.pt", # 默认使用训练结束时保存的最终模型
        help="用于推理的checkpoint文件名 (位于 config.CHECKPOINT_PATH 下)。例如: 'model_epoch_50.pt'"
    )
    parser.add_argument(
        "--inference_date", 
        type=str, 
        default=None, # 默认从测试集第一天开始
        help="推理开始的日期 (YYYY-MM-DD)。例如: '2020-03-15'"
    )
    # 可以根据需要为训练等添加更多命令行参数来覆盖config中的值

    args = parser.parse_args()

    # (可选) 在此处设置全局随机种子，尽管各个脚本内部也设置了
    if hasattr(config, 'RANDOM_SEED'):
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)
            # 以下两行可以根据是否严格要求复现性以及对性能的影响来决定是否启用
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False 
        print(f"全局随机种子已设置为: {config.RANDOM_SEED}")


    if args.preprocess:
        print("main.py: 请求执行数据预处理...")
        run_preprocessing()
    elif args.train:
        print("main.py: 请求执行模型训练...")
        run_training() # 使用别名 run_training
    elif args.inference:
        print("main.py: 请求执行模型推理与评估...")
        run_inference(
            checkpoint_filename=args.checkpoint,
            inference_start_date_str=args.inference_date
        )
    else:
        # 由于 action_group 设置了 required=True，理论上不会执行到这里
        print("错误：请指定一个操作 (--preprocess, --train, 或 --inference)。")
        parser.print_help()

if __name__ == "__main__":
    main()