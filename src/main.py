# main.py

import argparse
import sys
import os

try:
    from src.preprocess import run_preprocessing
    from src.train import  run_training
    from src.inference import run_inference
except ModuleNotFoundError:
    print("错误：无法导入 src 包。请确保从项目根目录运行此脚本。")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SST时空序列预测模型的控制脚本")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--preprocess", action="store_true", help="执行数据预处理流程。")
    action_group.add_argument("--train", action="store_true", help="执行模型训练流程。")
    action_group.add_argument("--inference", action="store_true", help="执行模型推理与评估流程。")
    parser.add_argument("--checkpoint",type=str,help="用于推理的checkpoint在checkpoints/目录下的相对路径。例如: '20250609_21/model_final.pt'")
    parser.add_argument("--date", type=str, default=None, help="推理开始的日期 (YYYY-MM-DD)。")
    args = parser.parse_args()

    if args.preprocess:
        run_preprocessing()
    elif args.train:
        run_training()
    elif args.inference:
        if not args.checkpoint:
            parser.error("--inference 操作需要一个 --checkpoint 参数。")
        run_inference(checkpoint_relative_path=args.checkpoint, inference_start_date_str=args.date)

if __name__ == "__main__":
    # python -m src.main --preprocess && python -m src.main --train && python -m src.main --inference --checkpoint "20250609_21/model_final.pt"
    main()