# main.py

import argparse
import sys
import os
from datetime import datetime

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
    action_group.add_argument("--preprocess", action="store_true", help="[独立] 仅执行数据预处理。")
    action_group.add_argument("--train", action="store_true", help="[独立] 仅执行模型训练。")
    action_group.add_argument("--inference", action="store_true", help="[独立] 仅对已有模型执行推理。")
    action_group.add_argument("--pipeline", action="store_true", help="[自动] 按顺序执行 预处理->训练->推理 的完整流程。")
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="用于 --train: (可选) 为本次训练的目录指定一个名称。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="用于 --inference: 指定要加载模型的相对路径。\n例如: 'run_20250611_111000/model_final.pt'"
    )

    args = parser.parse_args()

    if args.preprocess:
        print(" === [独立模式] 执行数据预处理 ===")
        run_preprocessing()
    elif args.train:
        print(" === [独立模式] 执行模型训练 ===")
        run_name = args.run_name if args.run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_training(run_name)
    elif args.inference:
        print(" === [独立模式] 执行模型推理 ===")
        if not args.checkpoint:
            parser.error("--inference 操作需要一个 --checkpoint 参数来指定模型。")
        run_inference(checkpoint_relative_path=args.checkpoint)
    elif args.pipeline:
        print(" === [自动化流程模式] 开始执行 ===")
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H')}"
        print(f"本次流程名称：{run_name}")

        print("\n--- 步骤 1/3: 执行数据预处理 ---")
        run_preprocessing()
        print("--- 数据预处理完成 ---\n")

        print("--- 步骤 2/3: 执行模型训练 ---")
        run_training(run_name)
        print("--- 模型训练完成 ---\n")

        print("--- 步骤 3/3: 执行模型推理 ---")
        checkpoint_for_pipeline = os.path.join(run_name, "model_final.pt")
        run_inference(checkpoint_relative_path=checkpoint_for_pipeline)
        print("--- 模型推理完成 ---\n")
    
    print("=== 操作完成 ===")

if __name__ == "__main__":

    #
    # 模式一: 自动化完整流程
    # -------------------------------------------------------------
    # 一键执行从数据预处理到推理的全部流程
    # python -m src.main --pipeline
    #
    #
    # 模式二: 独立执行模块
    # -------------------------------------------------------------
    # 1. 仅执行数据预处理
    #    python -m src.main --preprocess
    #
    # 2. 仅执行训练
    #    a) 自动创建带时间戳的运行名称 (run_YYYYMMDD_HHMMSS)
    #       python -m src.main --train
    #
    #    b) 为本次训练指定一个清晰的名称
    #       python -m src.main --train --run-name "my_gan_experiment"
    #
    # 3. 仅对一个已存在的模型执行推理 (必须提供 --checkpoint)
    #    python -m src.main --inference --checkpoint "my_gan_experiment/model_final.pt"
    #    python -m src.main --inference --checkpoint "run_20250611_111000/model_epoch_50.pt"
    #
    
    main()