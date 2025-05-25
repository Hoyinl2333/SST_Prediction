# SST Prediction

一个用于预测海表温度（SST, Sea Surface Temperature）的深度学习项目

## 项目结构

SSTPredictionProject/
│
├── data/ # 数据存放目录（已被 git 忽略，保留空目录）
│ └── .gitkeep
│
├── notebooks/ # Jupyter 分析笔记本
├── src/ # 训练代码（模型、数据集、训练脚本等）
├── scripts/ # 启动脚本（如训练、预测）
├── configs/ # 配置文件（如 YAML 格式）
├── checkpoints/ # 模型保存路径
├── requirements.txt # Python 依赖
├── .gitignore # Git 忽略规则
└── README.md # 项目说明

## 快速开始

1. 克隆仓库
2. 安装依赖
```
pip intall -r requirements.txt
```
3. 运行训练脚本(还没实现)
```
bash scripts/run_train.sh
```