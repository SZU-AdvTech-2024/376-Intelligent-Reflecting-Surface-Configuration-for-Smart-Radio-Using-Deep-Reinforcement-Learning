# Intelligent Reflecting Surface Configurations for
Smart Radio Using Deep Reinforcement Learning 复现项目

## 概述
本项目复现了 DDQN（双深度 Q 网络）算法，并对其进行了改进。本次开发环境为 Python 3.11.7，项目包含以下五个主要文件：

1. **Main.py**：负责系统初始化以及相关参数的设置。
2. **MuMIMOClass.py**：对 TDD 多用户 MIMO 系统进行环境建模。
3. **DQN.py**：实现了 DDQN 算法。
4. **SAC.py**：实现了 SAC（Soft Actor-Critic）算法。
5. **plot_result.py**：用于展示复现结果。

## 文件说明

### Main.py
该文件是项目的入口，主要负责：
- 设置仿真环境。
- 初始化系统参数。
- 协调训练和评估过程。

### MuMIMOClass.py
该模块对 TDD 多用户 MIMO 系统进行建模，定义了强化学习算法运行的环境，包括：
- 通信设置。
- 状态空间和动作空间的定义。
- 奖励结构。

### DQN.py
包含 DDQN 算法的实现。DDQN 相比传统的 DQN，通过减少 Q 值预测的高估偏差进行改进。主要内容包括：
- Q 函数的神经网络架构。
- 经验回放机制。
- 目标网络更新。

### SAC.py
实现了 SAC 算法，这是一种前沿的强化学习方法，通过离线策略优化随机策略。主要特点包括：
- Actor 和 Critic 网络的设计。
- 稳定性的软更新。
- 平衡探索和利用的熵调整。

### plot_result.py
该脚本用于可视化训练和评估结果，包括：
- 学习曲线。
- DDQN 和 SAC 的性能对比。
- 结果的统计分析。

## 环境要求
请确保已安装 Python 3.11.7 和 pytorch-1.12.1