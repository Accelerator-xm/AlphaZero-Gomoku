# 论文3 AlphaZero

> **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm** Science 2018
>
> 只给游戏规则，不给人类知识，通过自我博弈 + 深度强化学习 + 蒙特卡洛树搜索，自动学习出超越人类和传统程序的棋类智能体

---

## 解决的问题

传统棋类AI的问题：使用大量人工设计特征
- 例如国际象棋Stockfish的人工设计特征：棋子价值、王安全、兵结构、活跃度等
- 传统流程：人类知识 -> 游戏规则 -> 评价函数 -> 搜索

AlphaZero目标：用统一算法解决多个领域，而不是针对某个游戏设计特殊技巧
- 流程：游戏规则 -> 随机神经网络 -> 自己和自己下 -> 棋强化学习 -> 越来越强

---

## AlphaZero 架构

与论文AlphaGo Zero几乎相同，需要先看AlphaGo Zero论文，这里基本只给出与AlphaGo Zero的**不同点**

- **整体架构**：
  - 从棋盘状态到最终动作：

    ```math
    s
    \rightarrow f_\theta(s) = (p,v)
    \rightarrow \operatorname{MCTS}
    \rightarrow \pi(a \mid s)
    ```

- **输入设计**：由多个棋盘平面组成：

  ```math
  N \times N \times (MT+L)
  ```

  - $N$：棋盘尺寸
  - $M$：棋子特征，例如围棋黑白2个，国际象棋12个（王、后、车、象、马、兵，黑白两方）
  - $T$：历史数量，例如围棋8个历史状态，黑白均需要记录八个盘面，所以是 $M T$ 个
  - $L$：额外规则信息，例如围棋中当前执棋方需要1个特征平面

- **共享残差主干**

- **Policy Head策略头**
  - 围棋和五子棋的动作很简单：落子位置（$N \times N$ 即可），围棋多一个pass
  - 国际象棋 $8×8×73=4672$
    - 56个“类后移动”平面：8个方向 × 1～7格
    - 8个马步平面
    - 9个兵的欠升变平面
  - 将棋 $9×9×139=11259$

- **Value Head价值头**：输出范围为 $[-1,1]$
  - **获胜概率**推广为**期望结果**：其他棋类有平局， $z \in \{-1, 0, +1\}$，语义上不是胜率，而是最终比赛结果的期望值

- **MCTS搜索**：选择、扩展/评估、回传、动作选择

---

## 强化学习训练

- **博弈数据采集**
  - 对于国际象棋和将棋不能采用对称变换扩展数据，围棋和五子棋可以
  - AlphaZero 不进行训练数据对称增强，也不在 MCTS 中变换棋盘

- **损失函数**：

  ```math
  L(\theta)
  =
  (z-v)^2
  -
  \pi^\top \log p
  +
  c\lVert\theta\rVert^2
  ```

- **训练流程简化和在线化**
  - **Self-play Generator**
    - 最新网络参数 **自己与自己下棋**，而不是最优模型
  - **Optimizer**
  - 不再要求 ~~**Evaluator**~~
    - 即：

      ```math
      \theta_0
      \rightarrow \theta_1
      \rightarrow \theta_2
      \rightarrow \cdots
      ```

    - 省掉大量候选模型对战
    - 训练流水线更简单
    - 避免模型因为没有超过55%阈值而长时间停留在旧版本
    - 可能产生策略振荡或灾难性遗忘

- 训练配置
  - 每个 batch 4096 个局面
  - 每步执行 800 次 MCTS
