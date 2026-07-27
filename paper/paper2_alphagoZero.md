# 论文2 AlphagoZero

> **Mastering the game of Go without human knowledge** Nature 2017
>
> 不使用人类数据，只依靠自博弈强化学习实现人工智能下围棋
>
> **核心思想**：神经网络指导搜索，搜索反过来训练神经网络，形成了一个不断**自我增强的闭环**

---

## 解决的问题

早期 AlphaGo 依赖：

- 人类职业棋谱进行监督学习
- 人工设计的围棋特征
- 单独的策略网络和价值网络
- 快速 rollout 策略进行完整棋局模拟
- 强化学习继续微调策略

AlphaGo Zero 关键简化：

- 从随机参数开始
- 只输入棋子和历史状态
- 策略网络和价值网络策略和价值合并到一个网络：共享主干、两个输出头
- 全程自博弈强化学习

复杂的多阶段系统统一：一个神经网络+一个树搜索+一个自博弈循环

---

## AlphaGo Zero 架构

### 整体算法

从强化学习角度看，AlphaGo Zero 是一种**近似策略迭代**

经典策略迭代包含两个步骤：

- 策略评估：评估当前策略有多好
- 策略改进：根据价值函数构造更好的策略

AlphaGo Zero：

- MCTS 根据网络给出的 $(p,v)$ 生成更强的搜索策略 $\pi$，相当于策略改进
- 使用搜索策略进行自博弈，最终胜负 $z$ 用来评估该策略
- 再把 $\pi$ 和 $z$ 压缩回神经网络

```math
p_\theta
\xrightarrow{\text{MCTS 改进}}
\pi
\xrightarrow{\text{自博弈评估}}
z
\xrightarrow{\text{监督训练}}
\left(p_{\theta'},v_{\theta'}\right)
```

**注意**：$\pi$ 通常比 $p_\theta$ 更强，因为 $p_\theta$ 只经过一次神经网络前向传播，而 $\pi$ 综合了大量树搜索模拟结果。因此，MCTS 实际上充当了神经网络的“老师”

### 网络设计

#### 输入设计：

相比alphago去除人工设计的特征，**只保留棋子和历史状态**

- 规模17个特征平面，即19×19×17
  - 8个回合的历史局面（执棋方和对手共16平面）
  - 棋手特征：全1或全0，表示执棋方

#### 共享残差主干：

- 首卷积模块，提取输入特征

  ```math
  19 \times 19 \times 17
  \rightarrow
  \mathrm{Conv}_{3 \times 3}(256)
  ```

- 随后经过n个残差块
  - 每个残差块：

    ```math
    x
      \rightarrow \mathrm{Conv}_{3 \times 3}(256)
      \rightarrow \mathrm{BN}
      \rightarrow \mathrm{ReLU}
      \rightarrow \mathrm{Conv}_{3 \times 3}(256)
      \rightarrow \mathrm{BN}
    \rightarrow +x
      \rightarrow \mathrm{ReLU}
    ```

  - $+x$：表示**残差连接**或**跳跃连接**，$F(x) + x$
  - 论文中：
    - 20-block 网络表示1个首卷积模块+19个残差块，约2280万参数
    - 40-block 网络表示39个残差块，约4640 万参数

残差网络的优势：

- 普通深层卷积网络容易出现：梯度消失、训练退化、层数增加但性能反而下降
- 残差连接学习的是 $F(x) = H(x) - x$，输出 $H(x) = F(x) + x$，梯度可以沿跳跃连接传播，使几十层网络仍能稳定训练
- **实验表明**：使用残差网络比普通卷积网络提高超过约 600 Elo

共享主干优势：

- 共享计算：一次主干前向传播同时产生 $p$ 和 $v$
- 共享表示：落子选择和胜负判断都依赖棋形、连接等共同特征；
- 多任务正则化：策略任务和价值任务互相约束，降低过拟合
- **实验表明**：
  - 将策略和价值合并到一个网络中提高超过约 600 Elo
  - 虽然共享网络的人类落子预测准确率略低，但实际棋力更高、价值误差更低

#### Policy Head策略头：

- 网络结构：

  ```math
  19 \times 19 \times 256
  \rightarrow \mathrm{Conv}_{1 \times 1}(2)
  \rightarrow \mathrm{reshape}(19 \times 19 \times 2)
  \rightarrow \mathrm{FC}(362)
  \rightarrow \mathrm{Softmax}
  ```

  - $\text{reshape}(19×19×2)$ 表示把三维图像转化成一维向量
  - 362输出：361个棋盘交叉点、1个pass（弃权本轮下棋）
  - 含义：在当前局面下，每个合法动作作为候选的**先验概率**，并不是最终直接使用的动作策略

#### Value Head价值头：

- 网络结构：

  ```math
  19 \times 19 \times 256
  \rightarrow \mathrm{Conv}_{1 \times 1}(1)
  \rightarrow \mathrm{reshape}(19 \times 19 \times 1)
  \rightarrow \mathrm{FC}(256)
  \rightarrow \mathrm{FC}(1)
  \rightarrow \tanh
  ```

  - 输出：$v \in [-1, 1]$，期望胜负
  - 胜率可以近似为：

    ```math
    P(\mathrm{win} \mid s) = \frac{v+1}{2}
    ```

### MCTS 的架构设计

每条搜索树边 $(s,a)$ 存储四个量：

- $P(s,a)$：策略网络给出的先验概率 
- $N(s,a)$：该动作的访问次数 
- $W(s,a)$：累计价值
- $Q(s,a)$：平均动作价值，$Q = W/N$

AlphaGo论文已经详细介绍了MCTS搜索，这里仅介绍区别

#### Selection选择：同ALphaGo论文

#### Expand and Evaluate：扩展和评估

- 当搜索到一个尚未展开的叶节点 $s_L$，只调用一次神经网络：

  ```math
  \left(P(s_L,\cdot),v\right) = f_\theta(s_L)
  ```

  - $P$ 初始化新节点所有动作的先验
  - $v$ 用作该叶节点的价值
  - 不再从该节点随机模拟到终局获取 $z$

#### Backup回传：同ALphaGo论文

#### Play最终动作选择：

- 构造：

  ```math
  \pi(a \mid s)
  =
  \frac{N(s,a)^{1/\tau}}
       {\sum_b N(s,b)^{1/\tau}}
  ```

  - $\tau$ 温度参数
    - $\tau=1$：按访问次数比例采样，探索性较强
    - $\tau=0$：近似选择访问次数最多的动作（AlphaGo的方法）

---

## 强化学习训练

### 训练及配置

#### 博弈数据采集

每个局面保存一个训练样本：$(s_t, \pi_t, z_t)$

- $s_t$：局面
- $\pi_t$：该局面经过 MCTS 后的访问次数分布
- $z_t$：整盘结束后，从时刻 $t$ 当前玩家视角得到的最终胜负
- 八种对称变换扩展数据

#### 损失函数

- **损失函数**：

  ```math
  \begin{aligned}
  L(\theta)
  &=
  (z-v)^2
  -
  \pi^\top \log p
  +
  c\lVert\theta\rVert^2, \\
  L_v &= (z-v)^2, \\
  L_p &= -\pi^\top\log p, \\
  L_{\mathrm{reg}} &= c\lVert\theta\rVert^2
  \end{aligned}
  ```

  - **价值损失**：均方误差
  - **策略损失**：交叉熵
    - $\top$ 表示将这个列向量（或行向量）进行转置
  - **L2正则化**：用于抑制过拟合
    - 论文设置：$c = 10^{-4}$

#### 完整训练流水线

三个异步并行模块

1. **Self-play Generator**

   当前最优模型 $\theta^*$ **自己与自己下棋**

   - 每一步：
     - 执行 1600 次 MCTS 模拟
     - 得到访问次数分布 $\pi_t$，选择动作
     - 保存 $(s_t, \pi_t)$，游戏结束后补上 $z_t$
   - 增加探索：
     - 前 30 手设置 $\tau = 1$
     - 30 手之后设置 $\tau \rightarrow 0$
     - 根节点先验加入 Dirichlet 噪声：

       ```math
       \begin{aligned}
       P'(s,a) &= (1-\varepsilon)p_a + \varepsilon\eta_a, \\
      \eta &\sim \mathrm{Dir}(0.03),
       \qquad \varepsilon = 0.25
       \end{aligned}
       ```

       - 防止所有自博弈对局快速收敛到同一种开局

2. **Optimizer**

   优化器从最近 50 万盘自博弈数据的所有局面中均匀采样

   - 论文配置：
     - 64 个 GPU worker
     - 19 个 CPU parameter server
     - 每个 worker batch size 为 32
     - SGD + momentum；
     - momentum =0.9；
     - 每 1000 次梯度更新生成一个 checkpoint。

3. **Evaluator**

   每个新 checkpoint（候选模型）与当前最优模型进行 400 盘对局，只有候选模型胜率超过 55%，它才会成为新的最优模型：

   ```math
   \theta^* \leftarrow \theta_{\mathrm{candidate}}
   ```

   - 对局设置：
     - 每步 1600 次 MCTS
     - $\tau \rightarrow 0$，使用确定性的最强下法

#### 训练配置

- 3 天版本
  - 20-block 网络
  - 生成 490 万盘自博弈
  - 70 万个 mini-batch
  - 每个 batch 2048 个局面
  - 每步执行 1600 次 MCTS
- 40 天版本
  - 40-block 网络
  - 生成 2900 万盘自博弈
  - 310 万个 mini-batch
  - 每个 batch 2048 个局面

### 性能结果：

- Elo指标
  - AlphaGo Zero 5185，AlphaGo Master 4858，AlphaGo Lee 3739
    - 3 天版本对 AlphaGo Lee：100:0
    - 40 天版本对 AlphaGo Master：89:11
  - AlphaGo Zero 原始网络，不使用搜索 3055
    - 说明神经网络本身已经很强，而 MCTS 又把它提升到了完全不同的棋力层级
- Top-1 准确率：
  - AlphaGo Zero 低于 AlphaGo
  - 说明：预测人类落子准确 $\neq$ 最终胜率最高
- 价值网络 MSE：
  - 40-block 与 20-block 差不多
  - 离线价值误差也不是最终棋力的完美代理指标

### 对比 Actor-Critic 算法

> 基于神经网络和树搜索的近似策略迭代

| 维度 | 标准 Actor-Critic | AlphaGo Zero |
| :--- | :--- | :--- |
| **Actor (策略)** | 通常通过策略梯度优化 | Policy Head 用交叉熵模仿 MCTS ($\pi$) |
| **Critic (价值)** | 常使用 TD / Bellman 目标 | Value Head 使用终局胜负 $z$ |
| **行为策略** | 主要来自 Actor 自身 | 行为策略来自 MCTS 的 $\pi$ |
| **交互关系** | Critic 直接指导 Actor 更新 | Value 先指导搜索，搜索再生成策略目标 |
| **环境模型** | 通常不需要显式模型 | MCTS 需要完整游戏规则和状态转移 |

---
