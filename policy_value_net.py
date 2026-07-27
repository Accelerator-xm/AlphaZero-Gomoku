import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        # 批量归一化
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, features):
        residual = features
        features = F.relu(self.bn1(self.conv1(features)))
        features = self.bn2(self.conv2(features))
        # 残差连接，防止梯度消失
        return F.relu(features + residual)


class Net(nn.Module):
    """策略价值网络结构"""

    def __init__(self, board_size, channels=64, num_blocks=4):
        """
        channels特征通道数
        num_blocks残差块数量
        """
        super().__init__()
        board_area = board_size ** 2

        # 输入层
        # 接收4通道的输入：当前棋盘状态 (当前玩家盘面、对手盘面、上一轮落子位子、当前玩家信息)
        # 3*3 卷积层：kernel_size=3
        self.input_conv = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # 残差塔：num_blocks个残差块
        # *解包符，列表[Block1, Block2, ...] -> 独立参数Block1, Block2, ...
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_blocks))
        )

        # 策略头
        # 输出每个位置的落子概率
        self.policy_conv = nn.Conv2d(channels, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_area, board_area)

        # 价值头，当前状态的价值
        self.value_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_area, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):

        features = F.relu(
            self.input_bn(self.input_conv(state_input))
        )
        features = self.residual_tower(features)

        # flatten(start_dim=1)展平
        policy = F.relu(
            self.policy_bn(self.policy_conv(features))
        ).flatten(start_dim=1)
        log_action_probs = F.log_softmax(self.policy_fc(policy), dim=1)

        value = F.relu(
            self.value_bn(self.value_conv(features))
        ).flatten(start_dim=1)
        value = F.relu(self.value_fc1(value))
        # tanh 范围[-1, 1]
        value = torch.tanh(self.value_fc2(value))
        return log_action_probs, value


class PolicyValueNet:
    """策略价值网络"""

    def __init__(
        self,
        board_size,
        model_file=None,
        device="cpu",
        l2_const=1e-4,
        channels=64,
        num_blocks=4,
        learn_rate=2e-3,
        lr_multiplier=1.0,
        epochs=5,
        kl_target=0.02,
    ):
        self.board_size = board_size
        self.device = torch.device(device)
        self.l2_const = l2_const    # l2正则化惩罚项、权重衰减参数
        self.learn_rate = learn_rate    # 学习率 
        self.lr_multiplier = lr_multiplier  # 根据 KL 自适应调整学习率
        self.kl_target = kl_target  # KL目标值
        self.epochs = epochs        # 每组数据训练次数
        self.channels = channels    # 残差通道数
        self.num_blocks = num_blocks    # 残差块数
        
        self.policy_value_net = Net(
            board_size,
            channels=self.channels,
            num_blocks=self.num_blocks,
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            weight_decay=self.l2_const,
        )

        if model_file is not None:
            net_params = torch.load(
                model_file,
                map_location=self.device,
                weights_only=True,
            )
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        计算策略和价值 ———— 批量计算
        输入：一批状态
        输出：对应的动作概率和状态价值
        """
        states = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.device)
        
        self.policy_value_net.eval()
        with torch.inference_mode():
            log_act_probs, value = self.policy_value_net(states)
        
        return (
            log_act_probs.exp().cpu().numpy(),
            value.cpu().numpy(),
        )

    def policy_value_fn(self, board):
        """
        计算策略和价值 ———— 单步决策

        输入：棋盘
        输出：所有可选动作的 (动作, 概率) 列表，以及当前局面的价值
        """
        legal_positions = board.availables

        # 输入给网络的维度是 (batch_size, 4, board_size, board_size)
        current_state = np.ascontiguousarray(
            board.current_state().reshape(
                -1, 4, self.board_size, self.board_size
            )
        )
        state = torch.as_tensor(current_state, dtype=torch.float32, device=self.device)
        
        self.policy_value_net.eval()
        with torch.inference_mode():
            log_act_probs, value = self.policy_value_net(state)

        act_probs = log_act_probs.exp().flatten().cpu().numpy()
        # 过滤可落子的位置
        legal_probs = zip(legal_positions, act_probs[legal_positions])
        return legal_probs, value.item()

    def train_step(self, state_batch, mcts_probs, winner_batch):
        """执行一次梯度更新"""
        
        states = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.device)
        # 目标值
        target_probs = torch.as_tensor(np.asarray(mcts_probs), dtype=torch.float32, device=self.device)
        target_values = torch.as_tensor(np.asarray(winner_batch), dtype=torch.float32, device=self.device)

        self.policy_value_net.train()
        # 清空梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learn_rate * self.lr_multiplier

        # 预测值
        log_act_probs, value = self.policy_value_net(states)
        # 损失计算
        # 注意：L2 正则已通过优化器参数体现
        value_loss = F.mse_loss(value.flatten(), target_values) # 均方误差
        policy_loss = -torch.mean(torch.sum(target_probs * log_act_probs, dim=1))   # 交叉熵
        loss = value_loss + policy_loss
        # 反向传播并更新参数
        loss.backward()
        self.optimizer.step()

        # 计算策略熵，仅用于监控
        entropy = -torch.mean(
            torch.sum(log_act_probs.exp() * log_act_probs, dim=1)
        )
        return loss.item(), entropy.item()

    def policy_update(self, state_batch, mcts_probs, winner_batch):
        """训练多个 epoch，并根据 KL 散度自适应调整学习率"""
        old_probs, old_v = self.policy_value(state_batch)

        for _ in range(self.epochs):
            loss, entropy = self.train_step(
                state_batch,
                mcts_probs,
                winner_batch,
            )

            new_probs, new_v = self.policy_value(state_batch)
            
            # 计算KL散度
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
             # 如果 KL 过大则提前停止
            if kl > self.kl_target * 4:
                break
        
        # 自适应调整学习率
        if kl > self.kl_target * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_target / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 解释方差，越接近1越好，越小越差
        winner_var = np.var(winner_batch)
        if winner_var == 0:
            explained_var_old = 0.0
            explained_var_new = 0.0
        else:
            explained_var_old = 1 - np.var(np.asarray(winner_batch) - old_v.flatten()) / winner_var
            explained_var_new = 1 - np.var(np.asarray(winner_batch) - new_v.flatten()) / winner_var


        print(
            "kl:{:.5f}, lr_multiplier:{:.3f}, loss:{:.4f}, "
            "entropy:{:.4f}, explained_var_old:{:.3f}, "
            "explained_var_new:{:.3f}".format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )

        return loss, entropy

    def copy_from(self, other):
        """将另一个同结构网络的参数复制到当前网络"""
        self.policy_value_net.load_state_dict(
            other.policy_value_net.state_dict()
        )

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        """保存模型参数"""
        torch.save(self.get_policy_param(), model_file)
