import argparse
import random
import time
from collections import defaultdict, deque
from datetime import timedelta
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class TrainPipeline:
    """训练流程"""

    def __init__(self, args):
        self.board_width = args.width
        self.board_height = args.height
        self.n_in_row = args.n_in_row
        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row,
        )
        self.game = Game(self.board)

        self.n_playout = args.playouts  # MCTS搜索模拟次数
        self.c_puct = args.c_puct       # MCTS 探索系数
        self.play_batch_size = args.play_batch_size # 每批自博弈局数
        self.buffer_size = args.buffer_size # 缓冲区大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = args.batch_size   # 训练批次大小
        self.game_batch_num = args.game_batch_num   # 训练轮数
        self.check_freq = args.check_freq   # 评测频率
        self.evaluate_games = args.evaluate_games   # 一次评估测试轮数
        self.update_threshold = args.update_threshold   # best权重更新阈值
        self.current_model = args.current_model # 当前模型存储位置
        self.best_model = args.best_model       # 最好模型存储位置

        # 训练网络负责梯度更新
        # best 网络固定用于生成自博弈数据和充当评估基准。
        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=args.init_model,
            device=args.device,
            l2_const=args.l2_const,
            channels=args.channels,
            num_blocks=args.num_blocks,
            learn_rate=args.learn_rate,
            epochs=args.epochs,
            kl_target=args.kl_target,
        )
        self.best_policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            device=args.device,
            l2_const=args.l2_const,
            channels=args.channels,
            num_blocks=args.num_blocks,
            learn_rate=args.learn_rate,
            epochs=args.epochs,
            kl_target=args.kl_target,
        )
        self.best_policy_value_net.copy_from(self.policy_value_net)
        
        # 自博弈玩家
        self.selfplay_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
            temp=args.temp,
        )
        self.episode_len = 0

    def get_equi_data(self, play_data):
        """通过旋转和翻转扩充数据集"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)),
                    i
                )
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )

                # 水平翻转
                flipped_state = np.array([np.fliplr(s) for s in equi_state])
                flipped_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (flipped_state, np.flipud(flipped_mcts_prob).flatten(), winner)
                )
        return extend_data

    def collect_selfplay_data(self, n_games):
        """收集自我对弈数据用于训练"""
        for _ in range(n_games):
            _, play_data = self.game.start_self_play(self.selfplay_player)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            self.data_buffer.extend(self.get_equi_data(play_data))

    def policy_train(self):
        """训练策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        return self.policy_value_net.policy_update(
            state_batch,
            mcts_probs_batch,
            winner_batch,
        )

    def policy_evaluate(self):
        """与 best 网络对战，评估当前策略"""
        current_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
        )
        best_player = MCTSPlayer(
            self.best_policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
        )
        result_count = defaultdict(int)
        for i in range(self.evaluate_games):
            winner = self.game.start_play(
                current_player,
                best_player,
                start_player=i % 2,
                is_shown=0,
            )
            result_count[winner] += 1

        win_ratio = result_count[1] / self.evaluate_games
        print(
            "候选网络 vs best：胜 {}，负 {}，平 {}，胜率 {:.2%}".format(
                result_count[1],
                result_count[2],
                result_count[-1],
                win_ratio,
            )
        )
        return win_ratio

    def update_best_policy(self):
        """复制训练网络参数到 best 网络，并保存 best 模型"""
        self.best_policy_value_net.copy_from(self.policy_value_net)
        self.best_policy_value_net.save_model(self.best_model)

    def run(self):
        """运行完整训练流程"""
        training_start_time = time.perf_counter()

        for i in range(1, self.game_batch_num + 1):
            self.collect_selfplay_data(self.play_batch_size)
            
            print(
                "批次：{}，本局步数：{}，回放池样本数：{}".format(
                    i,
                    self.episode_len,
                    len(self.data_buffer),
                )
            )

            if len(self.data_buffer) >= self.batch_size:
                self.policy_train()

            elapsed_seconds = time.perf_counter() - training_start_time
            average_batch_seconds = elapsed_seconds / i
            remaining_seconds = average_batch_seconds * (self.game_batch_num - i)
            estimated_total_seconds = elapsed_seconds + remaining_seconds
            print(
                "训练时间：已用 {}，预计剩余 {}，预计总耗时 {}".format(
                    timedelta(seconds=int(elapsed_seconds)),
                    timedelta(seconds=int(remaining_seconds)),
                    timedelta(seconds=int(estimated_total_seconds)),
                )
            )

            if i % self.check_freq == 0:
                self.policy_value_net.save_model(self.current_model)
                win_ratio = self.policy_evaluate()
                if win_ratio > self.update_threshold:
                    print(
                        "候选网络胜率 {:.2%} 超过阈值 {:.2%}，更新 best".format(
                            win_ratio,
                            self.update_threshold,
                        )
                    )
                    self.update_best_policy()
                else:
                    print(
                        "候选网络胜率 {:.2%} 未超过阈值 {:.2%}，保留 best".format(
                            win_ratio,
                            self.update_threshold,
                        )
                    )


def parse_args():
    parser = argparse.ArgumentParser(description="训练 AlphaZero")
    parser.add_argument("--init-model", help="继续训练时加载的初始模型路径")
    parser.add_argument("--width", type=int, default=6, help="棋盘宽度")
    parser.add_argument("--height", type=int, default=6, help="棋盘高度")
    parser.add_argument("--n-in-row", type=int, default=4, help="获胜所需连子数")
    parser.add_argument("--learn-rate", type=float, default=2e-3, help="基础学习率")
    parser.add_argument("--temp", type=float, default=1.0, help="自博弈采样温度")
    parser.add_argument("--playouts", type=int, default=500, help="每步 MCTS 模拟次数")
    parser.add_argument("--c-puct", type=float, default=5.0, help="MCTS 探索系数")
    parser.add_argument("--buffer-size", type=int, default=10000, help="经验回放池容量")
    parser.add_argument("--batch-size", type=int, default=512, help="训练批量大小")
    parser.add_argument("--play-batch-size", type=int, default=1, help="每批自博弈局数")
    parser.add_argument("--epochs", type=int, default=5, help="每批数据训练轮数")
    parser.add_argument("--kl-target", type=float, default=0.02, help="KL 散度目标值")
    parser.add_argument("--check-freq", type=int, default=50, help="模型评估间隔批次")
    parser.add_argument("--game-batch-num", type=int, default=1200, help="总训练批次数")
    parser.add_argument("--l2-const", type=float, default=1e-4, help="L2 权重衰减系数")
    parser.add_argument("--channels", type=int, default=64, help="残差网络特征通道数")
    parser.add_argument("--num-blocks", type=int, default=2, help="残差块数量")
    parser.add_argument("--evaluate-games", type=int, default=20, help="每次候选网络与 best 网络的评估对局数")
    parser.add_argument("--update-threshold", type=float, default=0.55, help="更新 best 所需的候选网络胜率，必须严格超过该值")
    parser.add_argument("--device", default="cpu", help="计算设备，例如 cpu 或 cuda:0")
    parser.add_argument("--current-model", default="current_policy.model", help="候选模型保存路径")
    parser.add_argument("--best-model", default="best_policy.model", help="best 模型保存路径")
    return parser.parse_args()


def main():
    args = parse_args()
    TrainPipeline(args).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练已停止")
