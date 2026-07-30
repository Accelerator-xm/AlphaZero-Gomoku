"""
多卡训练
"""

import argparse
import os
import queue
import random
import time
import traceback
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from game import Board, Game
from mcts_alphaZero import BatchedMCTSPlayer
from policy_value_net import PolicyValueNet


def get_equi_data(play_data, board_size):
    """通过旋转和翻转将一局数据扩充为 8 倍。"""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # 逆时针旋转
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(
                    mcts_prob.reshape(board_size, board_size)
                ),
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

def _configure_process_device(device):
    """配置设备"""
    torch.set_num_threads(1)    # 限制当前进程只使用1个 CPU 计算线程
    parsed_device = torch.device(device)
    # 绑定设备
    if parsed_device.type == "cuda":
        torch.cuda.set_device(parsed_device)


def _load_published_model(policy_value_net, model_path):
    """加载模型"""
    params = torch.load(
        model_path,
        map_location=policy_value_net.device,
        weights_only=True,
    )
    policy_value_net.policy_value_net.load_state_dict(params)


def _put_with_stop(target_queue, item, stop_event):
    """写队列，并允许主进程要求退出。"""
    while not stop_event.is_set():
        # 主进程还没有要求停止，就继续尝试写入
        try:
            target_queue.put(item, timeout=0.5)
            return True
        except queue.Full:
            # 队列已满循环等待
            continue
    return False


def self_play_actor(
    actor_id,
    device,
    args,
    data_queue,
    error_queue,
    stop_event,
    model_version,
):
    """
    Actor 自博弈进程：
    每次完成一整局，并在局与局之间加载 latest 模型
    """
    try:
        # 初始化设备、随机数、网络、棋盘、玩家
        _configure_process_device(device)
        seed = args.seed + actor_id * 100003
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        policy_value_net = PolicyValueNet(
            args.board_size,
            device=device,
            l2_const=args.l2_const,
            channels=args.channels,
            num_blocks=args.num_blocks,
            learn_rate=args.learn_rate,
            epochs=args.epochs,
            kl_target=args.kl_target,
        )
        board = Board(size=args.board_size, n_in_row=args.n_in_row)
        game = Game(board)
        player = BatchedMCTSPlayer(
            policy_value_net.policy_value_fn_batch,
            c_puct=args.c_puct,
            n_playout=args.playouts,
            is_selfplay=1,
            temp=args.temp,
            leaf_batch_size=args.leaf_batch_size,
            virtual_loss=args.virtual_loss,
        )

        # 循环不断对局
        loaded_version = -1     # 当前加载的模型版本版本
        actor_game_index = 0    # 当前棋局id
        # stop_event共享停止信号，is_set()为True代表结束
        while not stop_event.is_set():
            # 最新模型版本
            published_version = model_version.value
            if published_version != loaded_version:
                # 加载最新模型
                _load_published_model(
                    policy_value_net,
                    args.latest_model,
                )
                loaded_version = published_version

            # 开始自博弈
            _, play_data = game.start_self_play(player)
            play_data = list(play_data)
            actor_game_index += 1   # 下一轮棋局
            message = (
                actor_id,
                actor_game_index,
                loaded_version,
                play_data,
            )
            # 队列容量 = 棋局数
            # 一个数据是一整局的棋局数
            if not _put_with_stop(data_queue, message, stop_event):
                break
    except Exception:
        error_message = (
            actor_id,
            device,
            traceback.format_exc(),
        )
        try:
            error_queue.put(error_message, timeout=1.0)
        except queue.Full:
            pass
        stop_event.set()


class ParallelTrainPipeline:
    """GPU 0 Learner 与多个独立 Self-play Actor 的协调器。"""

    def __init__(self, args, learner_device, actor_devices):
        self.args = args
        self.learner_device = learner_device    # 学习器设备
        self.actor_devices = actor_devices      # 数据采集器设备
        self.data_buffer = deque(maxlen=args.buffer_size)   # 训练数据缓冲区
        # 主进程Learner网络
        self.policy_value_net = PolicyValueNet(
            args.board_size,
            model_file=args.init_model,
            device=learner_device,
            l2_const=args.l2_const,
            channels=args.channels,
            num_blocks=args.num_blocks,
            learn_rate=args.learn_rate,
            epochs=args.epochs,
            kl_target=args.kl_target,
        )
        self.update_count = 0   # 更新轮数，完成多少次policy_update()
        self.last_published_update = -1 # 上一次保存模型对应的id

    def _publish_latest(self, model_version, force=False):
        # 发布最新版本
        # 避免重复：self.update_count == self.last_published_update
        if not force and self.update_count == self.last_published_update:
            return

        target = Path(self.args.latest_model)
        target.parent.mkdir(parents=True, exist_ok=True)
        # 临时文件名
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.tmp"
        )
        try:
            # 保存临时文件
            self.policy_value_net.save_model(str(temporary))
            # 临时文件和目标文件位于同一目录，直接原子替换
            os.replace(temporary, target)
        finally:
            # 清理临时文件
            if temporary.exists():
                temporary.unlink()

        # 更新模型版本
        with model_version.get_lock():
            model_version.value += 1
            published_version = model_version.value
        # 记录当前参数已经发布
        self.last_published_update = self.update_count  
        print(
            f"[Learner] 发布 latest v{published_version}，"
            f"update={self.update_count}，路径={target}",
            flush=True,
        )

    def _policy_train(self):
        """训练策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.args.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        self.policy_value_net.policy_update(
            state_batch,
            mcts_probs_batch,
            winner_batch,
        )
        self.update_count += 1  # 更新次数

    @staticmethod
    def _poll_actor_error(error_queue):
        # 捕获 Actor 主动上报的错误详情
        try:
            actor_id, device, details = error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(
            f"Actor {actor_id}（{device}）异常退出：\n{details}"
        )

    @staticmethod
    def _check_actor_processes(processes):
        """
        检查 Actor 进程
        有些情况下 Actor 可能被操作系统杀死，来不及向 error_queue 写入报错
        因此还要检查进程退出码
        """
        failed = [
            process
            for process in processes
            if process.exitcode is not None
        ]
        if failed:
            summary = ", ".join(
                f"{process.name}: exitcode={process.exitcode}"
                for process in failed
            )
            raise RuntimeError(f"Self-play Actor 提前退出：{summary}")

    def _shutdown_actors(self, processes, stop_event):
        """结束actor，终止进程"""

        # 广播停止信号
        stop_event.set()
        # 设置期限，循环结束
        deadline = time.monotonic() + self.args.shutdown_timeout
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)

        forced = []
        for process in processes:
            if process.is_alive():
                forced.append(process.name)
                # 强制终止
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=2.0)
        if forced:
            print(
                "[Learner] 以下 Actor 未能在当前对局结束前退出，"
                f"已终止：{', '.join(forced)}",
                flush=True,
            )

    def run(self):
        # 创建多进程共享信息
        context = mp.get_context("spawn")
        data_queue = context.Queue(maxsize=self.args.queue_size)
        error_queue = context.Queue(maxsize=max(1, len(self.actor_devices)))
        stop_event = context.Event()
        model_version = context.Value("i", -1)
        processes = []
        games_collected = 0 # 已接收的总棋局数
        pending_train_steps = 0 # 还有多少次网络更新没有执行
        start_time = time.perf_counter()
        interrupted = False

        try:
            # 发布初始化版本参数，版本号0
            self._publish_latest(model_version, force=True)

            # 开启并行进程采集数据
            for actor_id, device in enumerate(self.actor_devices, start=1):
                process = context.Process(
                    target=self_play_actor,
                    name=f"self-play-actor-{actor_id}",
                    args=(
                        actor_id,
                        device,
                        self.args,
                        data_queue,
                        error_queue,
                        stop_event,
                        model_version,
                    ),
                )
                process.start()
                processes.append(process)
                print(
                    f"[Learner] 启动 Actor {actor_id} -> {device}",
                    flush=True,
                )

            # 主训练循环
            # 还没有收够目标棋局、训练步骤没完成
            while (
                games_collected < self.args.game_batch_num
                or pending_train_steps > 0
            ):
                # 检查错误
                self._poll_actor_error(error_queue)

                # 接收数据
                received_game = False
                if games_collected < self.args.game_batch_num:
                    try:
                        # 超时时间
                        timeout = 0.05 if pending_train_steps else 0.5
                        # 取队头数据，一次取一个对局数据
                        message = data_queue.get(timeout=timeout)
                        received_game = True
                    except queue.Empty:
                        # 检查actor存活
                        self._check_actor_processes(processes)

                if received_game:
                    # 已经接收一局数据
                    (
                        actor_id,
                        actor_game_index,
                        actor_model_version,
                        play_data,
                    ) = message
                    games_collected += 1    # 收集一次
                    # 数据增强
                    augmented = get_equi_data(play_data, self.args.board_size)
                    # 添加进缓冲区
                    self.data_buffer.extend(augmented)
                    # 有warmup_samples个数据才开始训练
                    if len(self.data_buffer) >= self.args.warmup_samples:
                        # 每收集一局数据增加训练次数
                        pending_train_steps += self.args.train_steps_per_game

                    # 第一局、达到日志输出频率、最终局
                    # 输出训练日志
                    if (
                        games_collected == 1
                        or games_collected % self.args.log_freq == 0
                        or games_collected == self.args.game_batch_num
                    ):
                        elapsed = time.perf_counter() - start_time
                        games_per_hour = (
                            games_collected / elapsed * 3600
                            if elapsed
                            else 0.0
                        )
                        print(
                            "[Learner] games={}/{}, actor={}#{}, "
                            "actor_model=v{}, moves={}, replay={}, "
                            "updates={}, speed={:.1f} games/h, elapsed={}".format(
                                games_collected,
                                self.args.game_batch_num,
                                actor_id,
                                actor_game_index,
                                actor_model_version,
                                len(play_data),
                                len(self.data_buffer),
                                self.update_count,
                                games_per_hour,
                                timedelta(seconds=int(elapsed)),
                            ),
                            flush=True,
                        )

                # 执行网络更新
                # 有待执行训练任务 并且 回放池中至少有一个 batch 的数据
                if (
                    pending_train_steps > 0
                    and len(self.data_buffer) >= self.args.batch_size
                ):
                    # 训练 pending_train_steps
                    self._policy_train()
                    pending_train_steps -= 1
                    # 保存参数
                    if self.update_count % self.args.publish_freq == 0:
                        self._publish_latest(model_version)

            self._publish_latest(model_version)

        except KeyboardInterrupt:
            interrupted = True
            print("\n[Learner] 收到停止请求，正在保存并关闭 Actor", flush=True)
            self._publish_latest(model_version)
        except Exception:
            # Actor 或协调逻辑异常时也保留最近一次已经完成的 Learner 更新。
            if self.update_count != self.last_published_update:
                try:
                    self._publish_latest(model_version)
                except Exception as publish_error:
                    print(
                        f"[Learner] 异常退出时保存 latest 失败：{publish_error}",
                        flush=True,
                    )
            raise
        finally:
            # 清理资源
            self._shutdown_actors(processes, stop_event)
            data_queue.cancel_join_thread()
            error_queue.cancel_join_thread()
            data_queue.close()
            error_queue.close()

        if interrupted:
            return
        elapsed = time.perf_counter() - start_time
        print(
            "[Learner] 并行训练完成：games={}，updates={}，总耗时={}".format(
                games_collected,
                self.update_count,
                timedelta(seconds=int(elapsed)),
            ),
            flush=True,
        )


def _resolve_devices(args):
    """分配设备"""
    # 获取设备列表
    if args.devices:
        devices = [
            device.strip()
            for device in args.devices.split(",")
            if device.strip()
        ]
    else:
        available = torch.cuda.device_count()
        gpu_count = args.num_gpus if args.num_gpus is not None else available
        if gpu_count > available:
            raise ValueError(
                f"请求 {gpu_count} 张 GPU，但 PyTorch 只检测到 {available} 张"
            )
        devices = [f"cuda:{index}" for index in range(gpu_count)]

    if len(devices) < 2:
        raise ValueError(
            "并行训练至少需要 2 个设备（1 个 Learner + 1 个 Actor）；"
            "例如 --num-gpus 8。CPU 冒烟测试可使用 --devices cpu,cpu"
        )

    # 转为 torch.device 对象
    # 参数解析和合法性检查
    parsed = [torch.device(device) for device in devices]

    # 转会字符串
    return str(parsed[0]), [str(device) for device in parsed[1:]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "多 GPU AlphaZero 训练：第一个设备运行 Learner，"
            "其余设备各运行一个批量 MCTS Self-play Actor"
        )
    )
    parser.add_argument("--init-model", help="Learner 继续训练时加载的模型")
    parser.add_argument("--latest-model", default="current_policy.model", help="模型保存路径")
    parser.add_argument("--num-gpus", type=int, help="使用前 N 张可见 GPU")
    parser.add_argument(
        "--devices",
        help=(
            "显式设置设备列表，覆盖 --num-gpus；第一个是 Learner，"
            "其余是 Actor，例如 cuda:0,cuda:2,cuda:3"
        )
    )
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--n-in-row", type=int, default=4)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--playouts", type=int, default=500)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--leaf-batch-size", type=int, default=16)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--warmup-samples", type=int, default=512, help="回放池达到该样本数后 Learner 才开始更新")
    parser.add_argument("--game-batch-num", dest="game_batch_num", type=int, default=1200, help="Learner 接收的自博弈总局数")
    parser.add_argument("--train-steps-per-game", type=int, default=1, help="每收到一局新数据执行的 Learner 更新次数")
    parser.add_argument("--batch-size", type=int, default=512, help="训练批量大小")
    parser.add_argument("--epochs", type=int, default=5, help="每批数据训练轮数")
    parser.add_argument("--learn-rate", type=float, default=2e-3, help="基础学习率")
    parser.add_argument("--kl-target", type=float, default=0.02, help="KL 散度目标值")
    parser.add_argument("--publish-freq", type=int, default=10, help="模型保存频率")
    parser.add_argument("--l2-const", type=float, default=1e-4, help="L2 权重衰减系数")
    parser.add_argument("--channels", type=int, default=64, help="残差网络特征通道数")
    parser.add_argument("--num-blocks", type=int, default=2, help="残差块数量")
    parser.add_argument("--queue-size", type=int, default=32, help="完整对局队列容量")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shutdown-timeout", type=float, default=10.0, help="等待 Actor 完成本局并退出的总秒数")
    return parser.parse_args()


def main(args):
    """使用给定配置启动并行训练。"""
    learner_device, actor_devices = _resolve_devices(args)
    # resolve()绝对路径
    args.latest_model = str(Path(args.latest_model).resolve())
    if args.init_model:
        args.init_model = str(Path(args.init_model).resolve())

    print(
        f"设备布局：Learner={learner_device}，"
        f"Actors={', '.join(actor_devices)}",
        flush=True,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _configure_process_device(learner_device)
    pipeline = ParallelTrainPipeline(
        args,
        learner_device,
        actor_devices,
    )
    pipeline.run()


if __name__ == "__main__":
    main(parse_args())
