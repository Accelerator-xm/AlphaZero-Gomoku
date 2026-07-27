"""使用 Tkinter 实现的 AlphaZero 五子棋人机对战界面。

本文件是独立的 GUI 入口，只调用现有的棋盘、MCTS 和策略价值网络，
不会修改或介入 ``train.py`` 与 ``human_play.py`` 的执行流程。
"""

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class GomokuGUI:
    """管理窗口绘制、点击落子以及人机回合切换。"""

    # 界面配色
    BOARD_COLOR = "#D9A85F"
    GRID_COLOR = "#4B3423"
    BLACK_COLOR = "#191919"
    WHITE_COLOR = "#F4F4F4"
    LAST_MOVE_COLOR = "#D83A34"

    def __init__(self, root, args, policy):
        self.root = root
        self.args = args
        self.policy = policy
        self.board = None
        self.ai_player = None
        self.human_player_id = None
        self.ai_player_id = None
        self.game_over = False
        self.ai_thinking = False
        # 每次重新开始都会递增，用于识别并丢弃上一局尚未结束的 AI 计算结果。
        self.game_id = 0
        # 后台线程只向队列写入结果，所有 Tkinter 操作仍由主线程执行。
        self.ai_results = queue.Queue()

        self.root.title("AlphaZero 五子棋")
        self.root.minsize(520, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        controls = tk.Frame(root, padx=12, pady=10)
        controls.pack(fill=tk.X)

        self.human_first = tk.BooleanVar(value=args.human_first)
        tk.Radiobutton(
            controls,
            text="我先手（黑棋）",
            variable=self.human_first,
            value=True,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            controls,
            text="AI 先手（黑棋）",
            variable=self.human_first,
            value=False,
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(
            controls,
            text="重新开始",
            command=self.new_game,
            padx=12,
        ).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            root,
            background=self.BOARD_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        # 鼠标左键负责落子，窗口缩放时重新计算棋盘尺寸并绘制。
        self.canvas.bind("<Button-1>", self.on_board_click)
        self.canvas.bind("<Configure>", lambda _event: self.draw_board())

        self.status = tk.StringVar()
        tk.Label(
            root,
            textvariable=self.status,
            anchor=tk.W,
            padx=14,
            pady=9,
        ).pack(fill=tk.X)

        self.new_game()
        self.root.after(50, self.process_ai_results)

    def new_game(self):
        """创建新棋盘，并根据界面选项分配黑白棋与先后手。"""
        self.game_id += 1
        self.board = Board(
            size=self.args.board_size,
            n_in_row=self.args.n_in_row,
        )
        # Board 中玩家 1 固定先行，因此玩家 1 在 GUI 中对应黑棋。
        self.board.init_board(start_player=0)
        black, white = self.board.players

        if self.human_first.get():
            self.human_player_id, self.ai_player_id = black, white
        else:
            self.ai_player_id, self.human_player_id = black, white

        # 每局使用一棵全新的 MCTS 搜索树，策略网络参数则继续复用。
        self.ai_player = MCTSPlayer(
            self.policy.policy_value_fn,
            c_puct=5,
            n_playout=self.args.playouts,
            temp=self.args.temp,
        )
        self.ai_player.set_player_ind(self.ai_player_id)
        self.game_over = False
        self.ai_thinking = False
        self.draw_board()

        if self.board.current_player == self.ai_player_id:
            self.root.after(100, self.start_ai_turn)
        else:
            self.status.set("轮到你落子：点击棋盘交叉点")

    def board_geometry(self):
        """根据画布大小计算棋盘左上角坐标和网格间距。"""
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        margin = 42
        usable_width = max(canvas_width - 2 * margin, 1)
        usable_height = max(canvas_height - 2 * margin, 1)
        steps = max(self.args.board_size - 1, 1)
        spacing = min(usable_width / steps, usable_height / steps)
        board_pixel_size = spacing * steps
        origin_x = (canvas_width - board_pixel_size) / 2
        origin_y = (canvas_height - board_pixel_size) / 2
        return origin_x, origin_y, spacing

    def draw_board(self):
        """重绘网格、坐标、全部棋子以及最后一步标记。"""
        if self.board is None:
            return

        self.canvas.delete("all")
        origin_x, origin_y, spacing = self.board_geometry()
        right = origin_x + spacing * (self.args.board_size - 1)
        bottom = origin_y + spacing * (self.args.board_size - 1)

        for col in range(self.args.board_size):
            x = origin_x + col * spacing
            self.canvas.create_line(
                x, origin_y, x, bottom, fill=self.GRID_COLOR, width=2
            )
            self.canvas.create_text(
                x,
                origin_y - min(22, spacing * 0.45),
                text=str(col),
                fill=self.GRID_COLOR,
            )

        for row in range(self.args.board_size):
            y = origin_y + row * spacing
            self.canvas.create_line(
                origin_x, y, right, y, fill=self.GRID_COLOR, width=2
            )
            self.canvas.create_text(
                origin_x - min(22, spacing * 0.45),
                y,
                text=str(row),
                fill=self.GRID_COLOR,
            )

        radius = max(5, min(spacing * 0.39, 24))
        for move, player in self.board.states.items():
            row, col = self.board.move_to_location(move)
            x = origin_x + col * spacing
            y = origin_y + row * spacing
            fill = self.BLACK_COLOR if player == self.board.players[0] else self.WHITE_COLOR
            outline = "#050505" if player == self.board.players[0] else "#777777"
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline=outline,
                width=2,
            )
            if move == self.board.last_move:
                marker_radius = max(2, radius * 0.16)
                self.canvas.create_oval(
                    x - marker_radius,
                    y - marker_radius,
                    x + marker_radius,
                    y + marker_radius,
                    fill=self.LAST_MOVE_COLOR,
                    outline="",
                )

    def on_board_click(self, event):
        """把鼠标点击转换为棋盘坐标，并提交一次人类落子。"""
        # 对局结束、AI 思考中或并非人类回合时，不接受点击。
        if (
            self.game_over
            or self.ai_thinking
            or self.board.current_player != self.human_player_id
        ):
            return

        origin_x, origin_y, spacing = self.board_geometry()
        # 找到距离点击位置最近的交叉点。
        col = round((event.x - origin_x) / spacing)
        row = round((event.y - origin_y) / spacing)
        if not (
            0 <= row < self.args.board_size
            and 0 <= col < self.args.board_size
        ):
            return

        x = origin_x + col * spacing
        y = origin_y + row * spacing
        if abs(event.x - x) > spacing * 0.45 or abs(event.y - y) > spacing * 0.45:
            return

        move = self.board.location_to_move([row, col])
        if move not in self.board.availables:
            self.status.set("这个位置已经有棋子了，请选择其他位置")
            return

        self.board.do_move(move)
        self.draw_board()
        # 人类落子未结束对局时，立即进入 AI 回合。
        if not self.finish_if_needed():
            self.start_ai_turn()

    def start_ai_turn(self):
        """启动后台线程计算 AI 落子，避免 MCTS 阻塞界面事件循环。"""
        if (
            self.game_over
            or self.ai_thinking
            or self.board.current_player != self.ai_player_id
        ):
            return

        self.ai_thinking = True
        self.status.set("AI 正在思考…")
        game_id = self.game_id
        board = self.board
        ai_player = self.ai_player

        def calculate_move():
            """仅执行计算并写入队列，不在子线程中操作 Tkinter。"""
            try:
                move = ai_player.get_action(board)
                self.ai_results.put((game_id, board, move, None))
            except Exception as exc:
                self.ai_results.put((game_id, board, None, exc))

        threading.Thread(target=calculate_move, daemon=True).start()

    def process_ai_results(self):
        """由主线程轮询 AI 结果队列，并更新棋盘和界面状态。"""
        try:
            while True:
                game_id, board, move, error = self.ai_results.get_nowait()
                # 用户可能在 AI 思考期间重新开始；旧结果不能落到新棋盘上。
                if game_id != self.game_id or board is not self.board:
                    continue

                self.ai_thinking = False
                if error is not None:
                    self.game_over = True
                    self.status.set("AI 计算失败，可以点击“重新开始”")
                    messagebox.showerror("AI 计算失败", str(error))
                    continue

                if move not in self.board.availables:
                    self.game_over = True
                    self.status.set("AI 返回了无效落子，可以点击“重新开始”")
                    continue

                self.board.do_move(move)
                self.draw_board()
                if not self.finish_if_needed():
                    self.status.set("轮到你落子：点击棋盘交叉点")
        except queue.Empty:
            pass

        # 定时轮询不会长时间占用 Tkinter 主线程。
        if self.root.winfo_exists():
            self.root.after(50, self.process_ai_results)

    def finish_if_needed(self):
        """检查胜负或平局；对局结束返回 True，否则返回 False。"""
        end, winner = self.board.game_end()
        if not end:
            return False

        self.game_over = True
        self.ai_thinking = False
        if winner == self.human_player_id:
            result = "你赢了！"
        elif winner == self.ai_player_id:
            result = "AI 获胜"
        else:
            result = "平局"
        self.status.set(result + "；点击“重新开始”再来一局")
        messagebox.showinfo("对局结束", result)
        return True


def parse_args():
    """解析模型、棋盘、MCTS 和运行设备等启动参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="best_policy.model",
        help="train.py 生成的 PyTorch 模型（默认：best_policy.model）",
    )
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--n-in-row", type=int, default=4)
    parser.add_argument("--playouts", type=int, default=400)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--temp", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--human-first",
        action="store_true",
        help="启动时由人类先手（也可以在窗口中切换）",
    )
    return parser.parse_args()


def run():
    """校验参数、加载模型并启动 Tkinter 主事件循环。"""
    args = parse_args()
    if args.board_size < 2:
        raise ValueError("棋盘边长必须至少为 2")
    if args.n_in_row < 2 or args.n_in_row > args.board_size:
        raise ValueError("n-in-row 必须在 2 和棋盘边长之间")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError("找不到模型文件：{}".format(model_path))

    # 模型只在程序启动时加载一次，重新开始对局不会重复读取模型文件。
    policy = PolicyValueNet(
        args.board_size,
        model_file=str(model_path),
        device=args.device,
        channels=args.channels,
        num_blocks=args.num_blocks,
    )
    root = tk.Tk()
    GomokuGUI(root, args, policy)
    root.mainloop()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
