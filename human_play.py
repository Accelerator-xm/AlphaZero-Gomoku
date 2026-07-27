import argparse
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


class Human(object):
    """
    人类玩家
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            # 解析输入：row, col
            location = input("Your move: ")
            if isinstance(location, str):  # Python 3 下输入的是字符串
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("无效落子")
            # 无效重试
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="PyTorch model created by train.py")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--n-in-row", type=int, default=4)
    parser.add_argument("--playouts", type=int, default=400)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--temp", type=float, default=1e-3)
    parser.add_argument("--device", default="cup")
    # 人类先手
    parser.add_argument(
        "--human-first",
        action="store_true",
        help="Let the human make the first move",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    board = Board(width=args.width, height=args.height, n_in_row=args.n_in_row)
    game = Game(board)
    policy = PolicyValueNet(
        args.width,
        args.height,
        model_file=args.model,
        device=args.device,
        channels=args.channels,
        num_blocks=args.num_blocks,
    )
    ai_player = MCTSPlayer(
        policy.policy_value_fn,
        c_puct=5,
        n_playout=args.playouts,
        temp=args.temp,
    )
    human = Human()
    game.start_play(
        human,
        ai_player,
        start_player=0 if args.human_first else 1,
        is_shown=1,
    )

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nQuit")
