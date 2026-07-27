import numpy as np

class Board:
    """游戏棋盘"""

    def __init__(self, width=8, height=8, n_in_row=5):
        # 棋盘大小，默认8*8
        self.width = int(width)
        self.height = int(height)
        # 获胜条件：连子数，默认5
        self.n_in_row = int(n_in_row)
        self.players = [1, 2]  # 玩家1和2

    def init_board(self, start_player=0):
        """初始化棋盘"""
        # 棋盘一排必须能放下n_in_row个棋子
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        # 初始执子玩家
        self.current_player = self.players[start_player]  
        # 用列表保存可落子位置，编号0 ~ width*height-1
        self.availables = list(range(self.width * self.height))
        # 棋盘状态，记录落子
        # key：棋落子位置
        # value：对应的棋手编号
        self.states = {}
        # 最后一次落子位置索引
        self.last_move = -1

    def move_to_location(self, move):
        """
        移动索引 -> 坐标
        例如 3*3 棋盘:
        0 1 2
        3 4 5
        6 7 8
        索引 5 的坐标是 (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """坐标 -> 移动索引"""
        # 必须是(x,y)
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 必须在编号范围内
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        当前玩家视角下的棋盘状态
        输入给网络的特征
        状态形状为 4*width*height。
        """
        # 棋盘状态
        square_state = np.zeros((4, self.width, self.height))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # 当前玩家的落子的索引列表
            move_curr = moves[players == self.current_player]
            # 对手的落子的索引列表
            move_oppo = moves[players != self.current_player]
            # 第0层：当前玩家的落子情况
            # 有棋子为1
            square_state[0][move_curr // self.width,
                            move_curr % self.width] = 1.0
            # 第1层：对手玩家的落子情况
            square_state[1][move_oppo // self.width,
                            move_oppo % self.width] = 1.0
            # 第2层：最后一步落子位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.width] = 1.0
        
        # 第3层：标记当前该落哪一方的棋
        square_state[3][:,:] = ( 1 if self.current_player == self.players[0] else 0)

        return square_state

    def do_move(self, move):
        """落子"""
        self.states[move] = self.current_player
        self.availables.remove(move)
        # 切人
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move


    # def has_a_winner(self):
    #     """
    #     胜负判定
    #     结束：(True, 胜方)
    #     为结束：(False, -1)
    #     """

    #     width = self.width
    #     height = self.height
    #     states = self.states
    #     n = self.n_in_row

    #     moved = list(set(range(width * height)) - set(self.availables))
    #     # 总步数少于 2*n_in_row-1 不可能结束
    #     if len(moved) < self.n_in_row *2-1:
    #         return False, -1

    #     for m in moved:
    #         h = m // width
    #         w = m % width
    #         player = states[m]

    #         # 从m向右取n个位置，判断是否连子
    #         # n个位置上有棋为玩家id，无棋为-1
    #         # set()去重，如果只有一种元素（全1或全2）则结束
    #         # m一定有棋，所以不可能是全 -1
    #         if (w in range(width - n + 1) and
    #                 len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
    #             return True, player

    #         # 从m向下取n个位置
    #         if (h in range(height - n + 1) and
    #                 len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
    #             return True, player

    #         # 向右下取n个位置
    #         if (w in range(width - n + 1) and h in range(height - n + 1) and
    #                 len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
    #             return True, player

    #         # 向左下取n个位置
    #         if (w in range(n - 1, width) and h in range(height - n + 1) and
    #                 len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
    #             return True, player

    #     return False, -1

    def has_a_winner(self):
        """
        胜负判定
        结束：(True, 胜方)
        为结束：(False, -1)

        更高效的判断方式：
        - 如果结束了一定包含last_move，设last_move的玩家为i
        - 横向：从last_move向左遍历a个i，向右b个i，a+b+1 = n则i赢
        - 纵向、右上到左下，右下到左上同理
        - 其他情况为未结束

        """
        
        # 总步数少于 2*n_in_row-1 不可能结束
        if len(self.states) < self.n_in_row * 2 - 1:
            return False, -1
            
        m = self.last_move      
        states = self.states
        player = states[m] # 可能获胜的玩家
        width = self.width
        height = self.height
        n = self.n_in_row
        
        # 将一维的最后一步转换为二维坐标
        h = m // width
        w = m % width
        
        # 定义需要检查的 4 个轴向
        # 每个轴包含正反两个方向的偏移量 (delta_h, delta_w)
        # 分别为：横向(左右)、纵向(上下)、主对角线(左下右上)、副对角线(左上右下)
        axes = [
            [(0, -1), (0, 1)],
            [(-1, 0), (1, 0)],
            [(-1, -1), (1, 1)],
            [(-1, 1), (1, -1)]
        ]
        
        # 遍历 4 个轴向
        for axis in axes:
            count = 1  # 初始包含 last_move 本身（a+b+1 中的 1）
            
            # 向当前轴向的两个反方向分别延伸 (a 和 b)
            for dh, dw in axis:
                curr_h, curr_w = h + dh, w + dw
                
                # 在棋盘边界内循环延伸
                while 0 <= curr_h < height and 0 <= curr_w < width:
                    # 转换回一维位置
                    curr_m = curr_h * width + curr_w
                    
                    # 检查该位置是否是当前玩家的棋子
                    if states.get(curr_m, -1) == player:
                        count += 1
                        curr_h += dh
                        curr_w += dw
                    else:
                        break # 遇到对方棋子或空位，该方向延伸结束
                        
            # 如果正反两个方向连起来的同色棋子数量达到 n
            if count >= n:
                return True, player
                
        # 4 个轴向都检查完毕且没有触发获胜
        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1 # 平局
        return False, -1

    def get_current_player(self):
        """获取当前玩家"""
        return self.current_player


class Game:
    """游戏控制器"""

    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """绘制棋盘并显示游戏信息"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """开始一局两人对战"""

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        # 初始化棋盘、玩家序号
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        # is_shown是否可视化
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        
        while True:
            # 选择当前玩家
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            # 行动
            move = player_in_turn.get_action(self.board)
            # 落子
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            
            # 判断是否结束
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0):
        """
        使用 MCTS 玩家进行自我对弈
        复用搜索树并保存训练数据
        形式为 (state, mcts_probs, z)
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, return_prob=1)
            # 保存训练数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行一步落子
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, p1, p2)

            # 是否结束
            end, winner = self.board.game_end()
            if end:
                # 针对每个状态记录当前玩家视角下的胜负结果
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 胜方数据标签为1
                    winners_z[np.array(current_players) == winner] = 1.0
                    # 负方数据标签为-1
                    winners_z[np.array(current_players) != winner] = -1.0
                
                # 重置 MCTS 根节点
                player.reset_player()
                
                if is_shown:
                    if winner != -1:
                        print("对局结束，赢家是玩家：", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
