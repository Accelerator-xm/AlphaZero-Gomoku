import numpy as np
import copy


def softmax(x):
    # 把任意数字转换成概率
    # 减max(x)用于稳定数值，防止计算溢出
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    MCTS 树的节点
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到 TreeNode 的映射
        self._n_visits = 0  # 被访问的次数
        self._Q = 0         # 动作价值（转移到这个节点需要动作）
        self._u = 0         # 先验分数
        self._P = prior_p   # 先验概率（策略网络输出）

    def expand(self, action_priors):
        """
        扩展，新增孩子

        action_priors：动作及其先验概率的元组列表，由策略函数给出
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        PUCT选择
        在子节点中选择价值最大的分支。
        返回：(action, next_node) 元组。
        """
        # 匿名函数 lambda act_node: act_node[1].get_value(c_puct)
        # 参数名act_node，这里输入(action, child_node)
        # 取act_node[1]计算价值
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新Q值
        leaf_value：从当前玩家视角得到的子树评估值。
        """
        # 累加访问次数。
        self._n_visits += 1
        # 更新 Q
        # 访问值的均值：Q = (V1 + V2 +...+ Vn)/n
        # 采用增量更新
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        回传更新祖先节点
        """
        # 如果不是根节点，需要先更新父节点。
        if self._parent:
            # 取负：双方棋手每层交替，价值相反
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算该节点的价值 Q+u
        
        u：用于鼓励探索
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查是否为叶子节点（即下面没有展开的子节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """蒙特卡洛树搜索实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn：
        - 输入棋盘状态，
        - 输出
          - 当前玩家可用动作的 (动作, 概率) 列表
          - [-1, 1] 区间内的分数，-1表示输、1表示赢、0表示平
        
        c_puct：控制搜索向高价值策略收敛速度的参数。值越大，越依赖先验概率。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        模拟推演一次下棋
        state：棋盘
        从根节点到叶子节点执行一次模拟，并将叶子值回传到祖先节点
        状态会原地修改，因此外部必须传入副本
        """
        
        # 根据价值走到叶子节点
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪心选择下一步。
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 评估叶子节点，扩展子节点
        # 网络侧已经过滤了可选动作了
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否结束。
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 若已经结束，直接返回真实终局值。
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 回传更新
        # 网络输出：当前状态，轮到的棋手的动作概率和价值，棋手a的分数
        # 树节点Q表示：上一轮棋手b执行动作到这一步的价值，想象成边的值
        # 更新的是边的值，a的分数取负才是b的分数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        计算动作和概率
  
        state：当前游戏状态。棋盘类对象
        temp：温度参数，范围 (0, 1]，控制探索程度。

        输出可选动作和概率（网络侧已经过滤了可选动作）
        """

        # 推演 _n_playout 次，即看 _n_playout 步的未来情况
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点的访问次数计算动作概率。
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """
        根据实际移动裁剪树，以上一步的节点为根节点
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            # 清空树，用于重置
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于 MCTS 的 AI 玩家。"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, temp=1e-3):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.temp = temp

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        """重置MCTS树"""
        self.mcts.update_with_move(-1)

    def get_action(self, board, return_prob=0):
        sensible_moves = board.availables
        # MCTS 返回的 pi 向量，和 AlphaGo Zero 论文中的定义一致。
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, self.temp)
            # 合法动作概率 -> 动作概率（包含完整棋盘位置的对于概率）
            move_probs[list(acts)] = probs 
            
            if self._is_selfplay:
                # 添加 Dirichlet 噪声用于探索
                # dirichlet随机生成一组概率，保证求和等于1
                # 保留75%的MCTS真实结果，和25%的随机噪声
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并复用搜索树。
                self.mcts.update_with_move(move)
            else:
                # 在默认 temp=1e-3 时，效果几乎等价于选择概率最高的动作。
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                # 实际对战阶段用的是不同的网络，自己推演的结果不是对方的实际落子
                # 需要每次重新推演
                self.reset_player()

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("警告：棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)
