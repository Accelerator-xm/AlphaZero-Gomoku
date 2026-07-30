import numpy as np
import copy

def softmax(x):
    """数值稳定的 softmax。"""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class ParallelTreeNode:
    """
    MCTS 的树节点

    批量处理叶节点，用到ALphaGo里的 Virtual Loss

    Virtual Loss 使用独立的临时统计量，不会污染真实访问次数和 Q 值。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = float(prior_p)

        # Virtual Loss
        self._virtual_visits = 0
        self._virtual_value_sum = 0.0

    def expand(self, action_priors):
        """扩展"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = ParallelTreeNode(self, prob)

    def select(self, c_puct):
        """选择 PUCT 分数最高的分支。"""
        return max(
            self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct),
        )

    def add_virtual_loss(self, virtual_loss):
        """添加虚拟损失"""
        # 增加虚拟访问次数，减少虚拟价值，降低访问概率
        self._virtual_visits += 1
        self._virtual_value_sum -= float(virtual_loss)

    def revert_virtual_loss(self, virtual_loss, count=1):
        """撤销虚拟损失"""
        count = int(count)
        
        if self._virtual_visits < count:
            raise RuntimeError("Virtual Loss 计数不匹配")
        self._virtual_visits -= count
        self._virtual_value_sum += float(virtual_loss) * count

    def update(self, leaf_value, visit_count=1):
        """按 visit_count 次相同的评估结果更新真实访问统计。"""
        visit_count = int(visit_count)
        
        total_visits = self._n_visits + visit_count
        self._Q = (
            self._Q * self._n_visits + float(leaf_value) * visit_count
        ) / total_visits
        self._n_visits = total_visits

    def update_recursive(self, leaf_value, visit_count=1):
        """按 visit_count 次选择递归回传。"""
        if self._parent:
            self._parent.update_recursive(-leaf_value, visit_count)
        self.update(leaf_value, visit_count)

    def get_value(self, c_puct):
        """计算该节点的价值 Q+u"""
        # 带虚拟损失的访问次数
        effective_visits = self._n_visits + self._virtual_visits
        if effective_visits:
            # 带虚拟损失的q值
            effective_q = (self._Q * self._n_visits + self._virtual_value_sum) / effective_visits
        else:
            effective_q = 0.0

        # 带虚拟损失的父节点访问次数
        parent_visits = self._parent._n_visits + self._parent._virtual_visits
        # 探索分数
        self._u = c_puct * self._P * np.sqrt(max(0, parent_visits)) / (1 + effective_visits)
        
        return effective_q + self._u

    def is_leaf(self):
        return not self._children


class _PendingPlayout:
    """一个已选择的唯一叶节点及其在当前批次内的选择次数。"""

    def __init__(self, node, board, path, terminal, winner):
        self.node = node
        self.board = board
        self.path = path
        self.terminal = terminal
        self.winner = winner
        self.selection_count = 1


class BatchedMCTS:
    """在同一棵树上批量选择叶节点的 MCTS。"""

    def __init__(
        self,
        policy_value_batch_fn,
        c_puct=5,
        n_playout=500,
        leaf_batch_size=16,
        virtual_loss=1.0,
    ):

        self._root = ParallelTreeNode(None, 1.0)
        self._policy_batch = policy_value_batch_fn
        self._c_puct = float(c_puct)
        self._n_playout = int(n_playout)
        self._leaf_batch_size = int(leaf_batch_size)
        self._virtual_loss = float(virtual_loss)

    def _select_leaf(self, state):
        """选择一个叶节点，并立即在整条路径上放置 Virtual Loss。"""
        board = copy.deepcopy(state)
        node = self._root
        path = [node]

        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            board.do_move(action)
            path.append(node)

        # 添加虚拟损失
        for path_node in path:
            path_node.add_virtual_loss(self._virtual_loss)

        terminal, winner = board.game_end()
        return _PendingPlayout(node, board, path, terminal, winner)

    def _release_pending(self, pending):
        """撤销该叶节点全部选择所产生的 Virtual Loss。"""
        for path_node in pending.path:
            path_node.revert_virtual_loss(
                self._virtual_loss,
                pending.selection_count,
            )

    def _finish_playout(self, pending, action_priors, leaf_value):
        """完成推演 扩展+回传"""
        self._release_pending(pending)
        if not pending.terminal:
            pending.node.expand(action_priors)
        pending.node.update_recursive(
            -float(leaf_value),
            pending.selection_count,
        )

    @staticmethod
    def _terminal_value(pending):
        """终局结果"""
        if pending.winner == -1:
            return 0.0
        return (
            1.0
            if pending.winner == pending.board.get_current_player()
            else -1.0
        )

    def _playout_batch(self, state, max_batch_size):
        """选择并完成一轮模拟，返回本轮实际完成的模拟数。"""
        # 模拟max_batch_size次节点
        pending_by_node = {}    # 记录以访问过的节点
        for _ in range(max_batch_size):
            pending = self._select_leaf(state)
            existing = pending_by_node.get(pending.node)
            if existing is None:
                pending_by_node[pending.node] = pending
            else:
                existing.selection_count += 1

        # 同一叶节点只保留一份局面用于网络评估和扩展。
        pending_playouts = list(pending_by_node.values())

        # 没有结束的节点
        nonterminal = [pending for pending in pending_playouts if not pending.terminal]

        # 网络预测
        evaluations = (
            list(self._policy_batch([pending.board for pending in nonterminal]))
            if nonterminal
            else []
        )

        # 扩展+回传
        evaluation_iter = iter(evaluations) # 迭代器
        for pending in pending_playouts:
            if pending.terminal:
                action_priors = ()
                leaf_value = self._terminal_value(pending)
            else:
                action_priors, leaf_value = next(evaluation_iter)
            self._finish_playout(pending, action_priors, leaf_value)

        return sum(
            pending.selection_count
            for pending in pending_playouts
        )

    def get_move_probs(self, state, temp=1e-3):
        """运行指定次数模拟，并按根节点访问次数返回动作概率。"""
        # 推演 _n_playout 次
        # 批量进行模拟
        completed = 0
        while completed < self._n_playout:
            batch_size = min(
                self._leaf_batch_size,
                self._n_playout - completed,
            )
            completed += self._playout_batch(state, batch_size)

        # 根据根节点的访问次数计算动作概率。
        # 获取动作与访问次数
        act_visits = [
            (act, node._n_visits)
            for act, node in self._root._children.items()
        ]
        # 计算动作概率
        acts, visits = zip(*act_visits)
        act_probs = softmax(
            1.0 / temp * np.log(np.asarray(visits) + 1e-10)
        )
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        根据实际移动裁剪树，以上一步的节点为根节点
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = ParallelTreeNode(None, 1.0)


class BatchedMCTSPlayer:
    """供并行 Self-play Actor 使用的批量 MCTS 玩家。"""

    def __init__(
        self,
        policy_value_batch_fn,
        c_puct=5,
        n_playout=500,
        is_selfplay=0,
        temp=1e-3,
        leaf_batch_size=16,
        virtual_loss=1.0,
    ):
        self.mcts = BatchedMCTS(
            policy_value_batch_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            leaf_batch_size=leaf_batch_size,
            virtual_loss=virtual_loss,
        )
        self._is_selfplay = is_selfplay
        self.temp = temp

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, return_prob=0):
        sensible_moves = board.availables
        # 完整动作概率
        move_probs = np.zeros(board.size ** 2)
        if not sensible_moves:
            print("警告：棋盘已满")
            return None

        acts, probs = self.mcts.get_move_probs(board, self.temp)
        move_probs[list(acts)] = probs

        if self._is_selfplay:
            # 自博弈：添加 Dirichlet 噪声用于探索
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            # 实际对战选取概率最大的
            move = np.random.choice(acts, p=probs)
            self.reset_player()

        if return_prob:
            return move, move_probs
        return move

    def __str__(self):
        return "Batched MCTS {}".format(self.player)
