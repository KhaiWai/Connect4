import random
import math


BOT_NAME = "INSERT NAME FOR YOUR BOT HERE"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def get_move(self, state, depth=None):
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state, depth=None):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state, depth)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state, depth):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board
            depth: for this agent, the depth argument should be ignored!

        Returns: the exact minimax utility value of the state
        """
        nextp = state.next_player()

        if (state.is_full()):
            value = state.score()
            return value

        if nextp == 1:
            maxeval = -math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth)
                maxeval = max(maxeval, eval)

            return maxeval

        else:
            mineval = math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth)
                mineval = min(mineval, eval)

            return mineval


class HeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def minimax(self, state, depth):
        return self.minimax_depth(state, depth)

    def minimax_depth(self, state, depth):
        """Determine the heuristically estimated minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board
            depth: the maximum depth of the game tree that minimax should traverse before
                estimating the utility using the evaluation() function.  If depth is 0, no
                traversal is performed, and minimax returns the results of a call to evaluation().
                If depth is None, the entire game tree is traversed.

        Returns: the minimax utility value of the state
        """
        nextp = state.next_player()
        if (state.is_full() or depth == 0):
            value = self.evaluation(state)
            return value

        if nextp == 1:
            maxeval = -math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth-1)
                maxeval = max(maxeval, eval)

            return maxeval

        else:
            mineval = math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth-1)
                mineval = min(mineval, eval)

            return mineval

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!"""

        total_value = 0
        col_list = list(map(list, state.get_all_cols()))
        for m in col_list:
            count = 0
            neg_count = 0
            for x in m:
                if x == 1:
                    count += 1
                    total_value -= 10 ** neg_count
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += 10 ** count
                    count = 0

                else:
                    total_value -= 10 ** neg_count
                    total_value += 10 ** count
                    count = 0
                    neg_count = 0

            total_value += 10 ** count
            total_value -= 10 ** neg_count

        for n in state.get_all_rows():
            count = 0
            neg_count = 0
            # print(n)
            for x in n:
                if x == 1:
                    count += 1
                    total_value -= 10 ** neg_count
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += 10 ** count
                    count = 0

                else:
                    total_value -= 10 ** neg_count
                    total_value += 10 ** count
                    count = 0
                    neg_count = 0

            total_value += 10 ** count
            total_value -= 10 ** neg_count

        for n in state.get_all_diags():
            count = 0
            neg_count = 0
            for x in n:
                if x == 1:
                    count += 1
                    total_value -= 10 ** neg_count
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += 10 ** count
                    count = 0

                else:
                    total_value -= 10 ** neg_count
                    total_value += 10 ** count
                    count = 0
                    neg_count = 0

            total_value += 10 ** count
            total_value -= 10 ** neg_count

        return total_value


class PruneAgent(HeuristicAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def minimax(self, state, depth):
        return self.minimax_prune(state, depth, -math.inf, math.inf)

    def minimax_prune(self, state, depth, alpha, beta):

        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by ComputerAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the class variables
        GameState.p1_state_count and GameState.p2_state_count, which keep track of how many
        GameState objects were created over time.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: see ComputerDepthLimitAgent.minimax() above

        Returns: the minimax utility value of the state
        """
        nextp = state.next_player()
        if depth == 0 or state.is_full():
            value = self.evaluation(state)
            return value

        if nextp == 1:
            value = -math.inf
            for move, state in state.successors():
                value = max(value, self.minimax_prune(state, depth-1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return value

        else:
            value = math.inf
            for move, state in state.successors():
                value = min(value, self.minimax_prune(state, depth-1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value


class HeuristicAgent2(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def minimax(self, state, depth):
        return self.minimax_depth(state, depth)

    def minimax_depth(self, state, depth):
        """Determine the heuristically estimated minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board
            depth: the maximum depth of the game tree that minimax should traverse before
                estimating the utility using the evaluation() function.  If depth is 0, no
                traversal is performed, and minimax returns the results of a call to evaluation().
                If depth is None, the entire game tree is traversed.

        Returns: the minimax utility value of the state
        """
        nextp = state.next_player()
        if (state.is_full() or depth == 0):
            value = self.evaluation2(state)
            return value

        if nextp == 1:
            maxeval = -math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth - 1)
                maxeval = max(maxeval, eval)

            return maxeval

        else:
            mineval = math.inf
            for move, state in state.successors():
                eval = self.minimax(state, depth - 1)
                mineval = min(mineval, eval)

            return mineval

        # self.evaluation(state)
        # return 9  # Change this line!

    def evaluation2(self, state):
        col_list = list(map(list, state.get_all_cols()))
        total_value = 0

        for m in col_list:
            count = 0
            neg_count = 0
            for x in m:
                if x == 1:
                    count += 1
                    total_value -= neg_count ** 2
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += count ** 2
                    count = 0

                else:
                    total_value -= neg_count ** 2
                    total_value += count ** 2
                    count = 0
                    neg_count = 0

            total_value += count ** 2
            total_value -= neg_count ** 2

        for n in state.get_all_rows():
            count = 0
            neg_count = 0

            for x in n:
                if x == 1:
                    count += 1
                    total_value -= neg_count ** 2
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += count ** 2
                    count = 0

                else:
                    total_value -= neg_count ** 2
                    total_value += count ** 2
                    count = 0
                    neg_count = 0

            total_value += count ** 2
            total_value -= neg_count ** 2

        for d in state.get_all_diags():
            count = 0
            neg_count = 0

            for x in n:
                if x == 1:
                    count += 1
                    total_value -= neg_count ** 2
                    neg_count = 0

                elif x == -1:
                    neg_count += 1
                    total_value += count ** 2
                    count = 0

                else:
                    total_value -= neg_count ** 2
                    total_value += count ** 2
                    count = 0
                    neg_count = 0

            total_value += count ** 2
            total_value -= neg_count ** 2

        return total_value
