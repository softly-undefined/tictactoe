import random
import math
from functools import lru_cache
import matplotlib.pyplot as plt
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = (0,) * 9
        self.current_player = 1  # X starts
        return self.board, self.current_player

    def available_moves(self, board=None):
        b = board if board is not None else self.board
        return [i for i, v in enumerate(b) if v == 0]

    def step(self, action):
        b = list(self.board)
        if b[action] != 0:
            raise ValueError("Invalid move")
        b[action] = self.current_player
        self.board = tuple(b)

        winner = self.check_winner(self.board)
        if winner is not None:
            reward = winner * self.current_player
            done = True
        elif 0 not in self.board:
            reward = 0
            done = True
        else:
            reward = 0
            done = False

        self.current_player *= -1
        return (self.board, self.current_player), reward, done

    @staticmethod
    def check_winner(board):
        lines = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # cols
            (0,4,8), (2,4,6)            # diags
        ]
        for i, j, k in lines:
            s = board[i] + board[j] + board[k]
            if s == 3: return 1
            if s == -3: return -1
        return None

class MinimaxAgent:
    def __init__(self):
        self.policy = {}

    def train(self):
        self._minimax((0,)*9, 1)

    @lru_cache(maxsize=None)
    def _minimax(self, board, player):
        winner = TicTacToe.check_winner(board)
        if winner is not None:
            return winner * player
        if 0 not in board:
            return 0

        best_val = -math.inf
        best_move = None
        for move in [i for i, v in enumerate(board) if v == 0]:
            nb = list(board)
            nb[move] = player
            nb = tuple(nb)
            val = -self._minimax(nb, -player)
            if val > best_val:
                best_val, best_move = val, move
                if best_val == 1:
                    break
        self.policy[(board, player)] = best_move
        return best_val

    def get_action(self, board, player):
        return self.policy[(board, player)]

def simulate_games(num_games=1000):
    env     = TicTacToe()
    agent   = MinimaxAgent()
    agent.train()


    results = {'win': 0, 'draw': 0, 'loss': 0}
    for _ in tqdm(range(num_games)):
        board, player = env.reset()
        # pick whether Minimax is X (1) or O (-1) this game
        ai_player = random.choice([1, -1])

        done = False
        while not done:
            if player == ai_player:
                move = agent.get_action(board, player)
            else:
                move = random.choice(env.available_moves(board))

            (board, player), _, done = env.step(move)

        winner = TicTacToe.check_winner(board)
        # from Minimax’s perspective:
        if winner is None:
            results['draw'] += 1
        elif winner == ai_player:
            results['win']  += 1
        else:
            results['loss'] += 1

    return results

def simulate_games_2(num_games=1000):
    env     = TicTacToe()
    agent   = MinimaxAgent()
    agent.train()

    agent2 = MinimaxAgent()
    agent2.train()


    results = {'win': 0, 'draw': 0, 'loss': 0}
    for _ in tqdm(range(num_games)):
        board, player = env.reset()
        # pick whether Minimax is X (1) or O (-1) this game
        ai_player = random.choice([1, -1])

        done = False
        while not done:
            if player == ai_player:
                move = agent.get_action(board, player)
            else:
                move = agent2.get_action(board, player)

            (board, player), _, done = env.step(move)

        winner = TicTacToe.check_winner(board)
        # from Minimax’s perspective:
        if winner is None:
            results['draw'] += 1
        elif winner == ai_player:
            results['win']  += 1
        else:
            results['loss'] += 1

    return results

# Run simulation
num_games = 1_000_000
res = simulate_games(num_games)

# Plot results
categories = ['AI Wins', 'Draws', 'AI Losses']
counts = [res['win'], res['draw'], res['loss']]

plt.figure()
plt.bar(categories, counts)
plt.xlabel('Outcome')
plt.ylabel('Number of Games')
plt.title(f'AI vs Random Bot ({num_games} games)')
plt.show()
