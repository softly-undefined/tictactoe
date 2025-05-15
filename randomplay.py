import os
import time
import random
import math
from functools import lru_cache

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = (0,) * 9
        self.current_player = 1  # X starts
        return self.board, self.current_player

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

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
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
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
            return 0  # draw

        best_val, best_move = -math.inf, None
        for m in [i for i,v in enumerate(board) if v==0]:
            nb = list(board); nb[m] = player; nb = tuple(nb)
            val = -self._minimax(nb, -player)
            if val > best_val:
                best_val, best_move = val, m
                if best_val == 1:
                    break
        self.policy[(board, player)] = best_move
        return best_val

    def get_action(self, board, player):
        return self.policy[(board, player)]

def print_board(board):
    sym = {1:"X", -1:"O", 0:" "}
    for r in range(3):
        print(" " + " | ".join(sym[board[3*r + c]] for c in range(3)))
        if r < 2: print("---+---+---")
    print()

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == "__main__":
    NUM_GAMES = 1000      # how many games to show
    DELAY = 0.2        # seconds between moves

    env = TicTacToe()
    agent = MinimaxAgent()
    agent.train()

    agent2 = MinimaxAgent()
    agent2.train()

    for game in range(1, NUM_GAMES+1):
        board, player = env.reset()
        clear_screen()
        print(f"Game {game}/{NUM_GAMES}")
        print_board(board)
        time.sleep(DELAY)

        while True:
            if player == 1:
                move = agent.get_action(board, player)
            else:
                move = agent.get_action(board, player)

            (board, player), _, done = env.step(move)
            clear_screen()
            print(f"Game {game}/{NUM_GAMES}")
            print_board(board)
            time.sleep(DELAY)

            if done:
                w = TicTacToe.check_winner(board)
                if w == 1:
                    print("X wins!")
                elif w == -1:
                    print("O wins!")
                else:
                    print("Draw!")
                time.sleep(0.5)
                break
