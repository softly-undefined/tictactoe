import os
import time
import random
import math
from functools import lru_cache

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1  # X starts
        self.move_history = {1: [], -1: []}
        return tuple(self.board), self.current_player

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        hist = self.move_history[self.current_player]
        if len(hist) >= 3:
            old = hist.pop(0)
            self.board[old] = 0

        if self.board[action] != 0:
            raise ValueError("Invalid move")
        self.board[action] = self.current_player
        hist.append(action)

        winner = self.check_winner(self.board)
        if winner is not None:
            reward, done = winner * self.current_player, True
        else:
            reward, done = 0, False

        self.current_player *= -1
        return (tuple(self.board), self.current_player), reward, done

    @staticmethod
    def check_winner(b):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for i,j,k in lines:
            s = b[i] + b[j] + b[k]
            if s == 3:   return 1
            if s == -3:  return -1
        return None



class MinimaxAgent:
    def __init__(self, max_depth=60):
        # how many plies (half-moves) to search before declaring a draw
        self.max_depth = max_depth
        self.policy = {}  # maps (board, hist_x, hist_o, player) -> best_move

    def train(self):
        empty_board = (0,) * 9
        empty_hist  = ()
        # kick off minimax with full depth
        self._minimax(empty_board, empty_hist, empty_hist, 1, self.max_depth)

    @lru_cache(maxsize=None)
    def _minimax(self, board, hist_x, hist_o, player, depth):
        # 1) depth cutoff
        if depth == 0:
            return 0

        # 2) terminal win check
        winner = TicTacToe.check_winner(board)
        if winner is not None:
            return winner * player

        best_val, best_move = -math.inf, None

        for idx, v in enumerate(board):
            if v != 0:
                continue

            # simulate the 3-marker removal + placement
            b_list = list(board)
            hx, ho = list(hist_x), list(hist_o)

            if player == 1:
                if len(hx) >= 3:
                    old = hx.pop(0)
                    b_list[old] = 0
            else:
                if len(ho) >= 3:
                    old = ho.pop(0)
                    b_list[old] = 0

            b_list[idx] = player
            if player == 1:
                hx.append(idx)
            else:
                ho.append(idx)

            new_board  = tuple(b_list)
            new_hist_x = tuple(hx)
            new_hist_o = tuple(ho)

            val = -self._minimax(
                new_board, new_hist_x, new_hist_o,
                -player, depth - 1
            )

            if val > best_val:
                best_val, best_move = val, idx
                if best_val == 1:
                    break

        # store the best move we found at *this* depth
        self.policy[(board, hist_x, hist_o, player)] = best_move
        return best_val

    def get_action(self, board, hist_x, hist_o, player):
        key = (board, hist_x, hist_o, player)
        if key in self.policy:
            return self.policy[key]
        # fallback to random if somehow unseen
        empties = [i for i, v in enumerate(board) if v == 0]
        return random.choice(empties)



def print_board(board):
    sym = {1:"X", -1:"O", 0:" "}
    for r in range(3):
        print(" " + " | ".join(sym[board[3*r + c]] for c in range(3)))
        if r < 2: print("---+---+---")
    print()

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')


if __name__ == "__main__":
    NUM_GAMES = 10000
    DELAY     = 0.2
    PRINT     = False

    env   = TicTacToe()
    agent = MinimaxAgent()
    agent.train()

    x_wins = o_wins = draws = 0

    for game in range(1, NUM_GAMES + 1):
        board, player = env.reset()

        if PRINT:
            clear_screen()
            print(f"Game {game}/{NUM_GAMES}")
            print_board(board)
            time.sleep(DELAY)

        while True:
            if player == 1:
                move = random.choice(env.available_moves())
            else:
                hist_x = tuple(env.move_history[1])
                hist_o = tuple(env.move_history[-1])
                move   = agent.get_action(board, hist_x, hist_o, player)

            (board, player), _, done = env.step(move)

            if PRINT:
                clear_screen()
                print(f"Game {game}/{NUM_GAMES}")
                print_board(board)
                time.sleep(DELAY)

            if done:
                w = TicTacToe.check_winner(board)
                if w == 1:
                    x_wins += 1
                    if PRINT: print("X wins!")
                elif w == -1:
                    o_wins += 1
                    if PRINT: print("O wins!")
                else:
                    draws += 1
                    if PRINT: print("Draw!")
                if PRINT:
                    time.sleep(0.5)
                break

    # final summary (always printed)
    if PRINT:
        clear_screen()
    print("Simulation complete!")
    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"Draws:  {draws}")
