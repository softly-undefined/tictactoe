import math
from functools import lru_cache

class TicTacToe:
    def __init__(self):
        # board is a tuple of 9 elements:  1 for X, -1 for O, 0 for empty
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
            reward = winner * self.current_player  # +1 if current_player wins, -1 if loses
            done = True
        elif 0 not in self.board:
            reward = 0  # draw
            done = True
        else:
            reward = 0
            done = False

        self.current_player *= -1
        return (self.board, self.current_player), reward, done

    @staticmethod
    def check_winner(board):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),  # rows
            (0,3,6),(1,4,7),(2,5,8),  # cols
            (0,4,8),(2,4,6)           # diags
        ]
        for i,j,k in lines:
            s = board[i] + board[j] + board[k]
            if s == 3: return 1   # X wins
            if s == -3: return -1 # O wins
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

        best_val = -math.inf
        best_move = None
        for move in [i for i,v in enumerate(board) if v==0]:
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

def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    rows = [
        board[0:3],
        board[3:6],
        board[6:9],
    ]
    print("\n".join("|".join(symbols[cell] for cell in row) for row in rows))
    print()

if __name__ == "__main__":
    env = TicTacToe()
    agent = MinimaxAgent()
    print("Solving Tic-Tac-Toe...")
    agent.train()
    print("Done. Let's play!")

    # choose sides
    human_char = ""
    while human_char not in ("X","O"):
        human_char = input("Choose your side (X goes first) [X/O]: ").upper()
    human = 1 if human_char=="X" else -1
    ai = -human

    state, player = env.reset()
    print_board(state)

    # game loop
    pos_map = {str(i+1): i for i in range(9)}
    while True:
        if player == human:
            move = None
            while True:
                choice = input(f"Your move ({human_char}), choose 1–9: ")
                if choice in pos_map and state[pos_map[choice]]==0:
                    move = pos_map[choice]
                    break
                print(" Invalid. Pick an empty cell 1–9.")
        else:
            move = agent.get_action(state, player)
            print(f"Bot ({'X' if ai==1 else 'O'}) plays: {move+1}")

        (state, player), reward, done = env.step(move)
        print_board(state)

        if done:
            if reward == 1:
                winner = "X"
            elif reward == -1:
                winner = "O"
            else:
                winner = None

            if winner is None:
                print("It's a draw!")
            elif (winner == "X" and human==1) or (winner=="O" and human==-1):
                print("You win! 🎉")
            else:
                print("Bot wins. 😢")
            break
