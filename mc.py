import os
import time
import random
import copy
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1   # X starts
        self.move_history = {1: [], -1: []}
        return tuple(self.board), self.current_player

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        hist = self.move_history[self.current_player]
        # remove oldest if ≥3
        if len(hist) >= 3:
            old = hist.pop(0)
            self.board[old] = 0
        # place
        if self.board[action] != 0:
            raise ValueError("Invalid move")
        self.board[action] = self.current_player
        hist.append(action)
        # check win
        winner = self.check_winner(self.board)
        done = (winner is not None)
        reward = winner * self.current_player if done else 0
        # flip
        self.current_player *= -1
        return (tuple(self.board), self.current_player), reward, done

    @staticmethod
    def check_winner(b):
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        for i,j,k in lines:
            s = b[i] + b[j] + b[k]
            if s == 3:   return 1
            if s == -3:  return -1
        return None


class MonteCarloAgent:
    def __init__(self, sims_per_move=50, max_playout_length=50):
        self.sims = sims_per_move
        self.max_pl = max_playout_length

    def get_action(self, env: TicTacToe):
        player = env.current_player
        opp    = -player

        # 1) Immediate win
        for move in env.available_moves():
            sim = copy.deepcopy(env)
            (_, _), _, done = sim.step(move)
            if done and TicTacToe.check_winner(sim.board) == player:
                return move

        # 2) Avoid immediate loss
        safe1 = []
        for move in env.available_moves():
            sim = copy.deepcopy(env)
            (_, _), _, _ = sim.step(move)
            # opponent next move
            opp_wins = False
            for om in sim.available_moves():
                sim2 = copy.deepcopy(sim)
                (_, _), _, done2 = sim2.step(om)
                if done2 and TicTacToe.check_winner(sim2.board) == opp:
                    opp_wins = True
                    break
            if not opp_wins:
                safe1.append(move)

        # 3) Avoid giving opponent a fork (2-ply double threat)
        def creates_fork(sim_env):
            # count opponent winning replies
            win_replies = 0
            for om in sim_env.available_moves():
                sim2 = copy.deepcopy(sim_env)
                (_, _), _, done2 = sim2.step(om)
                if done2 and TicTacToe.check_winner(sim2.board) == opp:
                    win_replies += 1
                if win_replies >= 2:
                    return True
            return False

        safe2 = []
        for move in (safe1 or env.available_moves()):
            sim = copy.deepcopy(env)
            (_, _), _, _ = sim.step(move)
            if not creates_fork(sim):
                safe2.append(move)

        candidates = safe2 or safe1 or env.available_moves()

        # 4) Monte Carlo rollouts over candidates
        best_move, best_rate = None, -1.0
        for move in candidates:
            wins = 0
            for _ in range(self.sims):
                sim = copy.deepcopy(env)
                (_, _), _, done = sim.step(move)
                steps = 0
                while not done and steps < self.max_pl:
                    mv2 = random.choice(sim.available_moves())
                    (_, _), _, done = sim.step(mv2)
                    steps += 1
                if TicTacToe.check_winner(sim.board) == player:
                    wins += 1
            rate = wins / self.sims
            if rate > best_rate:
                best_rate, best_move = rate, move

        return best_move


def print_board(board):
    sym = {1:"X", -1:"O", 0:" "}
    for r in range(3):
        print(" " + " | ".join(sym[board[3*r+c]] for c in range(3)))
        if r < 2:
            print("---+---+---")
    print()

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == "__main__":
    NUM_GAMES           = 200
    SIMS_PER_MOVE       = 5000
    MAX_PLAYOUT_LENGTH  = 200
    DELAY               = 0.0
    PRINT               = False

    env   = TicTacToe()
    agent = MonteCarloAgent(sims_per_move=SIMS_PER_MOVE,
                            max_playout_length=MAX_PLAYOUT_LENGTH)
    agent2 = MonteCarloAgent(sims_per_move=SIMS_PER_MOVE,
                             max_playout_length=MAX_PLAYOUT_LENGTH)

    x_wins = o_wins = draws = 0

    games = (
        tqdm(range(1, NUM_GAMES+1), desc="Simulating")
        if not PRINT
        else range(1, NUM_GAMES+1)
    )

    for game in games:
        board, player = env.reset()

        if PRINT:
            clear_screen()
            print(f"Game {game}/{NUM_GAMES}")
            print_board(board)
            time.sleep(DELAY)

        move_count = 0
        while True:
            if env.current_player == 1:
                move = agent2.get_action(env)   # X as MC2
            else:
                move = agent.get_action(env)    # O as MC1

            (board, player), _, done = env.step(move)
            move_count += 1

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

            # hard draw cutoff
            if move_count >= 50:
                draws += 1
                break

    if PRINT:
        clear_screen()
    print("Simulation complete!")
    print(f"Out of {NUM_GAMES} games:")
    print(f"  X (MC2) wins:  {x_wins}")
    print(f"  O (MC1) wins:  {o_wins}")
    print(f"  Draws:         {draws}")
