import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from random_agent import RandomAgent
from tqdm import tqdm
from simplerulebased_agent import SimpleRuleBasedAgent

def simulate_game(env, agent_x, agent_o, verbose=False):
    obs = env.reset()
    done = False

    while not done:
        current_player = env.current_player
        agent = agent_x if current_player == 1 else agent_o
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        if verbose:
            print(f"{'X' if current_player == 1 else 'O'} plays: {action}")
            env.render()

    if info.get("invalid", False):
        return -current_player
    elif reward == 1.0:
        return current_player 
    elif reward == 0.5:
        return 0 
    else:
        return -current_player

if __name__ == "__main__":
    num_games = 10
    board_size = 5

    env = VanishingTicTacToeEnv(board_size=board_size)

    # these agents can be changed to different agents in the future!
    agent_x = RandomAgent(env.action_space)
    agent_o = SimpleRuleBasedAgent(env.action_space)


    results = {1: 0, -1: 0}  # X win, O win

    for _ in tqdm(range(num_games), desc=f'Running games in board_size {board_size}'):
        outcome = simulate_game(env, agent_x, agent_o)
        if outcome == 0:
            raise ValueError("Draw! (impossible)")
        results[outcome] += 1

    print(f"\nResults over {num_games} games (Board size: {board_size}x{board_size}):")
    print(f"  X wins:  {results[1]}")
    print(f"  O wins:  {results[-1]}")