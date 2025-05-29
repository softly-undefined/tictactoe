import matplotlib.pyplot as plt
import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from random_agent import RandomAgent
from complexrulebased_agent import ComplexRuleBasedAgent
from moderaterulebased_agent import ModerateRuleBasedAgent
from ddqn_agent import DDQNAgent
from simplerulebased_agent import SimpleRuleBasedAgent
import torch

# === CONFIG ===
BOARD_SIZE = 3
DELAY = 0.02
NUM_GAMES = 100

DELAY_BETWEEN_GAMES = 0.05

MODEL_PATH = "models/wr_75.pth"


# Agent stuff
# Get dimensions from a dummy env
dummy_env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)
state_dim = dummy_env.observation_space["board"].shape[0] + \
            dummy_env.observation_space["history_x"].shape[0] + \
            dummy_env.observation_space["history_o"].shape[0] + 1  # + current_player
action_dim = dummy_env.action_space.n

# Create and load agent
ddqn_agent = DDQNAgent(state_dim, action_dim, device="cpu")  # or "cuda"
ddqn_agent.load(MODEL_PATH)

# leaderboard
x_wins = 0
o_wins = 0

fig, ax = plt.subplots()

def render_board(env, board, game_title="", agent_x_name="", agent_o_name=""):
    ax.clear()
    history_x = list(env.move_history_x)
    history_o = list(env.move_history_o)

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            idx = i * BOARD_SIZE + j
            val = board[idx]
            color = 'white'

            if val == 1:
                if idx in history_x:
                    pos = history_x.index(idx)
                    color = ['#1c5ca8', '#4e8ed1', 'lightblue'][min(pos, 2)]
                else:
                    color = 'lightblue'
            elif val == -1:
                if idx in history_o:
                    pos = history_o.index(idx)
                    color = ['#a8281c', '#d1634e', 'salmon'][min(pos, 2)]
                else:
                    color = 'salmon'

            ax.add_patch(plt.Rectangle((j, BOARD_SIZE - 1 - i), 1, 1,
                                       facecolor=color, edgecolor='black'))
            label = "X" if val == 1 else ("O" if val == -1 else "")
            ax.text(j + 0.5, BOARD_SIZE - 1 - i + 0.5, label,
                    ha='center', va='center', fontsize=20)

    # Grid setup
    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(-1.2, BOARD_SIZE)  # More space below the board
    ax.set_aspect('equal')
    ax.axis('off')

    # Supertitle
    fig.suptitle(game_title, fontsize=16)

    # Agent labels BELOW the board on separate lines
    ax.text(BOARD_SIZE / 2, -0.5, f"X = {agent_x_name}", ha='center', va='center', fontsize=12)
    ax.text(BOARD_SIZE / 2, -0.9, f"O = {agent_o_name}", ha='center', va='center', fontsize=12)

    fig.canvas.draw()
    plt.pause(DELAY)

# === Main Loop for Multiple Games ===
for game_num in range(1, NUM_GAMES + 1):
    env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)
    agent_x = RandomAgent(env.action_space)
    agent_o = SimpleRuleBasedAgent(env.action_space)
    obs = env.reset()
    board = obs["board"]
    done = False

    while not done:
        current = int(obs["current_player"][0])
        agent = agent_x if current == 1 else agent_o
        move = agent.act(obs)

        print(f"Game {game_num} | {'X' if current == 1 else 'O'} plays: {move}")
        render_board(
            env, board,
            game_title=f"Game {game_num} | X Wins: {x_wins} | O Wins: {o_wins}",
            agent_x_name=agent_x.__class__.__name__,
            agent_o_name=agent_o.__class__.__name__
        )


        obs, reward, done, info = env.step(move)
        board = obs["board"]

    render_board(
        env, board,
        game_title=f"Game {game_num} | X Wins: {x_wins} | O Wins: {o_wins}",
        agent_x_name=agent_x.__class__.__name__,
        agent_o_name=agent_o.__class__.__name__
    )


    plt.pause(DELAY_BETWEEN_GAMES)

    if reward == 1.0:
        x_wins += 1
        print("Player X wins!")
    elif reward == -1.0:
        o_wins += 1
        print("Player O wins!")
    else:
        print("Draw? This shouldn't happen")


# === Final Summary ===
print("\n=== Final Leaderboard ===")
print(f"Agent X ({agent_x.__class__.__name__}): {x_wins} wins")
print(f"Agent O ({agent_o.__class__.__name__}): {o_wins} wins")
