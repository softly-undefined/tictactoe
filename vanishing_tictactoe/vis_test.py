import numpy as np
import matplotlib.pyplot as plt
from vanishing_tictactoe import VanishingTicTacToeEnv
from random_agent import RandomAgent
from simplerulebased_agent import SimpleRuleBasedAgent
from moderaterulebased_agent import ModerateRuleBasedAgent
from complexrulebased_agent import ComplexRuleBasedAgent
from ddqn_agent import DDQNAgent
import torch

# === CONFIG ===
DELAY = 0.5
BOARD_SIZE = 3
VISUALIZE = True
MODEL_PATH = "models/wr_75.pth"

# importing the ddqn agent
# getting environment details
dummy_env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)
state_dim = dummy_env.observation_space["board"].shape[0] + \
            dummy_env.observation_space["history_x"].shape[0] + \
            dummy_env.observation_space["history_o"].shape[0] + 1
action_dim = dummy_env.action_space.n

ddqn_agent = DDQNAgent(state_dim, action_dim, device="cpu")  # or "cuda"
ddqn_agent.load(MODEL_PATH)


if VISUALIZE:
    fig, ax = plt.subplots()

def render_board(env, board, title="", agent_x_name="", agent_o_name=""):
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

            ax.add_patch(plt.Rectangle((j, BOARD_SIZE - 1 - i), 1, 1, facecolor=color, edgecolor='black'))
            label = "X" if val == 1 else ("O" if val == -1 else "")
            ax.text(j + 0.5, BOARD_SIZE - 1 - i + 0.5, label, ha='center', va='center', fontsize=20)

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(-1.2, BOARD_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle(title, fontsize=16)
    ax.text(BOARD_SIZE / 2, -0.5, f"X = {agent_x_name}", ha='center', va='center', fontsize=12)
    ax.text(BOARD_SIZE / 2, -0.9, f"O = {agent_o_name}", ha='center', va='center', fontsize=12)
    fig.canvas.draw()
    plt.pause(DELAY)

if __name__ == "__main__":
    env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)

    agent_x = ComplexRuleBasedAgent(env.action_space)  # player X (1)
    agent_o = ddqn_agent  # player O (-1)

    obs = env.reset()
    done = False
    board = obs["board"]

    if VISUALIZE:
        render_board(env, board, title="Game Start", agent_x_name=agent_x.__class__.__name__, agent_o_name=agent_o.__class__.__name__)
    else:
        env.render(delay=DELAY)

    while not done:
        current_player = int(obs["current_player"][0])
        agent = agent_x if current_player == 1 else agent_o
        action = agent.act(obs)
        print(f"{'X' if current_player == 1 else 'O'} plays: {action}")
        obs, reward, done, info = env.step(action)
        board = obs["board"]

        if VISUALIZE:
            render_board(env, board, title=f"{'X' if current_player == 1 else 'O'} played", agent_x_name=agent_x.__class__.__name__, agent_o_name=agent_o.__class__.__name__)
        else:
            env.render(delay=DELAY)

    if info.get("invalid", False):
        print(f"Player {'X' if current_player == 1 else 'O'} made an invalid move and lost.")
    elif reward == 1.0:
        print(f"Player X wins!")
    elif reward == -1.0:
        print("Player O wins!")
    else:
        print(f"Draw? This shouldn't ever happen")