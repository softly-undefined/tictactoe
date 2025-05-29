import matplotlib.pyplot as plt
import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from complexrulebased_agent import ComplexRuleBasedAgent
from ddqn_agent import DDQNAgent
import torch

BOARD_SIZE = 3
DELAY = 0.5


human_wins = 0
ai_wins = 0
play_as_x = True

fig, ax = plt.subplots()
clicked_move = None


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

opponent = ddqn_agent

def get_vanish_indices(env):
    x = env.move_history_x[0] if len(env.move_history_x) >= env.disappear_turn else None
    o = env.move_history_o[0] if len(env.move_history_o) >= env.disappear_turn else None
    return x, o

def render_board(board, title=""):
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
                    if pos == 0:
                        color = '#1c5ca8'
                    elif pos == 1:
                        color = '#4e8ed1'
                    else:
                        color = 'lightblue'
                else:
                    color = 'lightblue'
            elif val == -1:
                if idx in history_o:
                    pos = history_o.index(idx)
                    if pos == 0:
                        color = '#a8281c'
                    elif pos == 1:
                        color = '#d1634e'
                    else:
                        color = 'salmon'
                else:
                    color = 'salmon'

            ax.add_patch(plt.Rectangle((j, BOARD_SIZE - 1 - i), 1, 1, facecolor=color, edgecolor='black'))
            label = "X" if val == 1 else ("O" if val == -1 else "")
            ax.text(j + 0.5, BOARD_SIZE - 1 - i + 0.5, label, ha='center', va='center', fontsize=20)

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    fig.canvas.draw()

def on_click(event):
    global clicked_move
    if event.inaxes != ax:
        return
    col = int(event.xdata)
    row = int(event.ydata)
    idx = (BOARD_SIZE - 1 - row) * BOARD_SIZE + col
    if 0 <= idx < env.num_cells and env.board[idx] == 0:
        clicked_move = idx

fig.canvas.mpl_connect('button_press_event', on_click)

# === Game Loop ===
while True:
    env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)
    agent = opponent
    obs = env.reset()
    board = obs["board"]
    HUMAN_PLAYER = 1 if play_as_x else -1
    AI_PLAYER = -HUMAN_PLAYER

    done = False
    winner = None

    while not done:
        label = f"{type(agent).__name__} | You: {'X' if HUMAN_PLAYER == 1 else 'O'} | Human: {human_wins}   AI: {ai_wins}"

        render_board(board, title=label)
        current = int(obs["current_player"][0])

        if current == HUMAN_PLAYER:
            clicked_move = None
            while clicked_move is None and not done:
                plt.pause(0.1)
            move = clicked_move
        else:
            move = agent.act(obs)
            print(f"Agent plays: row {move // BOARD_SIZE + 1}, col {move % BOARD_SIZE + 1}")

        obs, reward, done, info = env.step(move)
        board = obs["board"]

    label = f"Your symbol: {'X' if HUMAN_PLAYER == 1 else 'O'} | Human: {human_wins}   AI: {ai_wins}"
    render_board(board, title=label)
    plt.pause(1)

    if reward == 1.0:
        if current == HUMAN_PLAYER:
            human_wins += 1
            print("You win!")
        else:
            ai_wins += 1
            print("AI wins!")
    else:
        print("Draw!")

    play_as_x = not play_as_x  # alternate starting side
