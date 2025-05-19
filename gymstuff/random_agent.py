import gym
import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv

# Random agent
# Very simple, just samples legal_moves to choose a random move.


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        legal_actions = [i for i in range(len(observation)) if observation[i] == 0]
        if not legal_actions:
            return self.action_space.sample()
        return np.random.choice(legal_actions)
