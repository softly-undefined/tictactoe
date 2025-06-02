# Vanishing TicTacToe

Vanishing Tic Tac Toe presents an intriguing challenge in the domain of game-playing AI due to its deceptively simple rules and compact state space, which nonetheless give rise to rich strategic complexity. In this project, we investigate the application of Deep Reinforcement Learning (DRL) techniques to this niche game, aiming to develop an agent that achieves superhuman performance. Despite the limited number of possible game states, the dynamic nature of Vanishing Tic Tac Toe demands nuanced decision-making and long-term planning, making it a compelling testbed for evaluating the effectiveness of DRL methods in low-dimensional yet nontrivial environments. Our approach involves training agents using state-of-the-art deep learning architectures and reinforcement learning algorithms, with a focus on sample efficiency and convergence behavior. Ideally, the resulting agent will demonstrate strategic mastery surpassing that of human players, offering insights into how deep learning can uncover optimal policies in constrained yet complex game environments.

## Files within vanishing_tictactoe

### Environment

- vanishing_tictactoe.py: contains the gym environment which the experiment is based upon.
### Simulations

- human.py: allows the user to play against any given agent.
- vis_test.py: allows the user to view a single game between any two agents.
- sim_test.py: allows the user to simulate any number of games between any two agents.
### Agents

- random_agent.py: an agent which moves completely randomly.
- simplerulebased_agent.py: emphasizes making adjacent moves- the weakest of the rule based agent.
- moderaterulebased_agent.py: plays defensively and opportunistically—blocking opponent wins and taking winning moves when available, otherwise moving randomly.
- complexrulebased_agent.py: evaluates win conditions, blocks, forks, and board control. Prioritizes safe winning plays, blocks opponent forks, and prefers center/corner control. The strongest rule-based agent.
- ddqn_agent.py: Our implementation of a DDQN Agent. The code to load a specific model is in the different Simulation files.
### Training

- train.py: contains the code used to train our DDQN agent.

### Logs

- logs/: contains the training logs
- models/: conatins the model logs