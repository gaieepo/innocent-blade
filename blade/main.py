import random
from itertools import count

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from game import Game


###################################################
# general agents
###################################################
def human_agent():
    action = 'null'

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                action = 'close'
            elif event.key == pygame.K_1:
                action = 'barrack'
            elif event.key == pygame.K_2:
                action = 'blacksmith'
            elif event.key == pygame.K_3:
                action = 'windmill'
            elif event.key == pygame.K_4:
                action = 'footman'
            elif event.key == pygame.K_5:
                action = 'rifleman'
            elif event.key == pygame.K_SPACE:
                action = 'forward'
            elif event.key == pygame.K_BACKSPACE:
                action = 'backward'
            elif event.key == pygame.K_r:
                action = 'repair'
            elif event.key == pygame.K_t:
                action = 'stop_repair'

    return action


def random_agent(actions):
    return random.choice(actions)


###################################################
# PG related
###################################################
class Policy(nn.Module):
    def __init__(self, output_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(103, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)

        return torch.softmax(action_scores, dim=1)


def torch_agent(state):
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    return action.item()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # env setup
    game = Game(simple=True, prepro=True)
    state = game.reset()
    white_action, black_action = 'null', 'null'

    # naive torch agent
    episode_number = 0
    render = False
    white_wins, black_wins = 0, 0

    policy = Policy(output_dim=len(game.available_actions))
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    eps = np.finfo(np.float32).eps.item()

    while True:  # for c in count():
        if render:
            game.render(white_action=white_action, black_action=black_action)

        # TODO preprocess state curr and prev

        white_action = torch_agent(state['white'])
        black_action = random_agent(game.available_actions)

        if white_action == 'close' or black_action == 'close':
            break

        state, reward, done, info = game.step(white_action, black_action)

        if done:
            # an episode finished
            episode_number += 1
            state = game.reset()

        if reward[0] != 0:
            # suffix = '!!!' if reward[0] == 1 else ''
            suffix = ''

            if reward[0] == 1:
                white_wins += 1
            else:
                black_wins += 1
            print(
                f'Ep: {episode_number} reward: {reward}{suffix} white: {white_wins} black: {black_wins} white rate: {100. * white_wins / (white_wins + black_wins)}%'
            )

    game.close()
