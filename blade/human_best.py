import os
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from game import Game
from utils import SEED, WHITE


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
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, output_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.fc2(x)

        return torch.softmax(action_scores, dim=1)


def torch_agent(policy, state, actions):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()

    return actions[action.item()]


if __name__ == "__main__":
    """ Self-play training procedure """
    # env settings (should not use seeded random during training)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # env setup
    game = Game(simple=False)
    state = game.reset()

    # policy models
    policy = Policy(
        input_dim=state['white'].shape[0],
        output_dim=len(game.available_actions),
    )

    if os.path.exists('best.pth'):  # load best
        policy.load_state_dict(torch.load('best.pth'))
        print('Best weights loaded!!!')
    else:
        raise FileNotFoundError('best.pth is not found')

    # set policy untrainable
    policy.eval()

    # main loop
    episode_number = 0

    white_wins, black_wins = 0, 0

    while True:  # episode loop
        prev_state = {'white': None, 'black': None}
        state = game.reset()
        white_action, black_action = 'null', 'null'

        while True:  # infinite game time when plays against human
            game.render(white_action=white_action, black_action=black_action)

            # preprocess input state
            input_state = {
                'white': np.concatenate([state['white'], prev_state['white']])
                if prev_state['white'] is not None
                else np.concatenate(
                    [state['white'], np.zeros_like(state['white'])]
                ),
                'black': np.concatenate([state['black'], prev_state['black']])
                if prev_state['black'] is not None
                else np.concatenate(
                    [state['black'], np.zeros_like(state['black'])]
                ),
            }
            prev_state = state

            # generate actions
            white_action = human_agent()
            with torch.no_grad():
                black_action = torch_agent(
                    policy, input_state['black'], game.available_actions
                )

            # close game and app
            if white_action == 'close':
                game.close()

            # update env
            state, reward, done, info = game.step(white_action, black_action)

            if done:
                if reward[WHITE] == 1:
                    white_wins += 1
                else:
                    black_wins += 1
                white_win_rate = white_wins / (white_wins + black_wins)

                episode_number += 1

                print(
                    f'{episode_number} white: {white_wins} black: {black_wins} white rate: {100. * white_win_rate:.2f}%'
                )

                break
