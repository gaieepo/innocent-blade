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
def random_agent(actions):
    return random.choice(actions)


if __name__ == "__main__":
    # env settings
    random.seed(SEED)
    torch.manual_seed(SEED)

    # env setup
    game = Game(simple=True)
    state = game.reset()

    # main loop
    episode_number = 0

    white_wins, black_wins = 0, 0

    while True:  # episode loop
        state = game.reset()

        while True:  # infinite game time when plays against human

            # generate actions
            white_action = random_agent(game.available_actions)
            black_action = random_agent(game.available_actions)

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
