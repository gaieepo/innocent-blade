import argparse
import os
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from game import Game
from models import Model, obs_to_torch
from utils import BLACK, FULL_ACTIONS, WHITE
from wrapper import GameWrapper


###################################################
# general agents
###################################################
def human_agent():
    action = 'noop'

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
                action = 'steel_blade'
            elif event.key == pygame.K_5:
                action = 'long_barrelled_gun'
            elif event.key == pygame.K_6:
                action = 'keep'
            elif event.key == pygame.K_7:
                action = 'watchtower'
            elif event.key == pygame.K_8:
                action = 'monastery'
            elif event.key == pygame.K_9:
                action = 'transport'
            elif event.key == pygame.K_z:
                action = 'footman'
            elif event.key == pygame.K_x:
                action = 'rifleman'
            elif event.key == pygame.K_c:
                action = 'monk'
            elif event.key == pygame.K_SPACE:
                action = 'forward'
            elif event.key == pygame.K_BACKSPACE:
                action = 'backward'
            elif event.key == pygame.K_r:
                action = 'repair'
            elif event.key == pygame.K_t:
                action = 'stop_repair'
            elif event.key == pygame.K_LEFT:
                action = 'min_fps'
            elif event.key == pygame.K_RIGHT:
                action = 'reset_fps'
            elif event.key == pygame.K_UP:
                action = 'up_fps'
            elif event.key == pygame.K_DOWN:
                action = 'down_fps'

    return action


def random_agent(actions):
    return random.choice(actions)


if __name__ == "__main__":
    """ Self-play training procedure """
    parser = argparse.ArgumentParser(description='self play against best')
    parser.add_argument(
        '-d', '--debug', action='store_true', help='debug or not'
    )
    args = parser.parse_args()

    # env settings (should not use seeded random during training)
    # random.seed(SEED)
    # torch.manual_seed(SEED)

    # env setup
    gw = GameWrapper(debug=args.debug)

    device = (
        torch.device('cuda:2')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    model = Model()
    model.to(device)

    BEST_WEIGHT = 'best.pth'

    if os.path.exists(BEST_WEIGHT):  # load best
        model.load_state_dict(torch.load(BEST_WEIGHT))
        print('Best weights loaded!!!')
    else:
        raise FileNotFoundError(f'{BEST_WEIGHT} is not found')

    # set policy untrainable
    model.eval()

    # main loop
    episode_number = 0
    white_wins, black_wins = 0, 0

    while True:  # episode loop
        white_state, black_state = gw.reset()
        white_action, black_action = 'noop', 'noop'

        while True:  # infinite game time when plays against human
            gw.render(white_action=white_action, black_action=black_action)

            # human actions
            white_action = human_agent()

            # close game and app
            if white_action == 'close':
                gw.close()
            elif white_action == 'min_fps':
                gw.set_fps(step=-1)
                white_action = 'noop'
            elif white_action == 'up_fps':
                gw.set_fps(step=5)
                white_action = 'noop'
            elif white_action == 'down_fps':
                gw.set_fps(step=-5)
                white_action = 'noop'
            elif white_action == 'reset_fps':
                gw.set_fps()
                white_action = 'noop'

            # model actions
            with torch.no_grad():
                pi, v = model(obs_to_torch(black_state))
                black_action = pi.sample().item()

            # update env
            white_state, black_state, reward, done, info = gw.step(
                white_action, black_action
            )

            if done:
                if reward[WHITE] == 1:
                    white_wins += 1
                if reward[BLACK] == 1:
                    black_wins += 1
                white_win_rate = white_wins / (white_wins + black_wins)

                episode_number += 1

                print(
                    f"{episode_number} white: {white_wins} black: {black_wins} white rate: {100. * white_win_rate:.2f}% length: {info['length']}"
                )

                break
