import random
from itertools import count

import numpy as np
import pygame

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
def sigmoid(x):
    """ sigmoid 'squash' to interval [0, 1] """

    return 1.0 / (1.0 + np.exp(-x))


def numpy_agent(state, actions):
    h = np.dot(model['W1'], state)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)

    return actions[np.argmax(sigmoid(logp))]


if __name__ == "__main__":
    random.seed(42)

    # env setup
    game = Game(simple=True, prepro=True)
    state = game.reset()
    white_action, black_action = 'null', 'null'

    # naive numpy agent
    H = 200  # number of hidden layer neurons
    D = 103  # input dimensionality (# of features)
    episode_number = 0
    render = False
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # xavier
    model['W2'] = np.random.randn(len(game.available_actions), H) / np.sqrt(H)

    white_wins, black_wins = 0, 0

    while True:  # for c in count():
        if render:
            game.render(white_action=white_action, black_action=black_action)

        white_action = numpy_agent(state['white'], game.available_actions)
        black_action = random_agent(game.available_actions)

        # print(c, white_action, black_action)

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
