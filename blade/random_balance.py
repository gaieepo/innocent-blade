import argparse
import random
from itertools import count

import torch

from game import Game
from utils import SEED, SIMPLE_ACTIONS, WHITE


###################################################
# general agents
###################################################
def random_agent(actions):
    return random.choice(actions)


if __name__ == "__main__":
    """ script for quick examination of game balance """
    parser = argparse.ArgumentParser(
        description='script for quick examination of game balance'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print timer'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true', help='debug or not'
    )
    parser.add_argument(
        '-s', '--simple', action='store_true', help='simple actions'
    )
    parser.add_argument(
        '-i',
        '--identical',
        action='store_true',
        help='identical action for both black and white',
    )
    args = parser.parse_args()

    # env settings
    # random.seed(SEED)
    # torch.manual_seed(SEED)

    # env setup
    game = Game(simple=args.simple, debug=args.debug)
    state = game.reset()

    # main loop
    episode_number = 0

    white_wins, black_wins = 0, 0

    while True:  # episode loop
        state = game.reset()

        white_action, black_action = 'noop', 'noop'

        for i_step in count():  # infinite game time when plays against human
            # game.render(white_action=white_action, black_action=black_action)

            # generate actions
            white_action = random_agent(game.available_actions)
            if args.identical:
                black_action = white_action
            else:
                black_action = random_agent(SIMPLE_ACTIONS)

            # update env
            state, reward, done, info = game.step(white_action, black_action)

            if args.verbose and i_step % 1000 == 0:
                print(i_step)

            if done:
                if reward[WHITE] == 1:
                    white_wins += 1
                else:
                    black_wins += 1
                white_win_rate = white_wins / (white_wins + black_wins)

                episode_number += 1

                print(
                    f"{episode_number} white: {white_wins} black: {black_wins} white rate: {100. * white_win_rate:.2f}% length: {info['length']}"
                )

                break
