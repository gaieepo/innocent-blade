import argparse
import random

from game import Game
from utils import WHITE


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
    game = Game(simple=args.simple)
    state = game.reset()

    # main loop
    episode_number = 0

    white_wins, black_wins = 0, 0

    while True:  # episode loop
        state = game.reset()

        while True:  # infinite game time when plays against human

            # generate actions
            white_action = random_agent(game.available_actions)
            if args.identical:
                black_action = white_action
            else:
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
