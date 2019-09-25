import os
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from game import Game
from utils import EPS, GAMMA, LR, WHITE


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


def torch_agent(state, actions, save=True):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = best_policy(state)
    m = Categorical(probs)
    action = m.sample()
    best_policy.saved_log_probs.append(m.log_prob(action))

    return actions[action.item()]


def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    for r in best_policy.rewards[::-1]:
        R = r + GAMMA * R  # discount reward
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + EPS)

    for log_prob, R in zip(best_policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del best_policy.rewards[:]
    del best_policy.saved_log_probs[:]

    return policy_loss.item()


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)


if __name__ == "__main__":
    """ Self-play training procedure """
    # env settings (should not use seeded random during training)
    # random.seed(SEED)
    # torch.manual_seed(SEED)

    # env setup
    game = Game(simple=False)
    state = game.reset()

    # policy models
    best_policy = Policy(
        input_dim=state['white'].shape[0],
        output_dim=len(game.available_actions),
    )
    latest_policy = Policy(
        input_dim=state['white'].shape[0],
        output_dim=len(game.available_actions),
    )

    if os.path.exists('best.pth'):  # load best
        latest_policy.load_state_dict(torch.load('best.pth'))
        best_policy.load_state_dict(torch.load('best.pth'))
    else:
        weight_init(latest_policy)
        weight_init(best_policy)
        torch.save(best_policy.state_dict(), 'best.pth')

    # hyper-parameters
    optimizer = optim.Adam(best_policy.parameters(), lr=LR)

    # main loop
    episode_number = 0

    while True:
        # 1. self-play 1000 games
        print('Start self-play!!!')
        latest_policy.eval()
        best_policy.train()

        white_wins, black_wins = 0, 0

        for i_selfplay_episode in range(1, 1001):
            prev_state = {'white': None, 'black': None}
            state = game.reset()

            # load best model
            latest_policy.load_state_dict(torch.load('best.pth'))
            best_policy.load_state_dict(torch.load('best.pth'))

            for t in count(1):  # TODO unlimited game time ???
                # preprocess input state
                input_state = {
                    'white': np.concatenate(
                        [state['white'], prev_state['white']]
                    )
                    if prev_state['white'] is not None
                    else np.concatenate(
                        [state['white'], np.zeros_like(state['white'])]
                    ),
                    'black': np.concatenate(
                        [state['black'], prev_state['black']]
                    )
                    if prev_state['black'] is not None
                    else np.concatenate(
                        [state['black'], np.zeros_like(state['black'])]
                    ),
                }
                prev_state = state

                # generate actions
                white_action = torch_agent(
                    input_state['white'], game.available_actions
                )
                black_action = torch_agent(
                    input_state['black'], game.available_actions, save=False
                )

                # update env
                state, reward, done, info = game.step(
                    white_action, black_action
                )

                # record reward
                best_policy.rewards.append(reward[WHITE])

                if done:
                    returned_policy_loss = finish_episode()

                    if reward[WHITE] == 1:
                        white_wins += 1
                    else:
                        black_wins += 1
                    selfplay_white_win_rate = white_wins / (
                        white_wins + black_wins
                    )

                    episode_number += 1

                    print(
                        f'{episode_number} [{i_selfplay_episode}/1000-{t}] loss: {returned_policy_loss:.2f} white: {white_wins} black: {black_wins} white rate: {100. * selfplay_white_win_rate:.2f}%'
                    )

                    # IMPORTANT copy weight from best to latest
                    latest_policy.load_state_dict(best_policy.state_dict())

                    break

        # 2. 400 matches between latest and best
        print('Start matches!!!')
        best_policy.eval()

        white_wins, black_wins = 0, 0
        match_white_win_rate = 0.0

        for i_match_episode in range(1, 401):
            prev_state = {'white': None, 'black': None}
            state = game.reset()

            for t in count(1):  # TODO unlimited game time ???
                # preprocess input state
                input_state = {
                    'white': np.concatenate(
                        [state['white'], prev_state['white']]
                    )
                    if prev_state['white'] is not None
                    else np.concatenate(
                        [state['white'], np.zeros_like(state['white'])]
                    ),
                    'black': np.concatenate(
                        [state['black'], prev_state['black']]
                    )
                    if prev_state['black'] is not None
                    else np.concatenate(
                        [state['black'], np.zeros_like(state['black'])]
                    ),
                }
                prev_state = state

                # generate actions
                with torch.no_grad():
                    white_action = torch_agent(
                        best_policy,
                        input_state['white'],
                        game.available_actions,
                    )
                    black_action = torch_agent(
                        best_policy,
                        input_state['black'],
                        game.available_actions,
                    )

                # update env
                state, reward, done, info = game.step(
                    white_action, black_action
                )

                # record reward
                latest_policy.rewards.append(reward[WHITE])

                if done:
                    if reward[WHITE] == 1:
                        white_wins += 1
                    else:
                        black_wins += 1
                    match_white_win_rate = white_wins / (
                        white_wins + black_wins
                    )

                    print(
                        f'[{i_match_episode}/400-{t:6d}] white: {white_wins} black: {black_wins} white rate: {100. * match_white_win_rate:.2f}%'
                    )

                    break

        if match_white_win_rate > 0.55:
            torch.save(latest_policy.state_dict(), 'best.pth')
