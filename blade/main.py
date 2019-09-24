import os
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
from utils import GAMMA, MAX_GLOBAL_TIME, SEED, WHITE, LR


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
    policy.saved_log_probs.append(m.log_prob(action))

    return actions[action.item()]


def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    for r in latest_policy.rewards[::-1]:
        R = r + GAMMA * R  # discount reward
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(latest_policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del latest_policy.rewards[:]
    del latest_policy.saved_log_probs[:]

    return policy_loss.item()


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)


if __name__ == "__main__":
    # env settings
    random.seed(SEED)
    torch.manual_seed(SEED)

    # env setup
    game = Game(simple=False)
    state = game.reset()
    prev_state = {'white': None, 'black': None}

    # naive torch agent settings
    render = False
    resume = False
    updated = False

    # policy models
    best_policy = Policy(
        input_dim=state['white'].shape[0],
        output_dim=len(game.available_actions),
    )
    latest_policy = Policy(
        input_dim=state['white'].shape[0],
        output_dim=len(game.available_actions),
    )

    # misc

    if resume:
        if os.path.exists('latest.pth'):
            latest_policy.load_state_dict(torch.load('latest.pth'))
        elif os.path.exists('best.pth'):
            latest_policy.load_state_dict(torch.load('best.pth'))
        else:
            weight_init(latest_policy)

    if os.path.exists('best.pth'):
        best_policy.load_state_dict(torch.load('best.pth'))
    elif os.path.exists('latest.pth'):
        best_policy.load_state_dict(torch.load('latest.pth'))
    else:
        weight_init(best_policy)

    latest_policy.train()
    best_policy.eval()

    # hyper-parameters
    optimizer = optim.Adam(latest_policy.parameters(), lr=LR)
    eps = np.finfo(np.float32).eps.item()

    # main loop
    # running_reward = 0
    white_wins, black_wins = 0, 0

    for i_episode in count(1):
        # state, ep_reward = game.reset(), 0
        state = game.reset()
        white_action, black_action = 'null', 'null'

        if updated:
            # elite best
            updated = False
            best_policy.load_state_dict(torch.load('best.pth'))

        for t in range(1, int(MAX_GLOBAL_TIME)):  # finite loop while learning
            if render:
                game.render(
                    white_action=white_action, black_action=black_action
                )

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
            white_action = torch_agent(
                latest_policy, input_state['white'], game.available_actions
            )
            with torch.no_grad():
                black_action = torch_agent(
                    best_policy, input_state['black'], game.available_actions
                )

            if white_action == 'close' or black_action == 'close':
                game.close()

            # update env
            state, reward, done, info = game.step(white_action, black_action)

            # record reward
            latest_policy.rewards.append(reward[WHITE])
            # ep_reward += reward[WHITE]

            if done:
                break

        if not done:
            # when exceed time white loses
            latest_policy.rewards.append(-1)
            # ep_reward += -1

        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        returned_policy_loss = finish_episode()

        if reward[WHITE] == 1:
            white_wins += 1
        else:
            black_wins += 1
        white_win_rate = white_wins / (white_wins + black_wins)

        if i_episode % 50 == 0:
            torch.save(latest_policy.state_dict(), 'latest.pth')

        if white_win_rate > 0.6:
            updated = True
            torch.save(latest_policy.state_dict(), 'best.pth')
            print('Updated best policy!!!')

        print(
            f'Ep: {i_episode:3d} ends at time {t:6d} loss: {returned_policy_loss:.2f} white: {white_wins} black: {black_wins} white rate: {100. * white_win_rate:.2f}%'
        )
