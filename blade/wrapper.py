import random
import sys
import time

import numpy as np
import pygame

import rendering
from game import Game
from utils import FULL_ACTIONS, LEFT, RIGHT, STATE_SIZE


class GameWrapper:
    def __init__(self, seed=None, debug=False, max_limit=10000):
        self.env = Game(debug=debug)
        self.env.seed(seed)
        self.viewer = rendering.Viewer('The Blade of Innocence')
        self.max_limit = max_limit
        self.obs_left_8 = np.zeros((8, STATE_SIZE))
        self.obs_right_8 = np.zeros((8, STATE_SIZE))
        self.rewards_left = []
        self.rewards_right = []

    def step(self, left_action, right_action):
        """ apply maximum time limit for training game e.g. 10000 steps """
        obs, reward, done, info = self.env.step(left_action, right_action)
        self.rewards_left.append(reward[LEFT])
        self.rewards_right.append(reward[RIGHT])

        if len(self.rewards_left) >= self.max_limit:
            # exceeds limit both lose
            # reward is either 0 or 1, blade specific
            self.rewards_left[-1] = -1
            self.rewards_right[-1] = -1
            reward = (-1, -1)
            done = True  # will be resetted

        if done:
            episode_info = {
                'reward_left': sum(self.rewards_left),
                'reward_right': sum(self.rewards_right),
                'length': len(self.rewards_left),
            }
            self.reset()  # reset after done
        else:
            episode_info = None
            self.obs_left_8 = np.roll(self.obs_left_8, shift=-1, axis=0)
            self.obs_right_8 = np.roll(self.obs_left_8, shift=-1, axis=0)
            self.obs_left_8[-1, :] = obs['left']
            self.obs_right_8[-1, :] = obs['right']

        return (self.obs_left_8.reshape(-1), self.obs_right_8.reshape(-1), reward, done, episode_info)

    def reset(self):
        obs = self.env.reset()
        self.obs_left_8[0, :] = obs['left']
        self.obs_left_8[1, :] = obs['left']
        self.obs_left_8[2, :] = obs['left']
        self.obs_left_8[3, :] = obs['left']
        self.obs_left_8[4, :] = obs['left']
        self.obs_left_8[5, :] = obs['left']
        self.obs_left_8[6, :] = obs['left']
        self.obs_left_8[7, :] = obs['left']
        self.obs_right_8[0, :] = obs['right']
        self.obs_right_8[1, :] = obs['right']
        self.obs_right_8[2, :] = obs['right']
        self.obs_right_8[3, :] = obs['right']
        self.obs_right_8[4, :] = obs['right']
        self.obs_right_8[5, :] = obs['right']
        self.obs_right_8[6, :] = obs['right']
        self.obs_right_8[7, :] = obs['right']
        self.rewards_left = []
        self.rewards_right = []
        return self.obs_left_8.reshape(-1), self.obs_right_8.reshape(-1)

    def render(self, left_action, right_action):
        left_action, right_action = self.env.parse(left_action, right_action)
        self.viewer.render(game=self.env, left_action=left_action, right_action=right_action)

    @property
    def available_actions(self):
        return self.env.available_actions

    def close(self):
        self.viewer.close()
        sys.exit(0)

    def set_fps(self, *args, **kwargs):
        self.viewer.set_fps(*args, **kwargs)


def human_agent():
    action = 'noop'

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            action = 'close'
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
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


if __name__ == "__main__":
    gw = GameWrapper(seed=42)
    gw.reset()
    while True:
        gw.render('noop', 'noop')

        left_action = human_agent()

        if left_action == 'close':
            gw.close()
            break
        elif left_action == 'min_fps':
            gw.set_fps(step=-1)
            left_action = 'noop'
        elif left_action == 'up_fps':
            gw.set_fps(step=5)
            left_action = 'noop'
        elif left_action == 'down_fps':
            gw.set_fps(step=-5)
            left_action = 'noop'
        elif left_action == 'reset_fps':
            gw.set_fps()
            left_action = 'noop'

        *_, done, _ = gw.step(left_action, random.choice(FULL_ACTIONS))
        if done:
            gw.close()
            break
