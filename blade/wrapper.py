import numpy as np

from game import Game
from utils import BLACK, WHITE


class GameWrapper:
    def __init__(self, seed=None, debug=False):
        self.env = Game(debug=debug)
        self.env.seed(seed)
        self.obs_white_8 = np.zeros((8, 208))
        self.obs_black_8 = np.zeros((8, 208))
        self.rewards_white = []
        self.rewards_black = []

    def step(self, white_action, black_action):
        obs, reward, done, info = self.env.step(white_action, black_action)
        self.rewards_white.append(reward[WHITE])
        self.rewards_black.append(reward[BLACK])
        if done:
            episode_info = {
                'reward_white': sum(self.rewards_white),
                'reward_black': sum(self.rewards_black),
                'length': len(self.rewards_white),
            }
            self.reset()
        else:
            episode_info = None
            self.obs_white_8 = np.roll(self.obs_white_8, shift=-1, axis=0)
            self.obs_black_8 = np.roll(self.obs_white_8, shift=-1, axis=0)
            self.obs_white_8[-1, :] = obs['white']
            self.obs_black_8[-1, :] = obs['black']

        return (
            self.obs_white_8.reshape(-1),
            self.obs_black_8.reshape(-1),
            reward,
            done,
            episode_info,
        )

    def reset(self):
        obs = self.env.reset()
        self.obs_white_8[0, :] = obs['white']
        self.obs_white_8[1, :] = obs['white']
        self.obs_white_8[2, :] = obs['white']
        self.obs_white_8[3, :] = obs['white']
        self.obs_white_8[4, :] = obs['white']
        self.obs_white_8[5, :] = obs['white']
        self.obs_white_8[6, :] = obs['white']
        self.obs_white_8[7, :] = obs['white']
        self.obs_black_8[0, :] = obs['black']
        self.obs_black_8[1, :] = obs['black']
        self.obs_black_8[2, :] = obs['black']
        self.obs_black_8[3, :] = obs['black']
        self.obs_black_8[4, :] = obs['black']
        self.obs_black_8[5, :] = obs['black']
        self.obs_black_8[6, :] = obs['black']
        self.obs_black_8[7, :] = obs['black']
        self.rewards_white = []
        self.rewards_black = []
        return self.obs_white_8.reshape(-1), self.obs_black_8.reshape(-1)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    @property
    def available_actions(self):
        return self.env.available_actions

    def close(self):
        self.env.close()

    def set_fps(self, *args, **kwargs):
        self.env.viewer.set_fps(*args, **kwargs)


if __name__ == "__main__":
    gw = GameWrapper(42)
    print(gw.reset())
    print(gw.step('null', 'null'))
