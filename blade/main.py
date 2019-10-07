import multiprocessing
import multiprocessing.connection
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from game import Game

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')


class GameWrapper:
    def __init__(self, seed):
        self.env = Game()
        self.obs_8 = np.zeros((8, 199))
        self.rewards = []

    def step(self, white_action, black_action):
        obs, reward, done, info = self.env.step(white_action, black_action)
        self.rewards.append(reward)
        if done:
            episode_info = {
                'reward': sum(self.rewards),
                'length': len(self.rewards),
            }
            self.reset()
        else:
            episode_info = None
            self.obs_8 = np.roll(self.obs_8, shift=-1, axis=0)
            self.obs_8[-1, :] = obs

        return self.obs_8.reshape(-1), reward, done, episode_info

    def reset(self):
        obs = self.env.reset()
        self.obs_8[0, :] = obs
        self.obs_8[1, :] = obs
        self.obs_8[2, :] = obs
        self.obs_8[3, :] = obs
        self.obs_8[4, :] = obs
        self.obs_8[5, :] = obs
        self.obs_8[6, :] = obs
        self.obs_8[7, :] = obs
        self.rewards = []
        return self.obs_8.reshape(-1)


def worker_process(remote, seed):
    env = GameWrapper(seed)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            remote.send(env.step(data))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=worker_process, args=(parent, seed)
        )
        self.process.start()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(199 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.pi_logits = nn.Linear(128, 17)
        self.value = nn.Linear(128, 1)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value


def obs_to_torch(obs):
    return torch.tensor(obs, dtype=torch.float32, device=device)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self, samples, learning_rate, clip_range):
        pass


class Main:
    def __init__(self):
        self.gamma = 0.99
        self.lamda = 0.95
        self.updates = 10000
        self.epochs = 4
        self.n_workers = 8
        self.worker_steps = 128
        self.n_mini_batch = 4
        self.batch_size = self.n_workers * self.worker_steps  # 8 * 128 = 1024
        self.mini_batch_size = (
            self.batch_size // self.n_mini_batch
        )  # 8 * 128 // 4 = 256
        assert self.batch_size % self.n_mini_batch == 0

        self.workers = [Worker(42 + i) for i in range(self.n_workers)]

        self.obs = np.zeros((self.n_workers, 8, 199), dtype=np.uint8)

        for worker in self.workers:
            worker.child.send(('reset', None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        self.model = Model()
        self.model.to(device)
        self.trainer = Trainer(self.model)

    def sample(self):
        pass

    def _calc_advantages(self, dones, rewards, values):
        pass

    def train(self, samples, learning_rate, clip_range):
        train_info = []

        for _ in range(self.epochs):
            indexes = torch.randperm(self.batch_size)

    def run_training_loop(self):
        episode_info = deque(maxlen=100)

        for update in range(self.updates):
            time_start = time.time()

            # lr schedule
            progress = update / self.updates
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            samples, sample_episode_info = self.sample()
            self.train(samples, learning_rate, clip_range)

            time_end = time.time()

            fps = int(self.batch_size / (time_end - time_start))
            episode_info.extend(sample_episode_info)
            reward_mean, length_mean = Main._get_mean_episode_info(
                episode_info
            )
            print(
                f'{update:4}: fps={fps:3} reward={reward_mean:.2f} length={length_mean:.3f}'
            )

    @staticmethod
    def _get_mean_episode_info(episode_info):
        if len(episode_info) > 0:
            return (
                np.mean([info['reward'] for info in episode_info]),
                np.mean([info['length'] for info in episode_info]),
            )
        else:
            return np.nan, np.nan

    def destroy(self):
        for worker in self.workers:
            worker.child.send(('close', None))


if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
    m.destroy()
