import multiprocessing
import multiprocessing.connection
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from models import Model, obs_to_torch
from utils import SIMPLE_ACTIONS, WHITE
from wrapper import GameWrapper

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')


def worker_process(remote, seed):
    env = GameWrapper(seed)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            remote.send(env.step(*data))
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


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self, samples, learning_rate, clip_range):
        sampled_obs = samples['obs']
        sampled_action = samples['actions']
        sampled_return = samples['values'] + samples['advantages']
        sampled_normalized_advantage = Trainer._normalize(
            samples['advantages']
        )
        sampled_neg_log_pi = samples['neg_log_pis']
        sampled_value = samples['values']

        pi, value = self.model(sampled_obs)

        neg_log_pi = -pi.log_prob(sampled_action)
        ratio = torch.exp(sampled_neg_log_pi - neg_log_pi)

        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(
            ratio * sampled_normalized_advantage,
            clipped_ratio * sampled_normalized_advantage,
        )
        policy_reward = policy_reward.mean()

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        clipped_value = sampled_value + (value - sampled_value).clamp(
            min=-clip_range, max=-clip_range
        )
        vf_loss = torch.max(
            (value - sampled_return) ** 2,
            (clipped_value - sampled_return) ** 2,
        )
        vf_loss = 0.5 * vf_loss.mean()

        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        approx_kl_divergence = (
            0.5 * ((neg_log_pi - sampled_neg_log_pi) ** 2).mean()
        )
        clip_fraction = (
            (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()
        )

        return [
            policy_reward,
            vf_loss,
            entropy_bonus,
            approx_kl_divergence,
            clip_fraction,
        ]

    @staticmethod
    def _normalize(adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)


class Main:
    def __init__(self):
        self.gamma = 0.99
        self.lamda = 0.95
        self.updates = 100000
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

        self.obs = np.zeros((self.n_workers, 8 * 208), dtype=np.uint8)

        for worker in self.workers:
            worker.child.send(('reset', None))
        for i, worker in enumerate(self.workers):
            self.obs[i], _ = worker.child.recv()

        self.model = Model()
        self.model.to(device)
        if os.path.exists('weight.pth'):
            self.model.load_state_dict(torch.load('weight.pth'))
            print('loaded weight.pth')
        self.model.train()
        self.trainer = Trainer(self.model)

    def sample(self):
        rewards = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32
        )
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros(
            (self.n_workers, self.worker_steps, 8 * 208), dtype=np.uint8
        )
        neg_log_pis = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32
        )
        values = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32
        )
        episode_infos = []

        for t in range(self.worker_steps):
            obs[:, t] = self.obs

            pi, v = self.model(obs_to_torch(self.obs))
            values[:, t] = v.cpu().data.numpy()

            a = pi.sample()
            actions[:, t] = a.cpu().data.numpy()

            neg_log_pis[:, t] = -pi.log_prob(a).cpu().data.numpy()

            for w, worker in enumerate(self.workers):
                worker.child.send(
                    ('step', [actions[w, t], random.choice(SIMPLE_ACTIONS)])
                )

            for w, worker in enumerate(self.workers):
                white_obs, _, reward, done, info = worker.child.recv()
                self.obs[w] = white_obs
                rewards[w, t] = reward[WHITE]
                dones[w, t] = done

                if info:
                    info['obs_white'] = obs[w, t]
                    episode_infos.append(info)

        advantages = self._calc_advantages(dones, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'neg_log_pis': neg_log_pis,
            'advantages': advantages,
        }

        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat, episode_infos

    def _calc_advantages(self, dones, rewards, values):
        advantages = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32
        )
        last_advantage = 0

        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]
            last_advantage = delta + self.gamma * self.lamda * last_advantage

            advantages[:, t] = last_advantage
            last_value = values[:, t]

        return advantages

    def train(self, samples, learning_rate, clip_range):
        train_info = []

        for _ in range(self.epochs):
            indexes = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start:end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                res = self.trainer.train(
                    learning_rate=learning_rate,
                    clip_range=clip_range,
                    samples=mini_batch,
                )

                train_info.append(res)

        return np.mean(train_info, axis=0)

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
            reward_mean, _, length_mean = Main._get_mean_episode_info(
                episode_info
            )
            print(
                f'{update:4}: fps={fps:3} reward={reward_mean:.2f} length={length_mean:.3f}'
            )

            # save latest model
            if update % 100 == 0:
                torch.save(self.model.state_dict(), 'weight.pth')

    @staticmethod
    def _get_mean_episode_info(episode_info):
        if len(episode_info) > 0:
            return (
                np.mean([info['reward_white'] for info in episode_info]),
                np.mean([info['reward_black'] for info in episode_info]),
                np.mean([info['length'] for info in episode_info]),
            )
        else:
            return np.nan, np.nan, np.nan

    def destroy(self):
        for worker in self.workers:
            worker.child.send(('close', None))


if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
    m.destroy()
