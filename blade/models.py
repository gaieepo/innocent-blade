import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')


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
