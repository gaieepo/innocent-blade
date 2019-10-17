import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import STATE_SIZE

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.pi_logits = nn.Linear(256, 17)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)  # TODO tanh to squeeze [-1, 1]

        return pi, value


def obs_to_torch(obs):
    return torch.tensor(obs, dtype=torch.float32, device=device)


def weights_init(m):
    """ zero everything, looks neat """
    if isinstance(m, nn.Linear):
        m.weight.data.zero_()
        m.bias.data.zero_()


if __name__ == "__main__":
    model = Model()
    print(model.state_dict())
    model.apply(weights_init)
    print(model.state_dict())
    outs = model(torch.rand(STATE_SIZE * 8))
    print(outs)
