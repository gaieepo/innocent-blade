import copy

from utils import GOLD_SPEED, INITIAL_GOLD, REPAIR_COST, TECHS


class Faction:
    def __init__(self):
        self.tech = copy.deepcopy(TECHS)
        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED
        self.units = []
        self.repairing = False
        self.state = {}

    def reset(self):
        self.tech = copy.deepcopy(TECHS)
        self.gold = INITIAL_GOLD
        self.units = []
        # TODO process other state

        return self.state

    def step(self, action):
        # process action
        # state update
        # - passive update
        self.gold += self.gold_speed - self.repairing * REPAIR_COST
        # - active update

        self.state['gold'] = self.gold

        return self.state


class Game:
    def __init__(self):
        self.white = Faction()
        self.black = Faction()

        self.state = None

    def reset(self):
        white_state = self.white.reset()
        black_state = self.black.reset()
        # TODO process aggregated state

        self.state = [white_state, black_state]

        return self.state

    def render(self, mode='human', close=False):
        return

    def step(self, action):
        # conditional action on each side
        self.white.step(action)
        self.black.step(action)

        reward = 0
        done = False

        return self.state, reward, done, {}

    def close(self):
        return


if __name__ == "__main__":
    game = Game()
    state = game.reset()

    while True:
        state, reward, done, info = game.step(None)
        print(state)

    game.close()
