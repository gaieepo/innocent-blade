import copy
import sys

import pygame

from utils import (ACTIONS, FPS, GOLD_SPEED, HEIGHT, INITIAL_GOLD, REPAIR_COST,
                   SIMPLE_TECHS, SIMPLE_UNITS, TECHS, WIDTH)


class Faction:
    def __init__(self, side='white'):
        self.side = side
        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(SIMPLE_UNITS)
        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED
        self.army = []
        self.repairing = False
        self.state = {}

    def reset(self):
        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.gold = INITIAL_GOLD
        self.army = []
        # TODO process proper state

        return self.state

    def step(self, action):
        # process action
        # print(f'{self.side} receives action {action}')

        # state update
        # - active update
        # TODO handle tech / unit / idle ... separately

        if action in self.techs.keys():
            if (
                self.techs[action]['building']
                or self.techs[action]['built']
                or self.techs[action]['gold_cost'] > self.gold
            ):
                print(f'illegal action {action}')
            else:
                self.techs[action]['building'] = True
                self.gold -= self.techs[action]['gold_cost']
        elif action in self.units.keys():
            print(self.side, 'receives', action)
            # TODO action on unit logic

        # - passive update
        # 1. gold
        # 2. count down for techs
        # 3. count down for units
        # 4. units movement
        # 5. units health
        self.gold += self.gold_speed - self.repairing * REPAIR_COST

        for k, v in self.techs.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['built'] = True
                    v['building'] = False

        # prepare state

        for k, v in self.techs.items():
            self.state[k] = (
                not v['built']
                and not v['building']
                and v['gold_cost'] <= self.gold
            )

        self.state['gold'] = f'{self.gold:.0f}'

        return self.state


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('The Blade of Innocence')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.surface = pygame.Surface(self.screen.get_size())
        self.surface.fill((255, 255, 255))

        self.white = Faction('white')
        self.black = Faction('black')

        self.state = {}

    def reset(self):
        self.state['white'] = self.white.reset()
        self.state['black'] = self.black.reset()
        # TODO process aggregated state

        return self.state

    def render(self, mode='human', close=False):
        self.surface.fill((255, 255, 255))

        font = pygame.font.Font(None, 36)
        text = font.render(str(self.state['white']), 1, (10, 10, 10))
        # fps_text = font.render(str(self.clock.get_fps()), 1, (10, 10, 10))
        # fps_textpos = fps_text.get_rect()
        # fps_textpos.centery += 32

        self.surface.blit(text, text.get_rect())
        # self.surface.blit(fps_text, fps_textpos)
        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(FPS)

    def step(self, action):
        if action in ACTIONS:
            # TODO conditional action on each side
            self.state['white'] = self.white.step(action)
            self.state['black'] = self.black.step(action)
        else:
            print(f'invalid action {action}')

        # global movement
        # global health

        reward = 0
        done = False

        return self.state, reward, done, {}

    def close(self):
        return


if __name__ == "__main__":
    game = Game()
    state = game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_1:
                    game.step('barrack')
                elif event.key == pygame.K_2:
                    game.step('blacksmith')
                elif event.key == pygame.K_3:
                    game.step('windmill')
                elif event.key == pygame.K_4:
                    game.step('footman')
                elif event.key == pygame.K_5:
                    game.step('rifleman')
        else:
            game.step('null')
        game.render()

    game.close()
