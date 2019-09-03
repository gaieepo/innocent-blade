import copy
import sys

import pygame

from utils import FPS, GOLD_SPEED, INITIAL_GOLD, REPAIR_COST, TECHS


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
        pygame.init()
        pygame.display.set_caption('The Blade of Innocence')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((640, 480))
        self.surface = pygame.Surface(self.screen.get_size())
        self.surface.fill((255, 255, 255))

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
        self.surface.fill((255, 255, 255))

        font = pygame.font.Font(None, 36)
        text = font.render(str(self.state), 1, (10, 10, 10))
        fps_text = font.render(str(self.clock.get_fps()), 1, (10, 10, 10))
        fps_textpos = fps_text.get_rect()
        fps_textpos.centery += 32

        self.surface.blit(text, text.get_rect())
        self.surface.blit(fps_text, fps_textpos)
        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(FPS)

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
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        state, reward, done, info = game.step(None)
        game.render()

    game.close()
