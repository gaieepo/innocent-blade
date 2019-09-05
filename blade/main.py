import copy
import sys

import pygame

from utils import (ACTIONS, FPS, GOLD_SPEED, HEIGHT, INITIAL_GOLD, LANE_LENGTH,
                   REPAIR_COST, SIMPLE_TECHS, SIMPLE_UNITS, TECHS, WIDTH,
                   WINDMILL_GOLD_SPEED, Unit)


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

    def _all_require_satisfied(self, reqs):
        rv = True

        for req in reqs:
            if not self.techs[req]['built']:
                rv = False

        return rv

    def _frontier(self):
        """
        the frontier

        return:
            fr - longest distance
            un - frontier unit
        """
        fr, un = 0.0, None

        for unit in self.army:
            if unit.distance > fr:
                fr = unit.distance
                un = unit

        return fr, un

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
                or not self._all_require_satisfied(
                    self.techs[action]['require']
                )
            ):
                print(f'illegal action {action}: tech cannot been built')
            else:
                self.techs[action]['building'] = True
                self.gold -= self.techs[action]['gold_cost']
        elif action in self.units.keys():
            print(self.side, 'receives', action)

            if (
                self.units[action]['building']
                or self.units[action]['gold_cost'] > self.gold
                or not self._all_require_satisfied(
                    self.units[action]['require']
                )
            ):
                print(f'illegal action {action}: unit cannot been built')
            else:
                self.units[action]['building'] = True
                self.gold -= self.units[action]['gold_cost']

        # - passive update
        # 1. gold

        if self.techs['windmill']:
            self.gold_speed = WINDMILL_GOLD_SPEED

        self.gold += self.gold_speed - self.repairing * REPAIR_COST

        # 2. count down for techs

        for k, v in self.techs.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['built'] = True
                    v['building'] = False

        # 3. count down for units

        for k, v in self.units.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['building'] = False
                    v['count_down'] = v['time_cost']
                    self.army.append(UNIT_TEMPLATE[k](v))

        # # 4. units movement
        # delegate by game step

        # # 5. units health

        # prepare state

        for k, v in self.techs.items():
            status = 'not'

            if v['gold_cost'] <= self.gold and self._all_require_satisfied(
                v['require']
            ):
                status = 'can'

            if v['building']:
                status = 'building'

            if v['built']:
                status = 'built'

            self.state[k] = status

        for k, v in self.units.items():
            status = 'not'

            if v['gold_cost'] <= self.gold and self._all_require_satisfied(
                v['require']
            ):
                status = 'can'

            if v['building']:
                status = 'building'

            self.state[k] = status

        self.state['gold'] = self.gold
        self.state['army'] = self.army

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
        white_text = font.render('White', 1, (10, 10, 10))
        self.surface.blit(white_text, white_text.get_rect())

        black_text = font.render('Black', 1, (10, 10, 10))
        black_textpos = black_text.get_rect()
        black_textpos.right = WIDTH
        self.surface.blit(black_text, black_textpos)

        fps_text = font.render(str(self.clock.get_fps()), 1, (10, 10, 10))
        fps_textpos = fps_text.get_rect()
        fps_textpos.midtop = (WIDTH / 2, 0)
        self.surface.blit(fps_text, fps_textpos)

        white_offset = 50

        for k, v in self.state['white'].items():
            k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
            k_textpos = k_text.get_rect()
            k_textpos.top = white_offset
            self.surface.blit(k_text, k_textpos)
            white_offset += 30

        black_offset = 50

        for k, v in self.state['black'].items():
            k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
            k_textpos = k_text.get_rect()
            k_textpos.right = WIDTH
            k_textpos.top = black_offset
            self.surface.blit(k_text, k_textpos)
            black_offset += 30

        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(FPS)

    def step(self, action):
        if action in ['forward', 'backward']:
            for unit in self.white.army:
                if action == 'forward':
                    unit.movement = 1
                elif action == 'backward':
                    if unit.distance - unit.speed >= 0:
                        unit.movement = -1

            for unit in self.black.army:
                if action == 'forward':
                    unit.movement = 1
                elif action == 'backward':
                    if unit.distance - unit.speed >= 0:
                        unit.movement = -1
        elif action in ACTIONS:
            # TODO conditional action on each side
            self.state['white'] = self.white.step(action)
            self.state['black'] = self.black.step('null')
        else:
            print(f'invalid action {action}: unknown action')

        # global health

        for unit in self.white.army:
            fr, un = self.black._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if (
                target_distance < unit.attack_range
                and unit.count_down == unit.interval
            ):
                un.health -= unit.attack()

                if un.health <= 0:
                    self.black.army.remove(un)

                unit.count_down -= 1

                if unit.count_down == 0:
                    unit.count_down = unit.interval

        # global movement

        for unit in self.white.army:
            if unit.movement == 1 and not unit._in_range(
                self.black._frontier()
            ):
                unit.distance += unit.speed
            elif unit.movement == -1 and unit.distance - unit.speed >= 0:
                unit.distance -= unit.speed

                if unit.distance - unit.speed < 0:
                    unit.movement = 0

        for unit in self.black.army:
            if unit.movement == 1 and not unit._in_range(
                self.white._frontier()
            ):
                unit.distance += unit.speed
            elif unit.movement == -1 and unit.distance - unit.speed >= 0:
                unit.distance -= unit.speed

                if unit.distance - unit.speed < 0:
                    unit.movement = 0

        # gym support
        reward = 0
        done = False

        return self.state, reward, done, {}

    def close(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    state = game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game.close()
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
                elif event.key == pygame.K_SPACE:
                    game.step('forward')
                elif event.key == pygame.K_BACKSPACE:
                    game.step('backward')
        else:
            game.step('null')
        game.render()

    game.close()
