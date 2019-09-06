import copy
import sys

import pygame

from utils import (ACTIONS, FPS, GOLD_SPEED, HEIGHT, INITIAL_GOLD, LANE_LENGTH,
                   SIMPLE_TECHS, SIMPLE_UNITS, UNIT_TEMPLATE, WIDTH,
                   WINDMILL_GOLD_SPEED, Base, Footman, Rifleman)


class Faction:
    def __init__(self, side='white'):
        self.side = side

        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(SIMPLE_UNITS)

        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED

        self.base = Base(self)

        self.state = {}
        self.army = []

    def reset(self):
        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(SIMPLE_UNITS)
        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED
        self.base = Base(self)
        self.army = []
        self.state = {}
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

        if un is None:
            fr = 0.0
            un = self.base

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
        elif action in ['repair', 'stop_repair']:
            self.base.set_repair(action == 'repair')

        # - passive update
        # === count down for techs

        for k, v in self.techs.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['built'] = True
                    v['building'] = False

        # === count down for units

        for k, v in self.units.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['building'] = False
                    v['count_down'] = v['time_cost']
                    self.army.append(UNIT_TEMPLATE[k](v))

        # === units movement
        # === units health
        # delegate by game step

        # === base health
        self.base.update_max_health(self)
        self.base.step_repair()

        # === gold

        if self.techs['windmill']['built']:
            self.gold_speed = WINDMILL_GOLD_SPEED

        self.gold += self.gold_speed - self.base.repairing * Base.REPAIR_COST

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
        self.state['base'] = self.base.health
        self.state['repair'] = self.base.repairing
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

        # TODO dummy black
        dummy_unit1 = Footman(
            {
                'name': 'Footman',
                'require': ['barrack'],
                'gold_cost': 70,
                'time_cost': 15,
                'count_down': 15,
                'building': False,
            }
        )
        dummy_unit1.direction = 1
        dummy_unit1.static = False
        dummy_unit2 = Rifleman(
            {
                'name': 'Rifleman',
                'require': ['barrack', 'blacksmith'],
                'gold_cost': 90,
                'time_cost': 15,
                'count_down': 15,
                'building': False,
            }
        )
        dummy_unit2.direction = 1
        dummy_unit2.static = False

        self.black.army.append(dummy_unit1)
        self.black.army.append(dummy_unit2)

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
            if k != 'army':
                k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.top = white_offset
                self.surface.blit(k_text, k_textpos)
                white_offset += 30

        black_offset = 50

        for k, v in self.state['black'].items():
            if k != 'army':
                k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.right = WIDTH
                k_textpos.top = black_offset
                self.surface.blit(k_text, k_textpos)
                black_offset += 30

        # render army

        for unit in self.white.state['army']:
            pygame.draw.rect(
                self.surface,
                (0, 255, 0),
                pygame.Rect(
                    (WIDTH * (unit.distance / LANE_LENGTH), HEIGHT - 50),
                    (20, 50 * (unit.health / unit.max_health)),
                ),
            )

        for unit in self.black.state['army']:
            pygame.draw.rect(
                self.surface,
                (255, 0, 0),
                pygame.Rect(
                    (
                        WIDTH - WIDTH * (unit.distance / LANE_LENGTH),
                        HEIGHT - 50,
                    ),
                    (20, 50 * (unit.health / unit.max_health)),
                ),
            )

        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(FPS)

    def step(self, action):
        # gym support
        reward = 0
        done = False

        # global action (white only)

        if action in ['forward', 'backward']:
            for unit in self.white.army:
                if action == 'forward':
                    unit.direction = 1
                elif action == 'backward':
                    if unit.distance - unit.speed >= 0:
                        unit.direction = -1

        elif action in ACTIONS:
            # TODO conditional action on each side
            self.state['white'] = self.white.step(action)
            self.state['black'] = self.black.step('null')
        else:
            print(f'invalid action {action}: unknown action')

        # global movement

        for unit in self.white.army:
            if unit.direction == 1 and not unit._in_range(
                self.black._frontier()[0]
            ):
                unit.distance += unit.speed
                unit.static = False
            elif unit.direction == -1 and unit.distance - unit.speed >= 0:
                unit.distance -= unit.speed
                unit.static = False

                if unit.distance - unit.speed < 0:
                    unit.static = True
                    unit.direction = 0
            elif unit.direction == 1 and unit._in_range(
                self.black._frontier()[0]
            ):
                # keep the direction forward but remain static
                unit.static = True

        for unit in self.black.army:
            if unit.direction == 1 and not unit._in_range(
                self.white._frontier()[0]
            ):
                unit.distance += unit.speed
                unit.static = False
            elif unit.direction == -1 and unit.distance - unit.speed >= 0:
                unit.distance -= unit.speed
                unit.static = False

                if unit.distance - unit.speed < 0:
                    unit.static = True
                    unit.direction = 0
            elif unit.direction == 1 and unit._in_range(
                self.black._frontier()[0]
            ):
                # keep the direction forward but remain static
                unit.static = True

        # global health

        for unit in self.white.army:
            fr, un = self.black._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if unit.ready(target_distance):
                un.health -= unit.attack()

                if un.health <= 0:
                    if not isinstance(un, Base):
                        self.black.army.remove(un)
                    else:
                        # TODO process done event
                        done = True

                        break

                unit.cool_down = (unit.cool_down + 1) % unit.interval
            elif target_distance <= unit.attack_range and unit.cool_down > 0:
                unit.cool_down = (unit.cool_down + 1) % unit.interval
                # TODO hit and run

        for unit in self.black.army:
            fr, un = self.white._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if unit.ready(target_distance):
                un.health -= unit.attack()

                if un.health <= 0:
                    if not isinstance(un, Base):
                        self.white.army.remove(un)
                    else:
                        # TODO process done event
                        done = True

                        break

                unit.cool_down = (unit.cool_down + 1) % unit.interval
            elif target_distance <= unit.attack_range and unit.cool_down > 0:
                unit.cool_down = (unit.cool_down + 1) % unit.interval
                # TODO hit and run

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
                    state, reward, done, info = game.step('barrack')
                elif event.key == pygame.K_2:
                    state, reward, done, info = game.step('blacksmith')
                elif event.key == pygame.K_3:
                    state, reward, done, info = game.step('windmill')
                elif event.key == pygame.K_4:
                    state, reward, done, info = game.step('footman')
                elif event.key == pygame.K_5:
                    state, reward, done, info = game.step('rifleman')
                elif event.key == pygame.K_SPACE:
                    state, reward, done, info = game.step('forward')
                elif event.key == pygame.K_BACKSPACE:
                    state, reward, done, info = game.step('backward')
                elif event.key == pygame.K_r:
                    state, reward, done, info = game.step('repair')
                elif event.key == pygame.K_t:
                    state, reward, done, info = game.step('stop_repair')
        else:
            state, reward, done, info = game.step('null')

        if done:
            break

        game.render()

    game.close()
