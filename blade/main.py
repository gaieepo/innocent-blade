import copy
import random
import sys
from itertools import count

import numpy as np
import pygame

from utils import (ACTIONS, FPS, GOLD_SPEED, HEIGHT, INITIAL_GOLD, LANE_LENGTH,
                   MAX_POPULATION, SIMPLE_TECHS, SIMPLE_UNITS, UNIT_TEMPLATE,
                   WIDTH, WINDMILL_GOLD_SPEED, Base, Footman, Rifleman)


class Faction:
    def __init__(self, side):
        self.side = side

        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(SIMPLE_UNITS)

        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED

        self.base = Base(self)

        self.population = 0
        self.army = []

        self.render_state = {}

    def reset(self):
        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(SIMPLE_UNITS)

        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED

        self.base = Base(self)

        self.population = 0
        self.army = []

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

    def _handle_tech(self, action):
        if (
            self.techs[action]['building']
            or self.techs[action]['built']
            or self.techs[action]['gold_cost'] > self.gold
            or not self._all_require_satisfied(self.techs[action]['require'])
        ):
            print(
                f'{self.side} illegal action {action}: tech cannot been built'
            )
        else:
            self.techs[action]['building'] = True
            self.gold -= self.techs[action]['gold_cost']

    def _handle_unit(self, action):
        if (
            self.units[action]['building']
            or self.units[action]['gold_cost'] > self.gold
            or not self._all_require_satisfied(self.units[action]['require'])
            or self.population >= MAX_POPULATION
        ):
            print(
                f'{self.side} illegal action {action}: unit cannot been built'
            )
        else:
            self.population += 1
            self.gold -= self.units[action]['gold_cost']
            self.units[action]['building'] = True

    def _active_update(self, action):
        if action in self.techs.keys():
            self._handle_tech(action)
        elif action in self.units.keys():
            self._handle_unit(action)
        elif action in ['repair', 'stop_repair']:
            self.base.set_repair(action == 'repair')

    def _tech_count_down(self):
        for k, v in self.techs.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['built'] = True
                    v['building'] = False

    def _unit_count_down(self):
        for k, v in self.units.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['building'] = False
                    v['count_down'] = v['time_cost']
                    self.army.append(UNIT_TEMPLATE[k](v))

    def _gold_update(self):
        if self.techs['windmill']['built']:
            self.gold_speed = WINDMILL_GOLD_SPEED

        self.gold += self.gold_speed - self.base.repairing * Base.REPAIR_COST

    def _prepare_render_state(self):
        """
        for rendering ONLY
        """

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

            self.render_state[k] = status

        for k, v in self.units.items():
            status = 'not'

            if v['gold_cost'] <= self.gold and self._all_require_satisfied(
                v['require']
            ):
                status = 'can'

            if v['building']:
                status = 'building'

            self.render_state[k] = status

        self.render_state['gold'] = self.gold
        self.render_state['base'] = self.base.health
        self.render_state['repair'] = self.base.repairing
        self.render_state['frontier'] = '{:.1f} {}'.format(*self._frontier())
        self.render_state['population'] = self.population

    def step(self, action):
        # - active update
        self._active_update(action)

        # - passive update
        # === count down for techs
        self._tech_count_down()

        # === count down for units
        self._unit_count_down()

        # === units movement
        # === units health
        # delegate by game step

        # === base health
        self.base.update_max_health(self)
        self.base.step_repair()

        # === gold
        self._gold_update()

        # prepare state for render
        self._prepare_render_state()


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

    @property
    def all_actions(self):
        return tuple(ACTIONS)

    @property
    def available_actions(self):
        actions = [action for action in ACTIONS if action != 'close']

        return tuple(actions)

    @property
    def n_actions(self):
        return len(ACTIONS)

    def _observation(self):
        # TODO faction is not returning inner state
        # need to process aggregated state externally
        # items need to return as state:
        state = {'white': [], 'black': []}

        # 0. some globals
        state['white'].append(LANE_LENGTH)
        state['black'].append(LANE_LENGTH)

        # 1. current gold (gold speed determined by tech)
        state['white'].extend([self.white.gold, self.white.gold_speed])
        state['black'].extend([self.black.gold, self.black.gold_speed])

        # 2. techs

        for k, v in self.white.techs.items():
            state['white'].extend(
                [
                    v['time_cost'],
                    v['count_down'],
                    v['gold_cost'],
                    v['built'],
                    v['building'],
                ]
            )

        for k, v in self.black.techs.items():
            state['black'].extend(
                [
                    v['time_cost'],
                    v['count_down'],
                    v['gold_cost'],
                    v['built'],
                    v['building'],
                ]
            )

        # 3. army (population, attack, health determined by tech)

        for i in range(MAX_POPULATION):
            if i < len(self.white.army):
                state['white'].extend(
                    [
                        self.white.army[i].gold_cost,
                        self.white.army[i].time_cost,
                        self.white.army[i].count_down,
                        self.white.army[i].building,
                        self.white.army[i].distance,
                        self.white.army[i].speed,
                        self.white.army[i].direction,
                        self.white.army[i].cool_down,
                        self.white.army[i].max_health,
                        self.white.army[i].health,
                        self.white.army[i].attack_range,
                        *self.white.army[i].damage,
                        self.white.army[i].interval,
                    ]
                )
            else:
                state['white'].extend([0] * 13)

            if i < len(self.black.army):
                state['black'].extend(
                    [
                        self.black.army[i].gold_cost,
                        self.black.army[i].time_cost,
                        self.black.army[i].count_down,
                        self.black.army[i].building,
                        self.black.army[i].distance,
                        self.black.army[i].speed,
                        self.black.army[i].direction,
                        self.black.army[i].cool_down,
                        self.black.army[i].max_health,
                        self.black.army[i].health,
                        self.black.army[i].attack_range,
                        *self.black.army[i].damage,
                        self.black.army[i].interval,
                    ]
                )
            else:
                state['black'].extend([0] * 13)

        # 4. base (health determined by tech)
        state['white'].extend(
            [
                self.white.base.repairing,
                self.white.base.max_health,
                self.white.base.health,
            ]
        )
        state['black'].extend(
            [
                self.black.base.repairing,
                self.black.base.max_health,
                self.black.base.health,
            ]
        )

        # format
        state['white'] = np.array(state['white']).astype(np.float32)
        state['black'] = np.array(state['black']).astype(np.float32)

        return state

    def reset(self):
        # reset two factions
        self.white.reset()
        self.black.reset()

        # prepare state
        state = self._observation()

        return state

    def render(
        self, mode='human', close=False, white_action=None, black_action=None
    ):
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

        # render state text
        # (optional) render action

        if white_action is not None and black_action is not None:
            self.white.render_state['action'] = white_action
            self.black.render_state['action'] = black_action

        white_offset = 50

        for k, v in self.white.render_state.items():
            if k != 'army':
                k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.top = white_offset
                self.surface.blit(k_text, k_textpos)
                white_offset += 30

        black_offset = 50

        for k, v in self.black.render_state.items():
            if k != 'army':
                k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.right = WIDTH
                k_textpos.top = black_offset
                self.surface.blit(k_text, k_textpos)
                black_offset += 30

        # render comic army

        for unit in self.white.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 0),
                    pygame.Rect(
                        (
                            50.0
                            + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 30 * (unit.health / unit.max_health),
                        ),
                        (20, 30 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Rifleman):
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 255),
                    pygame.Rect(
                        (
                            50.0
                            + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )

        for unit in self.black.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (255, 0, 0),
                    pygame.Rect(
                        (
                            WIDTH
                            - 50.0
                            - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 30 * (unit.health / unit.max_health),
                        ),
                        (20, 30 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Rifleman):
                pygame.draw.rect(
                    self.surface,
                    (255, 255, 0),
                    pygame.Rect(
                        (
                            WIDTH
                            - 50.0
                            - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )

        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(FPS)

    def _global_action(self, white_action, black_action):
        for side, action in (('white', white_action), ('black', black_action)):
            if action in ['forward', 'backward']:
                if side == 'white':
                    for unit in self.white.army:
                        if action == 'forward':
                            unit.direction = 1
                        elif action == 'backward':
                            if unit.distance - unit.speed >= 0:
                                unit.direction = -1
                elif side == 'black':
                    for unit in self.black.army:
                        if action == 'forward':
                            unit.direction = 1
                        elif action == 'backward':
                            if unit.distance - unit.speed >= 0:
                                unit.direction = -1

            elif action in ACTIONS:
                if side == 'white':
                    self.white.step(action)
                elif side == 'black':
                    self.black.step(action)
            else:
                print(f'invalid {side} action {action}: unknown action')

    def _global_movement(self):
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

    def _global_health(self):
        done = False

        for unit in self.white.army:
            fr, un = self.black._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if unit.ready(target_distance):
                un.health -= unit.attack()

                if un.health <= 0:
                    if not isinstance(un, Base):
                        self.black.army.remove(un)
                        self.black.population -= 1
                    else:
                        # TODO process done event
                        done = True

                        return done

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
                        self.white.population -= 1
                    else:
                        # TODO process done event
                        done = True

                        return done

                unit.cool_down = (unit.cool_down + 1) % unit.interval
            elif target_distance <= unit.attack_range and unit.cool_down > 0:
                unit.cool_down = (unit.cool_down + 1) % unit.interval
                # TODO hit and run

    def step(self, white_action, black_action):
        # gym support
        reward = 0  # 0 for most cases because not scored at the moment
        done = False

        # global action
        self._global_action(white_action, black_action)

        # global movement
        self._global_movement()

        # global health
        done = self._global_health()

        # prepare state
        state = self._observation()

        return state, reward, done, {}

    def close(self):
        pygame.quit()
        sys.exit()


###################################################
# general agents
###################################################
def human_agent():
    action = 'null'

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                action = 'close'
            elif event.key == pygame.K_1:
                action = 'barrack'
            elif event.key == pygame.K_2:
                action = 'blacksmith'
            elif event.key == pygame.K_3:
                action = 'windmill'
            elif event.key == pygame.K_4:
                action = 'footman'
            elif event.key == pygame.K_5:
                action = 'rifleman'
            elif event.key == pygame.K_SPACE:
                action = 'forward'
            elif event.key == pygame.K_BACKSPACE:
                action = 'backward'
            elif event.key == pygame.K_r:
                action = 'repair'
            elif event.key == pygame.K_t:
                action = 'stop_repair'

    return action


def random_agent(actions):
    return random.choice(actions)


###################################################
# PG related
###################################################
def sigmoid(x):
    """ sigmoid 'squash' to interval [0, 1] """

    return 1.0 / (1.0 + np.exp(-x))


def numpy_agent(state, actions):
    h = np.dot(model['W1'], state)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)

    return actions[np.argmax(sigmoid(logp))]


if __name__ == "__main__":
    game = Game()
    state = game.reset()

    # naive numpy agent
    H = 200  # number of hidden layer neurons
    D = 112  # input dimensionality (# of grid)
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # xavier
    model['W2'] = np.random.randn(len(game.available_actions), H) / np.sqrt(H)

    for c in count():
        white_action = human_agent()
        black_action = numpy_agent(state['black'], game.available_actions)

        print(c, white_action, black_action)

        if white_action == 'close':
            game.close()

        state, reward, done, info = game.step(white_action, black_action)

        if done:
            break

        game.render(white_action=white_action, black_action=black_action)

    game.close()
