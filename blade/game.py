import copy
import random

import numpy as np

from utils import (BLACKSMITH_POPULATION_INCREMENT, DEFAULT_MAX_POPULATION,
                   FULL_ACTIONS, FULL_MAX_POPULATION, GOLD_SPEED, INITIAL_GOLD,
                   KEEP_POPULATION_INCREMENT, LANE_LENGTH, SEED,
                   SIMPLE_ACTIONS, SIMPLE_TECHS, TRANSPORT_GOLD_SPEED,
                   UNIT_INDEX, UNITS, WINDMILL_GOLD_SPEED)


class Unit:
    def __init__(self, *args, **kwargs):
        for d in args:
            for k in d:
                setattr(self, k, d[k])

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.min_distance = 0.0
        self.distance = 0.0
        self.speed = 0.1

        self.direction = 0  # -1, 0, 1
        self.static = True

        self.cool_down = 0
        self.dead = False

        self.damage = 0
        self.heal = 0

        self.healable = False

    @property
    def fmt_direction(self):
        l = ['v', 'O', '^']

        return l[self.direction + 1]

    def __repr__(self):
        return f'<{self.name[0]} {self.fmt_direction} H:{self.health:.1f} D:{self.distance:.1f}>'

    def _in_range(self, fr):
        return LANE_LENGTH - fr - self.distance <= self.attack_range

    def ready(self, dist):
        return self.static and dist <= self.attack_range and self.cool_down == 0


class Footman(Unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_health = 280.0 + (40.0 if self.faction.techs['steel_blade']['built'] else 0.0)
        self.health = self.max_health

        self.min_distance = 5.0
        self.distance = self.min_distance

        self.speed = 1.6 + random.random() / 5

        self.attack_range = 5.0
        self.damage = 5
        # self.attack_animation = 2
        # self.attack_backswing = 2
        self.interval = 14

    def attack(self):
        return random.uniform(self.damage, 2 * self.damage)


class Rifleman(Unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_health = 180.0
        self.health = self.max_health

        self.distance = self.min_distance

        self.speed = 1.3 + random.random() / 5

        self.attack_range = 10.0
        self.damage = 9 + (3 if self.faction.techs['long_barrelled_gun']['built'] else 0)
        # self.attack_animation = 2
        # self.attack_backswing = 2
        self.interval = 14

    def attack(self):
        return random.uniform(self.damage, 2 * self.damage)


class Monk(Unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_health = 120.0
        self.health = self.max_health

        self.distance = self.min_distance

        self.speed = 1.3 + random.random() / 5

        self.attack_range = 30.0
        self.heal = 150  # 15
        self.damage = 70  # 7
        # self.attack_animation = 2
        # self.attack_backswing = 2
        self.interval = 40

        self.healable = True

    def attack(self):
        return random.uniform(self.damage, 2 * self.damage)


UNIT_TEMPLATE = {'footman': Footman, 'rifleman': Rifleman, 'monk': Monk}


class Watchtower:
    def __init__(self):
        self.damage = 18.0
        self.attack_range = LANE_LENGTH / 3
        self.interval = 20
        self.cool_down = 0

    def ready(self, dist):
        return dist <= self.attack_range and self.cool_down == 0

    def attack(self):
        return random.uniform(self.damage, self.damage + 12)


class Base:
    HEALTH = [1600.0, 2200.0, 4000.0]
    REPAIR_COST = 0.25
    REPAIR_SPEED = 0.4

    def __init__(self, tag):
        self.tag = tag

        self.repairing = False
        self.max_health = self.HEALTH[0]
        self.health = self.max_health

    @property
    def fmt_repairing(self):
        return '+' if self.repairing else '_'

    def __repr__(self):
        return f'<{self.tag} {self.fmt_repairing} H:{self.health:.1f}>'

    def set_repair(self, flag):
        self.repairing = flag and self.health < self.max_health

    def update_max_health(self, faction):
        # TODO update correct number
        # if faction.techs['keep']
        #     return self.HEALTH[1]
        # if faction.techs['castle']:
        #     return self.HEALTH[2]
        return self.HEALTH[0]

    def step_repair(self, faction):
        if faction.gold < self.REPAIR_COST:
            self.repairing = False
        elif self.repairing and self.health < self.max_health:
            self.health += self.REPAIR_SPEED

            if self.health >= self.max_health:
                self.health = self.max_health
                self.repairing = False


class Faction:
    def __init__(self, tag):
        self.tag = tag

        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(UNITS)

        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED

        self.base = Base(self.tag)
        self.watchtower = None

        self.max_population = DEFAULT_MAX_POPULATION
        self.population = 0
        self.army = []

        self.do_render = True
        self.render_state = {}

    def reset(self, debug):
        self.techs = copy.deepcopy(SIMPLE_TECHS)
        self.units = copy.deepcopy(UNITS)

        self.gold = INITIAL_GOLD
        self.gold_speed = GOLD_SPEED

        self.base = Base(self.tag)

        self.population = 0
        self.army = []

        if debug:
            self.gold = 9999

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
        fr, un = -1.0, None

        for unit in self.army:
            if not unit.dead and unit.distance > fr:
                fr = unit.distance
                un = unit

        # base
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
            # print(
            #     f'{self.tag} illegal action: {action} cannot been built'
            # )
            pass
        else:
            self.techs[action]['building'] = True
            self.gold -= self.techs[action]['gold_cost']

    def _handle_unit(self, action):
        if (
            self.units[action]['building']
            or self.units[action]['gold_cost'] > self.gold
            or not self._all_require_satisfied(self.units[action]['require'])
            or self.population >= self.max_population
        ):
            # print(
            #     f'{self.tag} illegal action: {action} cannot been built'
            # )
            pass
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

                    # update population and watchtower
                    if k == 'blacksmith':
                        self.max_population += BLACKSMITH_POPULATION_INCREMENT
                    elif k == 'keep':
                        self.max_population += KEEP_POPULATION_INCREMENT
                        self.base.max_health = Base.HEALTH[1]
                    elif k == 'watchtower':
                        self.watchtower = Watchtower()

    def _unit_count_down(self):
        for k, v in self.units.items():
            if v['count_down'] and v['building']:
                v['count_down'] -= 1

                if v['count_down'] == 0:
                    v['building'] = False
                    v['count_down'] = v['time_cost']
                    self.army.append(UNIT_TEMPLATE[k]({'faction': self}, v))

    def _gold_update(self):
        if self.techs['windmill']['built']:
            self.gold_speed = WINDMILL_GOLD_SPEED
        if self.techs['transport']['built']:
            self.gold_speed = TRANSPORT_GOLD_SPEED

        self.gold += self.gold_speed - self.base.repairing * Base.REPAIR_COST

    def _prepare_render_state(self):
        """
        for rendering ONLY
        """

        for k, v in self.techs.items():
            status = 'not'

            if v['gold_cost'] <= self.gold and self._all_require_satisfied(v['require']):
                status = 'can'

            if v['building']:
                status = 'building'

            if v['built']:
                status = 'built'

            self.render_state[k] = status

        for k, v in self.units.items():
            status = 'not'

            if v['gold_cost'] <= self.gold and self._all_require_satisfied(v['require']):
                status = 'can'

            if v['building']:
                status = 'building'

            self.render_state[k] = status

        self.render_state['gold'] = int(self.gold)
        self.render_state['base'] = int(self.base.health)
        self.render_state['repair'] = self.base.repairing
        self.render_state['frontier'] = '{:.1f} {}'.format(*self._frontier())
        self.render_state['population'] = '{}/{}'.format(self.population, self.max_population)

    def faction_step(self, action):
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
        self.base.step_repair(self)

        # === gold
        self._gold_update()

        # prepare state for render
        if self.do_render:
            self._prepare_render_state()


class Game:
    def __init__(self, simple=False, debug=False):
        self.left = Faction('Red')
        self.right = Faction('Blue')

        if simple:
            self.actions = SIMPLE_ACTIONS
        else:
            self.actions = FULL_ACTIONS

        self.timer = 0
        self.debug = debug

    def seed(self, seed=None):
        if seed is None:
            seed = SEED

        random.seed(SEED)
        np.random.seed(SEED)

    @property
    def available_actions(self):
        return tuple(action for action in self.actions)

    def reset(self):
        # reset two factions
        self.left.reset(debug=self.debug)
        self.right.reset(debug=False)

        self.timer = 0

        # prepare state
        state = generate_observation(self)

        return state

    def parse(self, left_action, right_action):
        left_action = self._parse_action(left_action)
        right_action = self._parse_action(right_action)
        return left_action, right_action

    def _global_action(self, left_action, right_action):
        for side, action in (('left', left_action), ('right', right_action)):
            if action in ['forward', 'backward']:
                if side == 'left':
                    for unit in self.left.army:
                        if action == 'forward':
                            unit.direction = 1
                        elif action == 'backward':
                            if unit.distance - unit.speed >= 0:
                                unit.direction = -1
                elif side == 'right':
                    for unit in self.right.army:
                        if action == 'forward':
                            unit.direction = 1
                        elif action == 'backward':
                            if unit.distance - unit.speed >= 0:
                                unit.direction = -1

            elif action in self.actions:
                if side == 'left':
                    self.left.faction_step(action)
                elif side == 'right':
                    self.right.faction_step(action)
            else:
                raise NotImplementedError(f'invalid {side} action {action}: unknown action')

    def _global_movement(self):
        for unit in self.left.army:
            if unit.direction == 1 and not unit._in_range(self.right._frontier()[0]):
                unit.distance += unit.speed
                unit.static = False
            elif unit.direction == -1 and unit.distance - unit.speed >= unit.min_distance:
                unit.distance -= unit.speed
                unit.static = False

                if unit.distance - unit.speed < 0:
                    unit.static = True
                    unit.direction = 0
            elif unit.direction == 1 and unit._in_range(self.right._frontier()[0]):
                # keep the direction forward but remain static
                unit.static = True

        for unit in self.right.army:
            if unit.direction == 1 and not unit._in_range(self.left._frontier()[0]):
                unit.distance += unit.speed
                unit.static = False
            elif unit.direction == -1 and unit.distance - unit.speed >= unit.min_distance:
                unit.distance -= unit.speed
                unit.static = False

                if unit.distance - unit.speed < 0:
                    unit.static = True
                    unit.direction = 0
            elif unit.direction == 1 and unit._in_range(self.left._frontier()[0]):
                # keep the direction forward but remain static
                unit.static = True

    def _global_health(self):
        end_status = None

        # damage and health calculation
        for unit in self.left.army:
            # heal logic
            if unit.healable:
                healed = False
                # iterate all units in own army (still at battle front line)
                for un in sorted(self.left.army, key=lambda x: x.distance, reverse=True):
                    if un is not unit:  # does not heal self
                        target_distance = un.distance - unit.distance  # not base

                        if unit.ready(target_distance) and un.health < (un.max_health - unit.heal):
                            un.health = min(un.max_health, un.health + unit.heal)

                            # heal also cool down
                            unit.cool_down = (unit.cool_down + 1) % unit.interval
                            healed = True
                            break

                if healed:
                    # heal then do not attack
                    continue

            # attack logic
            fr, un = self.right._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if unit.ready(target_distance):
                un.health -= unit.attack()

                if un.health <= 0:
                    if not isinstance(un, Base):
                        un.dead = True
                        self.right.population -= 1
                    else:
                        end_status = 'left'

                        return end_status

                unit.cool_down = (unit.cool_down + 1) % unit.interval
            else:
                # TODO hit and run
                # target_distance <= unit.attack_range and unit.cool_down > 0
                unit.cool_down = (unit.cool_down + 1) % unit.interval

        # left watchtower
        if self.left.watchtower is not None:
            left_watchtower = self.left.watchtower
            fr, un = self.right._frontier()
            target_distance = LANE_LENGTH - fr

            if left_watchtower.ready(target_distance):
                un.health -= left_watchtower.attack()

                if un.health <= 0:
                    # assuleft not base
                    un.dead = True
                    self.right.population -= 1

                left_watchtower.cool_down = (left_watchtower.cool_down + 1) % left_watchtower.interval

        for unit in self.right.army:
            # heal logic
            if unit.healable:
                healed = False
                # iterate all units in own army (still at battle front line)
                for un in sorted(self.right.army, key=lambda x: x.distance, reverse=True):
                    if un is not unit:  # does not heal self
                        target_distance = un.distance - unit.distance  # not base

                        if unit.ready(target_distance) and un.health < (un.max_health - unit.heal):
                            un.health = min(un.max_health, un.health + unit.heal)

                            # heal also cool down
                            unit.cool_down = (unit.cool_down + 1) % unit.interval
                            healed = True
                            break

                if healed:
                    # heal then do not attack
                    continue

            # attack logic
            fr, un = self.left._frontier()
            target_distance = LANE_LENGTH - fr - unit.distance

            if unit.ready(target_distance):
                un.health -= unit.attack()

                if un.health <= 0:
                    if not isinstance(un, Base):
                        un.dead = True
                        self.left.population -= 1
                    else:
                        end_status = 'right'

                        return end_status

                unit.cool_down = (unit.cool_down + 1) % unit.interval
            else:
                # TODO hit and run
                # target_distance <= unit.attack_range and unit.cool_down > 0
                unit.cool_down = (unit.cool_down + 1) % unit.interval

        # right watchtower
        if self.right.watchtower is not None:
            right_watchtower = self.right.watchtower
            fr, un = self.left._frontier()
            target_distance = LANE_LENGTH - fr

            if right_watchtower.ready(target_distance):
                un.health -= right_watchtower.attack()

                if un.health <= 0:
                    # assuleft not base
                    un.dead = True
                    self.left.population -= 1

                right_watchtower.cool_down = (right_watchtower.cool_down + 1) % right_watchtower.interval

        # remove dead units
        for unit in self.left.army:
            if unit.dead:
                self.left.army.remove(unit)

        for unit in self.right.army:
            if unit.dead:
                self.right.army.remove(unit)

        return end_status

    def _parse_action(self, action):
        if isinstance(action, str):
            return action
        elif isinstance(action, (int, np.integer)):
            return self.actions[action]
        else:
            raise ValueError(f'unknown action type "{action}": {type(action)}')

    def step(self, left_action, right_action):
        left_action, right_action = self.parse(left_action, right_action)

        self.timer += 1
        reward = (0, 0)  # 0 for most cases because not scoleft at the moment
        done = False
        info = {}

        # global action
        self._global_action(left_action, right_action)

        # global movement
        self._global_movement()

        # global health
        end_status = self._global_health()

        if end_status is not None:
            done = True

            info['length'] = self.timer

            if end_status == 'left':
                reward = (1, -1)
            elif end_status == 'right':
                reward = (-1, 1)

        # prepare state
        state = generate_observation(self)

        return state, reward, done, info


def generate_observation(game):
    state = {'left': [], 'right': []}

    # 1. faction-wise misc
    state['left'].extend(
        [
            game.timer,
            # LANE_LENGTH,
            game.left.gold,
            # game.left.gold_speed,
            game.left.watchtower is not None,
            (0 if game.left.watchtower is None else game.left.watchtower.damage),
            (0 if game.left.watchtower is None else game.left.watchtower.attack_range / LANE_LENGTH),
            (0 if game.left.watchtower is None else game.left.watchtower.interval),
            (0 if game.left.watchtower is None else game.left.watchtower.cool_down),
        ]
    )
    state['right'].extend(
        [
            game.timer,
            # LANE_LENGTH,
            game.right.gold,
            # game.right.gold_speed,
            game.right.watchtower is not None,
            (0 if game.right.watchtower is None else game.right.watchtower.damage),
            (0 if game.right.watchtower is None else game.right.watchtower.attack_range / LANE_LENGTH),
            (0 if game.right.watchtower is None else game.right.watchtower.interval),
            (0 if game.right.watchtower is None else game.right.watchtower.cool_down),
        ]
    )

    # 2. techs
    for k, v in game.left.techs.items():
        state['left'].extend(
            [v['time_cost'], v['count_down'], v['gold_cost'], v['built'], v['building'],]
        )

    for k, v in game.right.techs.items():
        state['right'].extend(
            [v['time_cost'], v['count_down'], v['gold_cost'], v['built'], v['building'],]
        )

    # 3. unit builds
    for k, v in game.left.units.items():
        state['left'].extend(
            [UNIT_INDEX[v['name']], v['gold_cost'], v['time_cost'], v['count_down'], v['building'],]
        )

    for k, v in game.right.units.items():
        state['right'].extend(
            [UNIT_INDEX[v['name']], v['gold_cost'], v['time_cost'], v['count_down'], v['building'],]
        )

    # 4. army (population, attack, health determined by tech)
    state_left_army = []
    state_right_army = []

    for i in range(FULL_MAX_POPULATION):
        if i < len(game.left.army):
            un = game.left.army[i]
            state_left_army.extend(
                [
                    UNIT_INDEX[un.name],
                    un.distance / LANE_LENGTH,
                    un.direction,
                    un.cool_down,
                    un.interval,
                    un.health,
                    un.max_health,
                    un.damage,
                    un.heal,
                ]
            )
        else:
            state_left_army.extend([0] * 9)

        if i < len(game.right.army):
            un = game.right.army[i]
            state_right_army.extend(
                [
                    UNIT_INDEX[un.name],
                    un.distance / LANE_LENGTH,
                    un.direction,
                    un.cool_down,
                    un.interval,
                    un.health,
                    un.max_health,
                    un.damage,
                    un.heal,
                ]
            )
        else:
            state_right_army.extend([0] * 9)

    state['left'].extend(state_left_army)
    state['left'].extend(state_right_army)

    state['right'].extend(state_right_army)
    state['right'].extend(state_left_army)

    # 4. base (health determined by tech)
    state['left'].extend(
        [
            game.left.base.REPAIR_COST,
            game.left.base.REPAIR_SPEED / game.left.base.max_health,
            game.left.base.repairing,
            game.left.base.health / game.left.base.max_health,
            # game.left.base.max_health,
            game.right.base.repairing,
            game.right.base.health / game.right.base.max_health,
            # game.right.base.max_health,
        ]
    )
    state['right'].extend(
        [
            game.right.base.REPAIR_COST,
            game.right.base.REPAIR_SPEED / game.right.base.max_health,
            game.right.base.repairing,
            game.right.base.health / game.right.base.max_health,
            # game.right.base.max_health,
            game.left.base.repairing,
            game.left.base.health / game.left.base.max_health,
            # game.left.base.max_health,
        ]
    )
    for k, v in game.right.techs.items():
        if v['visible']:
            state['left'].append(v['built'])

    for k, v in game.left.techs.items():
        if v['visible']:
            state['right'].append(v['built'])

    # format
    state['left'] = np.array(state['left']).astype(np.float).ravel()
    state['right'] = np.array(state['right']).astype(np.float).ravel()

    return state
