import random

HEIGHT = 720
WIDTH = 960
FPS = 50

INITIAL_GOLD = 90.0
GOLD_SPEED = 1.0
WINDMILL_GOLD_SPEED = 2.0

LANE_LENGTH = 10.0
MAX_POPULATION = 7

FULL_ACTIONS = [
    'null',
    'close',
    'barrack',
    'blacksmith',
    'windmill',
    'footman',
    'rifleman',
    'forward',
    'backward',
    'repair',
    'stop_repair',
]

SIMPLE_ACTIONS = [
    'null',
    'close',
    'barrack',
    'blacksmith',
    'footman',
    'rifleman',
    'forward',
]

ACTIONS = SIMPLE_ACTIONS

SIMPLE_TECHS = {
    'barrack': {
        'name': 'Barrack',
        'require': [],
        'time_cost': 40,
        'count_down': 40,
        'gold_cost': 120,
        'built': False,
        'building': False,
    },
    'blacksmith': {
        'name': 'Blacksmith',
        'require': [],
        'time_cost': 50,
        'count_down': 50,
        'gold_cost': 220,
        'built': False,
        'building': False,
    },
    'windmill': {
        'name': 'Windmill',
        'require': [],
        'time_cost': 30,
        'count_down': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
}

TECHS = {
    'barrack': {
        'name': 'Barrack',
        'require': [],
        'time_cost': 30,
        'gold_cost': 120,
        'built': False,
        'building': False,
    },
    'keep': {
        'name': 'Keep',
        'require': [],
        'time_cost': 30,
        'gold_cost': 500,
        'built': False,
        'building': False,
    },
    'blacksmith': {
        'name': 'Blacksmith',
        'require': [],
        'time_cost': 30,
        'gold_cost': 220,
        'built': False,
        'building': False,
    },
    'windmill': {
        'name': 'Windmill',
        'require': [],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'watertower': {
        'name': 'Watertower',
        'require': ['blacksmith'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'steel_blade': {
        'name': 'Steel Blade',
        'require': ['blacksmith'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'long_barrelled_gun': {
        'name': 'Long-barrelled Gun',
        'require': ['blacksmith'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'iron_horse': {
        'name': 'Iron Horse',
        'require': ['blacksmith', 'castle'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'chancel': {
        'name': 'Chancel',
        'require': ['keep'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'monatery': {
        'name': 'Monatery',
        'require': ['keep'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'workshop': {
        'name': 'Workshop',
        'require': ['keep'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'castle': {
        'name': 'Castle',
        'require': ['keep'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'telescope': {
        'name': 'Telescope',
        'require': ['keep', 'watertower'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'transport': {
        'name': 'Transport',
        'require': ['keep'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'natural_power': {
        'name': 'Natural Power',
        'require': ['chancel'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'iron_gate': {
        'name': 'Iron Gate',
        'require': ['workshop'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'alchemy': {
        'name': 'Alchemy',
        'require': ['chancel'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'noble_metal': {
        'name': 'Noble Metal',
        'require': ['monatery', 'castle'],
        'time_cost': 30,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
}

SIMPLE_UNITS = {
    'footman': {
        'name': 'Footman',
        'require': ['barrack'],
        'gold_cost': 70,
        'time_cost': 15,
        'count_down': 15,
        'building': False,
    },
    'rifleman': {
        'name': 'Rifleman',
        'require': ['barrack', 'blacksmith'],
        'gold_cost': 90,
        'time_cost': 15,
        'count_down': 15,
        'building': False,
    },
}


class Base:
    HEALTH = [1600.0, 2200.0, 2800.0]
    REPAIR_COST = 0.5
    REPAIR_SPEED = 2.0

    def __init__(self, faction):
        self.faction = faction
        self.distance = 0.0

        self.repairing = False
        self.max_health = self.__class__.HEALTH[0]
        self.health = self.max_health

    @property
    def fmt_repairing(self):
        return '+' if self.repairing else '_'

    def __repr__(self):
        return f'<{self.faction.side} H:{self.health:.1f}>'

    def set_repair(self, flag):
        if flag:
            if self.health < self.max_health:
                self.repairing = True
        else:
            self.repairing = False

    def update_max_health(self, faction):
        # TODO update correct number
        # if faction.techs['keep']
        #     return self.__class__.HEALTH[1]
        # if faction.techs['castle']:
        #     return self.__class__.HEALTH[2]

        return self.__class__.HEALTH[0]

    def step_repair(self):
        if self.faction.gold < self.__class__.REPAIR_COST:
            self.repairing = False
        elif self.repairing and self.health < self.max_health:
            self.health += self.__class__.REPAIR_SPEED

            if self.health >= self.max_health:
                self.health = self.max_health
                self.repairing = False


class Unit:
    def __init__(self, *args, **kwargs):
        for d in args:
            for k in d:
                setattr(self, k, d[k])

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.distance = 0.1
        self.speed = 0.1

        self.direction = 0  # -1, 0, 1
        self.static = True

        self.cool_down = 0

    @property
    def fmt_direction(self):
        l = ['v', 'O', '^']

        return l[self.direction + 1]

    def __repr__(self):
        return f'<{self.name[0]} {self.fmt_direction} H:{self.health:.1f} D:{self.distance:.1f}>'

    def _in_range(self, fr):
        return LANE_LENGTH - fr - self.distance <= self.attack_range

    def ready(self, dist):
        return (
            self.static and dist <= self.attack_range and self.cool_down == 0
        )


class Footman(Unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_health = 300.0
        self.health = self.max_health

        self.attack_range = 2.0
        self.damage = (5, 10)
        # self.attack_animation = 2
        # self.attack_backswing = 2
        self.interval = 10

    def attack(self):
        return random.uniform(*self.damage)


class Rifleman(Unit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_health = 200.0
        self.health = self.max_health

        self.attack_range = 5.0
        self.damage = (20, 30)
        # self.attack_animation = 2
        # self.attack_backswing = 2
        self.interval = 12

    def attack(self):
        return random.uniform(*self.damage)


UNIT_TEMPLATE = {'footman': Footman, 'rifleman': Rifleman}
