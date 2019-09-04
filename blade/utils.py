HEIGHT = 720
WIDTH = 960
FPS = 10

GOLD_SPEED = 1.0
LANE_LENGTH = 100.0
INITIAL_GOLD = 90.0
MOVEMENT_SPEED = 1.0
POPULATION_LIMIT = 7
REPAIR_COST = 0.5

ACTIONS = ['null', 'barrack', 'blacksmith', 'windmill', 'footman', 'rifleman']

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


class Unit:
    def __init__(self):
        self.name = None

        self.gold_cost = None
        self.time_cost = None

        self.require = []

        self.health = None
        self.attack = None


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


class Unit:
    def __init__(self):
        pass
