import random
import uuid

import numpy as np

# env related
SEED = 42
MAX_GLOBAL_TIME = 5e4
MIN_MATCHES = 10
STATE_SIZE = 205

GAMMA = 0.99
LR = 2.5e-4
CLIP_RANGE = 0.1
EPS = np.finfo(np.float32).eps.item()
ELECT_THRESHOLD = 0.6


def generate_id():
    return uuid.uuid4().hex[:8]


# game related
LEFT = 0
RIGHT = 1

HEIGHT = 720
WIDTH = 960
FPS = 30

INITIAL_GOLD = 150.0
GOLD_SPEED = 0.4
WINDMILL_GOLD_SPEED = 0.6
TRANSPORT_GOLD_SPEED = 0.9
ALCHEMY_GOLD_SPEED = 1.3

LANE_LENGTH = 150.0

FULL_MAX_POPULATION = 7
DEFAULT_MAX_POPULATION = 4
BLACKSMITH_POPULATION_INCREMENT = 1
KEEP_POPULATION_INCREMENT = 2

FULL_ACTIONS = [
    'noop',
    'barrack',
    'blacksmith',
    'windmill',
    'steel_blade',
    'long_barrelled_gun',
    'keep',
    'watchtower',
    'monastery',
    'transport',
    'footman',
    'rifleman',
    'monk',
    'forward',
    'backward',
    'repair',
    'stop_repair',
]

SIMPLE_ACTIONS = [action for action in FULL_ACTIONS if action not in ['backward', 'stop_repair']]

SIMPLE_TECHS = {
    'barrack': {
        'name': 'Barrack',
        'visible': True,
        'require': [],
        'time_cost': 230,
        'count_down': 230,
        'gold_cost': 120,
        'built': False,
        'building': False,
    },
    'blacksmith': {
        'name': 'Blacksmith',
        'visible': True,
        'require': [],
        'time_cost': 400,
        'count_down': 400,
        'gold_cost': 220,
        'built': False,
        'building': False,
    },
    'windmill': {
        'name': 'Windmill',
        'visible': True,
        'require': [],
        'time_cost': 560,
        'count_down': 560,
        'gold_cost': 150,
        'built': False,
        'building': False,
    },
    'steel_blade': {
        'name': 'Steel Blade',
        'visible': False,
        'require': ['blacksmith'],
        'time_cost': 450,
        'count_down': 450,
        'gold_cost': 100,
        'built': False,
        'building': False,
    },
    'long_barrelled_gun': {
        'name': 'Long-barrelled Gun',
        'visible': False,
        'require': ['blacksmith'],
        'time_cost': 580,
        'count_down': 580,
        'gold_cost': 180,
        'built': False,
        'building': False,
    },
    'keep': {
        'name': 'Keep',
        'visible': True,
        'require': [],
        'time_cost': 600,
        'count_down': 600,
        'gold_cost': 500,
        'built': False,
        'building': False,
    },
    'watchtower': {
        'name': 'Watchtower',
        'visible': True,
        'require': ['blacksmith'],
        'time_cost': 490,
        'count_down': 490,
        'gold_cost': 160,
        'built': False,
        'building': False,
    },
    'monastery': {
        'name': 'Monastery',
        'visible': True,
        'require': ['keep'],
        'time_cost': 360,
        'count_down': 360,
        'gold_cost': 210,
        'built': False,
        'building': False,
    },
    'transport': {
        'name': 'Transport',
        'visible': False,
        'require': ['keep'],
        'time_cost': 580,
        'count_down': 580,
        'gold_cost': 260,
        'built': False,
        'building': False,
    },
}

TECHS = {
    'barrack': {'name': 'Barrack', 'require': [], 'built': False, 'building': False},
    'keep': {'name': 'Keep', 'require': [], 'built': False, 'building': False},
    'blacksmith': {'name': 'Blacksmith', 'require': [], 'built': False, 'building': False},
    'windmill': {'name': 'Windmill', 'require': [], 'built': False, 'building': False},
    'watchtower': {'name': 'Watchtower', 'require': ['blacksmith'], 'built': False, 'building': False},
    'steel_blade': {'name': 'Steel Blade', 'require': ['blacksmith'], 'built': False, 'building': False},
    'long_barrelled_gun': {'name': 'Long-barrelled Gun', 'require': ['blacksmith'], 'built': False, 'building': False},
    'iron_horse': {'name': 'Iron Horse', 'require': ['blacksmith', 'castle'], 'built': False, 'building': False},
    'chancel': {'name': 'Chancel', 'require': ['keep'], 'built': False, 'building': False},
    'monastery': {'name': 'Monastery', 'require': ['keep'], 'built': False, 'building': False},
    'workshop': {'name': 'Workshop', 'require': ['keep'], 'built': False, 'building': False},
    'castle': {'name': 'Castle', 'require': ['keep'], 'built': False, 'building': False},
    'telescope': {'name': 'Telescope', 'require': ['keep', 'watchtower'], 'built': False, 'building': False},
    'transport': {'name': 'Transport', 'require': ['keep'], 'built': False, 'building': False},
    'natural_power': {'name': 'Natural Power', 'require': ['chancel'], 'built': False, 'building': False},
    'iron_gate': {'name': 'Iron Gate', 'require': ['workshop'], 'built': False, 'building': False},
    'alchemy': {'name': 'Alchemy', 'require': ['chancel'], 'built': False, 'building': False},
    'noble_metal': {'name': 'Noble Metal', 'require': ['monastery', 'castle'], 'built': False, 'building': False},
}

UNITS = {
    'footman': {
        'name': 'Footman',
        'require': ['barrack'],
        'gold_cost': 70,
        'time_cost': 180,
        'count_down': 180,
        'building': False,
    },
    'rifleman': {
        'name': 'Rifleman',
        'require': ['barrack', 'blacksmith'],
        'gold_cost': 90,
        'time_cost': 250,
        'count_down': 250,
        'building': False,
    },
    'monk': {
        'name': 'Monk',
        'require': ['monastery'],
        'gold_cost': 80,
        'time_cost': 275,
        'count_down': 275,
        'building': False,
    },
}

UNIT_INDEX = {'Footman': 0, 'Rifleman': 1, 'Monk': 2}


VIZ = {'not': (255, 0, 1), 'can': (0, 0, 255), 'building': (255, 255, 0), 'built': (0, 255, 0)}
