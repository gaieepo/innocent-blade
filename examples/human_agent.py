#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import click

from gym_blade.env import BladeEnv


class HumanAgent:
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        return
