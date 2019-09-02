#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import gym

import gym_blade


class BladeEnvironmentTestCases(unittest.TestCase):
    def test_env(self):
        env = gym.make('Blade-v0')
        self.assertTrue(env is not None)
        env.seed(0)
        env.reset()
        env.step(0)
        self.assertTrue(env is not None)
