#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

from gym_blade.env import BladeEnv


class BaseAgent:
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])

            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action

        return random.choice(ava_actions)


def play(max_episode=10):
    episode = 0
    start_mark = 'white'
    env = BladeEnv()
    agents = [BaseAgent('white'), BaseAgent('black')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        while not done:
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)

                break
            else:
                _, mark = state

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1


if __name__ == "__main__":
    play()
