import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import time

RIGHT_ONLY = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],

]


ENV_NAME = 'SuperMarioBros-1-1-v0'

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

done = False
env.reset()




while not done:
    action = env.action_space.sample()
    _,_, done,_,_ = env.step(action)
    env.render()
    time.sleep(0.001)