import numpy as np
import gym
import matplotlib.pyplot as plt


class MountainCar:
    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self._env.reset()  # initialize the environment

    def run(self):
        pass


if __name__ == '__main__':
    MountainCar().run()
