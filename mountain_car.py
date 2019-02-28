import gym
import numpy as np


class MountainCar:
    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self._env.reset()  # initialize the environment
        self._q_table = self._initialize_q_table()

    def run(self):
        pass

    def _initialize_q_table(self):
        """
        Q table values will look like Q(s1, s2, a) where s1 is the cart position and s2 is cart velocity.

        The action space is 0, 1, and 2.

        This function generates an initial Q table with random, uniform values. The size of the table will
        be (cart position * cart velocity * number of actions).
        """
        state_space = self._get_state_space()
        return np.random.uniform(low=-1, high=1, size=(state_space[0], state_space[1], self._env.action_space.n))

    def _get_state_space(self):
        """
        The state space of the MountainCar environment ranges from [-1.2 -0.07] to [0.6 0.07] where the
        state vector represents [<cart's position> <cart's velocity>]. Therefore, the cart's position can
        range from -1.2 to 0.6 and the cart's velocity can range from -0.07 to 0.07.

        Since the state space is continuous (i.e. floats and not integers), and Q learning requires each
        state-action pair to be visited a sufficiently large number of times, we can "discretize" the state
        space by rounding the first element of the state vector (cart's position) to the nearest 0.1 and
        the second element of the state vector (cart's velocity) to the nearest 0.01, and then multiplying
        the first element by 10 and the second element by 100.  This will give us a state space of 855
        state-action pairs.
        """
        state_space = self._env.observation_space.high - self._env.observation_space.low
        state_space = state_space * np.array([10, 100])
        return np.round(state_space, decimals=0).astype(int)


if __name__ == '__main__':
    MountainCar().run()
