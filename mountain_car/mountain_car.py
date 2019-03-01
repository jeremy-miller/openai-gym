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
        Q table values will look like Q(s1, s2, a) where "s1" is the cart position, "s2" is cart velocity,
        and "a" is the action.

        This function generates an initial Q table with random, uniform values. Since the Q table must cover
        all possible state-action pairs, the size of the table will contain
        (cart position * cart velocity * number of actions) cells.
        """
        state_space = self._get_state_space_range()
        cart_position_range = state_space[0]
        cart_velocity_range = state_space[1]
        action_space_range = self._get_action_space_range()
        return np.random.uniform(low=-1, high=1, size=(cart_position_range, cart_velocity_range, action_space_range))

    def _get_state_space_range(self):
        """
        The state space of the MountainCar environment ranges from [-1.2 -0.07] to [0.6 0.07] where the
        state vector represents [<cart's position> <cart's velocity>]. Therefore, the cart's position can
        range from -1.2 to 0.6 and the cart's velocity can range from -0.07 to 0.07.

        Since the state space is continuous (i.e. floats and not integers), and Q learning requires each
        state-action pair to be visited a sufficiently large number of times, we can convert the state space
        to discrete values by rounding the first element of the state vector (cart's position) to the nearest
        0.1 and the second element of the state vector (cart's velocity) to the nearest 0.01, and then
        multiplying the first element by 10 and the second element by 100.  This will give us a state space
        of 855 unique state-action pairs.
        """
        state_space = self._env.observation_space.high - self._env.observation_space.low
        state_space = state_space * np.array([10, 100])
        return np.round(state_space, decimals=0).astype(int)

    def _get_action_space_range(self):
        """The action space is 3 discrete values (i.e. 0, 1, 2)"""
        return self._env.action_space.n


if __name__ == '__main__':
    MountainCar().run()
