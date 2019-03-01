import gym
import numpy as np


class MountainCar:
    def __init__(self, learning_rate, discount_rate, epsilon, min_epsilon, episodes):
        self._discount_rate = discount_rate
        self._env = gym.make("MountainCar-v0")
        self._episodes = episodes
        self._epsilon = epsilon
        self._epsilon_reduction_rate = (epsilon - min_epsilon) / episodes
        self._learning_rate = learning_rate
        self._min_epsilon = min_epsilon
        self._rendered_episodes = 5
        self._q_table = self._initialize_q_table()

    def run(self):
        for episode in range(self._episodes):
            done = False
            initial_state = self._env.reset()
            discrete_state1 = self._discretize_state_space(initial_state)
            while not done:
                self._render(episode)
                action = self._get_action(discrete_state1)
                state2, reward, done, _ = self._env.step(action)
                discrete_state2 = self._discretize_state_space(state2)
                if done and state2[0] >= 0.5:  # successfully reached the flag
                    self._q_table[discrete_state1[0], discrete_state1[1], action] = reward
                else:
                    self._update_q_table(discrete_state1, discrete_state2, action, reward)
                discrete_state1 = discrete_state2
            self._decay_epsilon()
        self._env.close()

    def _initialize_q_table(self):
        """
        The Q table will be a 3 dimensional array where each element can be found using the lookup
        Q(s1, s2, a), where "s1" is the cart position, "s2" is cart velocity, and "a" is the action.

        This function generates an initial Q table with random, uniform values. Since the Q table must cover
        all possible state-action pairs, the table will contain (cart position * cart velocity * number of
        actions) cells.
        """
        state_space = self._get_state_space_range()
        cart_position_range = state_space[0]
        cart_velocity_range = state_space[1]
        action_space_range = self._get_action_space_range()
        return np.random.uniform(low=-1, high=1, size=(cart_position_range, cart_velocity_range, action_space_range))

    def _get_state_space_range(self):
        """
        The state space of the MountainCar environment ranges from [-1.2, -0.07] to [0.6, 0.07] where the
        state vector represents [<cart's position>, <cart's velocity>]. Therefore, the cart's position can
        range from -1.2 to 0.6 and the cart's velocity can range from -0.07 to 0.07.
        """
        state_space = self._env.observation_space.high - self._env.observation_space.low
        return self._discretize_state_space(state_space)

    @staticmethod
    def _discretize_state_space(state_space):
        """
        Since the state space is continuous (i.e. floats and not integers), and Q learning requires a finite
        number of state-action pairs, we can convert the state space to discrete values by first multiplying
        the first element (cart's position) by 10 and the second element (cart's velocity) by 100, then
        rounding the result, and finally converting that result to integers.
        """
        adjusted_state_space = state_space * np.array([10, 100])
        return np.round(adjusted_state_space, decimals=0).astype(int)

    def _get_action_space_range(self):
        """
        The action space is 3 discrete values (i.e. 0, 1, 2) corresponding to the cart moving left,
        moving right, or doing nothing
        """
        return self._env.action_space.n

    def _render(self, episode):
        """
        Only render the final few episodes.

        This speeds up processing, but still shows the episodes once the learning has nearly finished.
        """
        if episode >= (self._episodes - self._rendered_episodes):
            self._env.render()

    def _get_action(self, state):
        """
        Get the next action. This action will either be the best possible action for the given state, or
        a random action using the epsilon-greedy learning strategy.
        """
        if np.random.random() < 1 - self._epsilon:
            return np.argmax(self._q_table[state[0], state[1]])
        else:
            return np.random.randint(0, self._env.action_space.n)

    def _update_q_table(self, old_state, new_state, action, reward):
        """
        Update the Q-value for the old state using the observed reward of the old state and the maximum
        possible reward for the new state.
        """
        max_possible_reward = np.max(self._q_table[new_state[0], new_state[1]])
        observed_reward = self._q_table[old_state[0], old_state[1], action]
        delta = self._learning_rate * (reward + self._discount_rate * max_possible_reward - observed_reward)
        self._q_table[old_state[0], old_state[1], action] += delta

    def _decay_epsilon(self):
        """
        Linearly decay epsilon each episode over the total number of episodes.  This means we will focus
        learning at the beginning of the run, and use learned actions more frequently as the episodes
        progress.
        """
        if self._epsilon > self._min_epsilon:
            self._epsilon -= self._epsilon_reduction_rate


if __name__ == "__main__":
    MountainCar(0.2, 0.9, 0.8, 0, 5000).run()
