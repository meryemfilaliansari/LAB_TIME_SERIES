# agent.py
import numpy as np

class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy policy.
    q_table shape: (size, size, 4)
    """

    def __init__(self, env, lr=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999):
        self.env = env
        self.size = env.size
        self.lr = lr
        self.gamma = gamma

        # exploration
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((self.size, self.size, 4))

    def choose_action(self, state, greedy=False):
        r, c = state
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(4)
        # break ties randomly for better exploration of equal Q-values
        best = np.max(self.q_table[r, c])
        candidates = np.flatnonzero(np.isclose(self.q_table[r, c], best))
        return np.random.choice(candidates)

    def update(self, state, action, reward, next_state):
        r, c = state
        nr, nc = next_state
        old = self.q_table[r, c, action]
        future = np.max(self.q_table[nr, nc])
        self.q_table[r, c, action] = old + self.lr * (reward + self.gamma * future - old)

    def train(self, episodes=1000, max_steps=200, verbose=False):
        """
        Basic training loop, returns stats dict with success rate.
        Episode ends when env.step returns done (goal or obstacle).
        """
        successes = 0
        for ep in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                a = self.choose_action(state)
                next_state, reward, done = self.env.step(a)
                self.update(state, a, reward, next_state)
                state = next_state
                if done:
                    # reward positive => success
                    if reward == self.env.goal_reward:
                        successes += 1
                    break
            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if verbose and (ep % max(1, episodes//10) == 0):
                print(f"Train ep {ep}/{episodes}, eps={self.epsilon:.4f}")
        success_rate = successes / episodes
        return {"episodes": episodes, "successes": successes, "success_rate": success_rate}
