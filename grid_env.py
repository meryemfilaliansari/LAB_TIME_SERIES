# grid_env.py
import numpy as np

class GridEnv:
    """
    Grid environment (Gym-like minimal).
    States: (row, col) with 0 <= row,col < size
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Behavior:
      - If new_pos == goal -> agent moves, reward = goal_reward, done = True
      - If new_pos in obstacles -> agent moves, reward = obstacle_reward, done = True
      - Else -> agent moves, reward = step_reward, done = False
    """
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }

    def __init__(self, size=5, start=(0,0), goal=None, obstacles=None,
                 step_reward=-0.2, goal_reward=50.0, obstacle_reward=-20.0):
        self.size = size
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (size - 1, size - 1)

        # If obstacles not provided, use exactly 2 default ones not overlapping start/goal
        if obstacles is None:
            # two sample obstacles — you can change
            self.obstacles = { (2,2), (3,1) }
        else:
            obs = [tuple(o) for o in obstacles]
            # ensure exactly 2 obstacles (if user passed more/less, we enforce 2 by trimming/padding)
            if len(obs) >= 2:
                self.obstacles = {obs[0], obs[1]}
            elif len(obs) == 1:
                # add a second default if only one provided
                default = (max(1, obs[0][0]+1) % size, (obs[0][1]+1) % size)
                self.obstacles = {obs[0], default}
            else:
                self.obstacles = { (2,2), (3,1) }

        # make sure obstacles don't overlap start/goal
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)
        # If removed one accidentally, ensure we still have 2 by adding a fallback
        while len(self.obstacles) < 2:
            for r in range(size):
                for c in range(size):
                    if (r,c) not in self.obstacles and (r,c) != self.start and (r,c) != self.goal:
                        self.obstacles.add((r,c))
                        break
                if len(self.obstacles) >= 2:
                    break

        # rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward

        # state
        self.reset()

    def reset(self, start=None):
        if start is not None:
            self.start = tuple(start)
        self.agent_pos = tuple(self.start)
        return self.agent_pos

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def step(self, action):
        """
        action: int 0..3
        returns: next_state (tuple), reward (float), done (bool)
        """
        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # keep in bounds
        if not self.in_bounds((nr, nc)):
            nr, nc = r, c  # bump -> stay in place (and give step reward)

        new_pos = (nr, nc)

        # if move into obstacle -> move there and episode ends (obstacle reached)
        if new_pos in self.obstacles:
            self.agent_pos = new_pos
            return self.agent_pos, self.obstacle_reward, True

        # if move into goal -> move there and episode ends (success)
        if new_pos == self.goal:
            self.agent_pos = new_pos
            return self.agent_pos, self.goal_reward, True

        # normal step
        self.agent_pos = new_pos
        return self.agent_pos, self.step_reward, False

    # utility for display: Qmax grid shaped (row-major)
    def qmax_to_grid(self, Q):
        """
        Q: numpy array shape (size, size, 4)
        returns grid (size, size) of max over actions
        """
        return np.max(Q, axis=2)

    def get_info(self):
        return {
            "size": self.size,
            "start": self.start,
            "goal": self.goal,
            "obstacles": set(self.obstacles),
            "rewards": {"step": self.step_reward, "goal": self.goal_reward, "obstacle": self.obstacle_reward}
        }
