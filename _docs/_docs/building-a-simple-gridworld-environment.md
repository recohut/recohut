---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="JgaLL-rEdzOR" -->
# Building a simple Gridworld Environment
<!-- #endregion -->

```python id="C2yeuMR4VLaG"
import copy
import sys
import numpy as np
from collections import defaultdict
```

```python id="AwcVbA9zVSX5"
# Grid cell state and color mapping
EMPTY = BLACK = 0
WALL = GRAY = 1
AGENT = BLUE = 2
BOMB = RED = 3
GOAL = GREEN = 4
```

```python id="mNWdDHcXVsx3"
# RGB color value table
COLOR_MAP = {
    BLACK: [0.0, 0.0, 0.0],
    GRAY: [0.5, 0.5, 0.5],
    BLUE: [0.0, 0.0, 1.0],
    RED: [1.0, 0.0, 0.0],
    GREEN: [0.0, 1.0, 0.0],
}
```

```python id="AN7AOH5DVv3y"
# Action mapping
NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4
```

```python id="yCiTDjM9VxV2"
class GridworldEnv(gym.Env):
    def __init__(self, max_steps=100):
        """Initialize Gridworld
        Args:
            max_steps (int, optional): Max steps per episode. Defaults to 100.
        """
        # Observations
        self.grid_layout = """
        1 1 1 1 1 1 1 1
        1 2 0 0 0 0 0 1
        1 0 1 1 1 0 0 1
        1 0 1 0 1 0 0 1
        1 0 1 4 1 0 0 1
        1 0 3 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 1 1 1 1 1 1 1
        """
        self.initial_grid_state = np.fromstring(self.grid_layout, dtype=int, sep=" ")
        self.initial_grid_state = self.initial_grid_state.reshape(8, 8)
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=self.grid_state.shape
        )
        self.img_shape = [256, 256, 3]
        self.metadata = {"render.modes": ["human"]}
        # Actions
        self.action_space = gym.spaces.Discrete(5)
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.action_pos_dict = defaultdict(
            lambda: [0, 0],
            {
                NOOP: [0, 0],
                UP: [-1, 0],
                DOWN: [1, 0],
                LEFT: [0, -1],
                RIGHT: [0, 1],
            },
        )
        (self.agent_state, self.goal_state) = self.get_state()
        self.step_num = 0  # To keep track of number of steps
        self.max_steps = max_steps
        self.done = False
        self.info = {"status": "Live"}
        self.viewer = None

    def step(self, action):
        """Return next observation, reward, done , info"""
        action = int(action)
        reward = 0.0

        next_state = (
            self.agent_state[0] + self.action_pos_dict[action][0],
            self.agent_state[1] + self.action_pos_dict[action][1],
        )

        next_state_invalid = (
            next_state[0] < 0 or next_state[0] >= self.grid_state.shape[0]
        ) or (next_state[1] < 0 or next_state[1] >= self.grid_state.shape[1])
        if next_state_invalid:
            # Leave the agent state unchanged
            next_state = self.agent_state
            self.info["status"] = "Next state is invalid"

        next_agent_state = self.grid_state[next_state[0], next_state[1]]

        # Calculate reward
        if next_agent_state == EMPTY:
            # Move agent from previous state to the next state on the grid
            self.info["status"] = "Agent moved to a new cell"
            self.grid_state[next_state[0], next_state[1]] = AGENT
            self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
            self.agent_state = copy.deepcopy(next_state)

        elif next_agent_state == WALL:
            self.info["status"] = "Agent bumped into a wall"
            reward = -0.1
        # Terminal states
        elif next_agent_state == GOAL:
            self.info["status"] = "Agent reached the GOAL "
            self.done = True
            reward = 1
        elif next_agent_state == BOMB:
            self.info["status"] = "Agent stepped on a BOMB"
            self.done = True
            reward = -1
        # elif next_agent_state == AGENT:
        else:
            # NOOP or next state is invalid
            self.done = False

        self.step_num += 1

        # Check if max steps per episode has been reached
        if self.step_num >= self.max_steps:
            self.done = True
            self.info["status"] = "Max steps reached"

        if self.done:
            done = True
            terminal_state = copy.deepcopy(self.grid_state)
            terminal_info = copy.deepcopy(self.info)
            _ = self.reset()
            return (terminal_state, reward, done, terminal_info)

        return self.grid_state, reward, self.done, self.info

    def reset(self):
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        (
            self.agent_state,
            self.agent_goal_state,
        ) = self.get_state()
        self.step_num = 0
        self.done = False
        self.info["status"] = "Live"
        return self.grid_state

    def get_state(self):
        start_state = np.where(self.grid_state == AGENT)
        goal_state = np.where(self.grid_state == GOAL)

        start_or_goal_not_found = not (start_state[0] and goal_state[0])
        if start_or_goal_not_found:
            sys.exit(
                "Start and/or Goal state not present in the Gridworld. "
                "Check the Grid layout"
            )
        start_state = (start_state[0][0], start_state[1][0])
        goal_state = (goal_state[0][0], goal_state[1][0])

        return start_state, goal_state

    def gridarray_to_image(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        scale_x = int(observation.shape[0] / self.grid_state.shape[0])
        scale_y = int(observation.shape[1] / self.grid_state.shape[1])
        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                for k in range(3):  # 3-channel RGB image
                    pixel_value = COLOR_MAP[self.grid_state[i, j]][k]
                    observation[
                        i * scale_x : (i + 1) * scale_x,
                        j * scale_y : (j + 1) * scale_y,
                        k,
                    ] = pixel_value
        return (255 * observation).astype(np.uint8)

    def render(self, mode="rgb_array", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self.gridarray_to_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        self.render(close=True)

    @staticmethod
    def get_action_meanings():
        return ["NOOP", "DOWN", "UP", "LEFT", "RIGHT"]
```

```python colab={"base_uri": "https://localhost:8080/"} id="d4tofDM9WPTM" executionInfo={"status": "ok", "timestamp": 1638436549900, "user_tz": -330, "elapsed": 824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b912c877-ca63-4e50-9cc8-928e7cc3c8aa"
if __name__ == "__main__":
    env = GridworldEnv()
    obs = env.reset()
    done = False
    step_num = 1
    # Run one episode
    while not done:
        # Sample a random action from the action space
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(f"step#:{step_num} reward:{reward} done:{done} info:{info}")
        step_num += 1
        env.render()
    env.close()
```

<!-- #region id="gBqT6gGfb140" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1gwfn3wCb143" executionInfo={"status": "ok", "timestamp": 1638436678248, "user_tz": -330, "elapsed": 4059, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5de97eaf-4b69-4334-d0b1-9c1e14e89c6d"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="wF2c-0j5b144" -->
---
<!-- #endregion -->

<!-- #region id="-PH0vMRxb145" -->
**END**
<!-- #endregion -->
