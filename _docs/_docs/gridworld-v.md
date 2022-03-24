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

<!-- #region id="2_l6TVZdgsOl" -->
# Building a simple Gridworld v2 Environment
<!-- #endregion -->

```python id="IB4O1CBZtogr"
import gym
import numpy as np
```

```python id="fLP8mGdPtrxQ"
class GridworldV2Env(gym.Env):
    def __init__(self, step_cost=-0.2, max_ep_length=500, explore_start=False):
        self.index_to_coordinate_map = {
            "0": [0, 0],
            "1": [0, 1],
            "2": [0, 2],
            "3": [0, 3],
            "4": [1, 0],
            "5": [1, 1],
            "6": [1, 2],
            "7": [1, 3],
            "8": [2, 0],
            "9": [2, 1],
            "10": [2, 2],
            "11": [2, 3],
        }
        self.coordinate_to_index_map = {
            str(val): int(key) for key, val in self.index_to_coordinate_map.items()
        }
        self.map = np.zeros((3, 4))
        self.observation_space = gym.spaces.Discrete(1)
        self.distinct_states = [str(i) for i in range(12)]
        self.goal_coordinate = [0, 3]
        self.bomb_coordinate = [1, 3]
        self.wall_coordinate = [1, 1]
        self.goal_state = self.coordinate_to_index_map[str(self.goal_coordinate)]  # 3
        self.bomb_state = self.coordinate_to_index_map[str(self.bomb_coordinate)]  # 7
        self.map[self.goal_coordinate[0]][self.goal_coordinate[1]] = 1
        self.map[self.bomb_coordinate[0]][self.bomb_coordinate[1]] = -1
        self.map[self.wall_coordinate[0]][self.wall_coordinate[1]] = 2

        self.exploring_starts = explore_start
        self.state = 8
        self.done = False
        self.max_ep_length = max_ep_length
        self.steps = 0
        self.step_cost = step_cost
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
        self.possible_actions = list(self.action_map.values())

    def reset(self):
        self.done = False
        self.steps = 0
        self.map = np.zeros((3, 4))
        self.map[self.goal_coordinate[0]][self.goal_coordinate[1]] = 1
        self.map[self.bomb_coordinate[0]][self.bomb_coordinate[1]] = -1
        self.map[self.wall_coordinate[0]][self.wall_coordinate[1]] = 2

        if self.exploring_starts:
            self.state = np.random.choice([0, 1, 2, 4, 6, 8, 9, 10, 11])
        else:
            self.state = 8
        return self.state

    def get_next_state(self, current_position, action):

        next_state = self.index_to_coordinate_map[str(current_position)].copy()

        if action == 0 and next_state[0] != 0 and next_state != [2, 1]:
            # Move up
            next_state[0] -= 1
        elif action == 1 and next_state[1] != 3 and next_state != [1, 0]:
            # Move right
            next_state[1] += 1
        elif action == 2 and next_state[0] != 2 and next_state != [0, 1]:
            # Move down
            next_state[0] += 1
        elif action == 3 and next_state[1] != 0 and next_state != [1, 2]:
            # Move left
            next_state[1] -= 1
        else:
            pass
        return self.coordinate_to_index_map[str(next_state)]

    def step(self, action):
        assert action in self.possible_actions, f"Invalid action:{action}"

        current_position = self.state
        next_state = self.get_next_state(current_position, action)

        self.steps += 1

        if next_state == self.goal_state:
            reward = 1
            self.done = True

        elif next_state == self.bomb_state:
            reward = -1
            self.done = True
        else:
            reward = self.step_cost

        if self.steps == self.max_ep_length:
            self.done = True

        self.state = next_state
        return next_state, reward, self.done
```

<!-- #region id="m56HRuQatzsF" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5OJ01_h1tzsI" executionInfo={"status": "ok", "timestamp": 1638441388194, "user_tz": -330, "elapsed": 3753, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="db7eed13-feb9-4a28-88d5-cd8e5398bc76"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="eGbBJ6D9tzsJ" -->
---
<!-- #endregion -->

<!-- #region id="EXVMZ6h8tzsL" -->
**END**
<!-- #endregion -->
