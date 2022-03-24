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

```python id="4sv1j8lRtwKQ" executionInfo={"status": "ok", "timestamp": 1630135087132, "user_tz": -330, "elapsed": 2231, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-bok"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="n8RiXVLstuoO"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="drkDUesKtuoT"
!git status
```

```python id="pWkwGc0ltuoT"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="RfQnUMWgjmMA" -->
---
<!-- #endregion -->

```python id="1RMFWE0aoa5y" executionInfo={"status": "ok", "timestamp": 1630135236805, "user_tz": -330, "elapsed": 565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import random
import numpy as np
```

<!-- #region id="Bqbxh9_hsiF_" -->
## Car Collision Problem
Suppose that we have a map of a parking lot, including a self-driving car, other cars, and pedestrians. Our self-driving car can move north, south, east, and west. The other cars and pedestrians remain stationary in this example.

The goal is for our car to navigate the road to its owner while colliding with as few cars and pedestrians as possible—ideally, not colliding with anything. Colliding with a car is not good because it damages the vehicles, but colliding with a pedestrian is more severe. In this problem, we want to minimize collisions, but if we have a choice between colliding with a car and a pedestrian, we should choose the car.
<!-- #endregion -->

<!-- #region id="f6psLuCzsp0n" -->
<!-- #endregion -->

<!-- #region id="Ag-a_amMsyO7" -->
The self-driving car, or agent, can take several actions in the environment. In this simple example, the actions are moving north, south, east, and west. Choosing an action results in the agent moving one block in that direction. The agent can’t move diagonally.

When actions are taken in the environment, rewards or penalties occur. A collision with another car is bad; a collision with a pedestrian is terrible. A move to an empty space is good; finding the owner of the self-driving car is better. The specified rewards aim to discourage collisions with other cars and pedestrians, and to encourage moving into empty spaces and reaching the owner. Note that there could be a reward for out-of-bounds movements, but we will simply disallow this possibility for the sake of simplicity.
<!-- #endregion -->

<!-- #region id="Ewo6Z9uFtAGR" -->
<!-- #endregion -->

```python id="fhXWjOG-odB7" executionInfo={"status": "ok", "timestamp": 1630135304031, "user_tz": -330, "elapsed": 924, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Set constants that represent symbols, rewards, and actions in the environment
ROAD_AGENT = '*'

ROAD_EMPTY = ' '
ROAD_EMPTY_REWARD = 100

ROAD_OBSTACLE_CAR = '#'
ROAD_OBSTACLE_CAR_REWARD = -100

ROAD_OBSTACLE_PERSON = '!'
ROAD_OBSTACLE_PERSON_REWARD = -1000

ROAD_GOAL = '@'
ROAD_GOAL_REWARD = 500

ROAD_OUT_OF_BOUNDS = '-'
ROAD_OUT_OF_BOUNDS_REWARD = -5

COMMAND_NORTH = 0
COMMAND_SOUTH = 1
COMMAND_EAST = 2
COMMAND_WEST = 3
```

```python id="BtPEk6cOotjP" executionInfo={"status": "ok", "timestamp": 1630135314581, "user_tz": -330, "elapsed": 662, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Get a random action
def get_random_action():
    return random.randint(0, 3)
```

```python id="alSa2KuqpBNQ" executionInfo={"status": "ok", "timestamp": 1630135394152, "user_tz": -330, "elapsed": 926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Initialize a map with the starting point and goal
DEFAULT_START_X = 0
DEFAULT_START_Y = 0
DEFAULT_GOAL_X = 9
DEFAULT_GOAL_Y = 5
DEFAULT_ROAD_SIZE_X = 10
DEFAULT_ROAD_SIZE_Y = 10
DEFAULT_ROAD = [[' ', '#', '#', '#', ' ', '#', ' ', ' ', '#', ' '],
                [' ', '!', ' ', ' ', ' ', ' ', ' ', '!', ' ', '!'],
                [' ', '#', '#', ' ', '#', '#', ' ', ' ', ' ', '#'],
                [' ', ' ', ' ', ' ', ' ', ' ', '#', '#', ' ', '#'],
                ['#', '!', '!', '#', ' ', '!', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', '#', ' ', ' ', '#', '#', '#', ' '],
                [' ', '#', '#', '#', '!', ' ', '#', '#', '!', ' '],
                [' ', ' ', ' ', '!', '#', ' ', '!', ' ', ' ', ' '],
                ['#', '!', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                [' ', '#', ' ', '#', '#', '@', ' ', '#', ' ', '!']]
```

<!-- #region id="42GzNb5kte0Z" -->
### Simulator
<!-- #endregion -->

<!-- #region id="5lGynxSNodFz" -->
The simulator should provide the following functionality and information as a minimum.

- Initialize the environment: This involves resetting the environment including the agent to the starting state.

- Get the current state of the environment: This function should provide the current state of the environment.
The state of the environment will change after each action is performed.

- Apply action to environment: This involves having the agent applying an action to the environment.
The environment will be impacted by the action which may result in a reward.

- Calculate the reward of an action: This is related to the "apply action to environment” function.
The reward for the action and impact in the environment needs to be calculated.

- Determine if the goal is achieved: This function results in whether or not the agent has achieved the goal.
This can also sometimes be represented as “is complete”; in the case of an environment where the goal could
potentially not be achieved, the simulator needs to signal completion when it deems necessary.
<!-- #endregion -->

```python id="BEk32f1Rpx1S" executionInfo={"status": "ok", "timestamp": 1630135593228, "user_tz": -330, "elapsed": 837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# The Simulator class encompasses the functions for the simulation environment
class Simulator:

    # Initialize the simulator with a road, its size, the starting point, and the goal point
    def __init__(self, road, road_size_x, road_size_y, agent_start_x, agent_start_y, goal_x, goal_y):
        self.road_size_x = road_size_x
        self.road_size_y = road_size_y
        self.road = road
        self.rewards = 0
        self.agent_x = agent_start_x
        self.agent_y = agent_start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.states = []

    # Move the agent and return the reward based on a command. The command is an action (North, South, East, West)
    def move_agent(self, command):
        reward_update = 0
        next_x = 0
        next_y = 0
        if command == COMMAND_NORTH:
            next_x = self.agent_x - 1
            next_y = self.agent_y
        elif command == COMMAND_SOUTH:
            next_x = self.agent_x + 1
            next_y = self.agent_y
        elif command == COMMAND_EAST:
            next_x = self.agent_x
            next_y = self.agent_y + 1
        elif command == COMMAND_WEST:
            next_x = self.agent_x
            next_y = self.agent_y - 1
        if self.is_within_bounds(next_x, next_y):
            reward_update = self.calculate_movement_reward(next_x, next_y)
            self.agent_x = next_x
            self.agent_y = next_y
            # print('Origin ', self.player_x, ',', self.player_y)
            # print('Target ', next_x, ',', next_y)
        else:
            reward_update = ROAD_OUT_OF_BOUNDS_REWARD
        self.rewards += reward_update
        #print('Immediate Reward: ', reward_update)
        #print('Lifetime Reward: ', self.rewards)
        return reward_update

    # Move the agent to a target point
    def move(self, target_x, target_y):
        self.agent_x = target_x
        self.agent_y = target_y

    # Calculate the reward for the movement based on an action
    def calculate_movement_reward(self, next_x, next_y):
        if self.road[next_x][next_y] == ROAD_OBSTACLE_PERSON:
            return ROAD_OBSTACLE_PERSON_REWARD
        elif self.road[next_x][next_y] == ROAD_OBSTACLE_CAR:
            return ROAD_OBSTACLE_CAR_REWARD
        elif self.road[next_x][next_y] == ROAD_GOAL:
            return ROAD_GOAL_REWARD
        else:
            return ROAD_EMPTY_REWARD

    # Determine if the target point is within bounds
    def is_within_bounds(self, target_x, target_y):
        if self.road_size_x > target_x >= 0 and self.road_size_y > target_y >= 0:
            return True
        return False

    # Determine if the goal is achieved
    def is_goal_achieved(self):
        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            return True
        return False

    # Get the state. This is a string encoding based on the immediate neighbors around the current point
    def get_state(self):
        state = ''
        for x in range(-1, 2):
            for y in range(-1, 2):
                if self.is_within_bounds(x, y):
                    state += self.road[x][y]
                else:
                    state += ROAD_OUT_OF_BOUNDS
        if state not in self.states:
            self.states.append(state)
        return self.states.index(state)

    def print_road(self):
        output = ''
        for x in range(self.road_size_x):
            for y in range(self.road_size_y):
                if x == self.agent_x and y == self.agent_y:
                    output += ROAD_AGENT
                else:
                    output += self.road[x][y]
            output += '\n'
        print('Agent x: ', self.agent_x)
        print('Agent y: ', self.agent_y)
        print(output)
```

<!-- #region id="Usg6kKzstjRt" -->
### A good solution
<!-- #endregion -->

<!-- #region id="hCXdn_CQtb8e" -->
<!-- #endregion -->

```python id="QcD71fZQp3SE" executionInfo={"status": "ok", "timestamp": 1630135616072, "user_tz": -330, "elapsed": 924, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Execute a known most rewarding path
def execute_happy_path():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.print_road()
```

<!-- #region id="XhUDvkRgtlFJ" -->
### A bad solution
<!-- #endregion -->

<!-- #region id="nq0YoCtytqc3" -->
<!-- #endregion -->

```python id="zQciZ_COp5KX" executionInfo={"status": "ok", "timestamp": 1630135623632, "user_tz": -330, "elapsed": 848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Execute a known path with penalties
def execute_sad_path():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    simulator.move_agent(COMMAND_NORTH)
    simulator.move_agent(COMMAND_WEST)
    simulator.move_agent(COMMAND_SOUTH)
    simulator.move_agent(COMMAND_EAST)
    simulator.print_road()
```

```python id="2OKc88wBp7nk" executionInfo={"status": "ok", "timestamp": 1630135633975, "user_tz": -330, "elapsed": 935, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Execute a random brute force approach until the goal is found
def execute_random_brute_force():
    simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                          DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
    simulator.print_road()
    while not simulator.is_goal_achieved():
        simulator.move_agent(get_random_action())
        simulator.print_road()
    simulator.print_road()
```

<!-- #region id="AJfTWk75qRXu" -->
### Train the Q-table using Q-learning
- Initialize simulator:
This involves resetting the environment to the starting state with the agent in a neutral state.

- Get environment state:
This function should provide the current state of the environment. The state of the environment will change after
each action is performed.

- Is goal achieved:
Determine if the goal is achieved (or the simulator deems the exploration to be complete). In our example, this
would be picking up the owner of the self-driving car. If the goal is achieved, the algorithm will end.

- Pick a random action:
Determine if a random action should be selected; if so, a random action will be selected out of north, south, east,
and west. Random actions are useful to explore the possibilities in the environment instead of learn only a narrow
subset.

- Reference action in Q-table: If the decision to select a random action is not selected, the current environment
state is transposed to the Q-table and the respective action is selected based on the values in table. More about
the Q-table is coming up.

- Apply action to environment: This involves applying the selected action to the environment; whether that is a
random action or an action selected from the Q-table. An action will have a consequence in the environment and
yield a reward.

- Update Q-table: describes the concepts involved in updating the Q-table and the steps that are
carried out.
Learning rate is represented as alpha.
Discount is represented as gamma.
<!-- #endregion -->

```python id="-OtB6Z53rXKe" executionInfo={"status": "ok", "timestamp": 1630136007967, "user_tz": -330, "elapsed": 586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def train_with_q_learning(observation_space, action_space, number_of_iterations, learning_rate, discount,
                          chance_of_random_move):
    # Initialize the Q-table
    q_table = np.zeros([observation_space, action_space], dtype=np.int8)

    # Repeat for a number of iterations
    for i in range(number_of_iterations):
        # Reset the simulator
        simulator = Simulator(DEFAULT_ROAD, DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y, DEFAULT_START_X, DEFAULT_START_Y,
                              DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
        state = simulator.get_state()
        done = False

        # Continue while the simulator is not terminated
        while not done:
            action = COMMAND_SOUTH
            # Choose a random action or choose the best move from the Q-table given the current state
            if random.uniform(0, 1) > chance_of_random_move: #exploration-exploitation
                action = get_random_action()
            else:
                action = np.argmax(q_table[state])

            # Apply the selected action to the simulation and get the changed state and reward
            reward = simulator.move_agent(action)
            next_state = simulator.get_state()
            done = simulator.is_goal_achieved()

            print(simulator.get_state())
            print('State: ', state)
            print('Action: ', action)

            # Calculate the Q-table value for the selected action
            current_value = q_table[state, action]
            next_state_max_value = np.max(q_table[next_state])
            new_value = (1 - learning_rate) * current_value + learning_rate * (reward + discount * next_state_max_value)
            q_table[state, action] = new_value
            state = next_state
            print(q_table)

    return q_table
```

<!-- #region id="gW5LFBYHuOzU" -->
The key aspect of Q-learning is the equation used to update the values of the Q-table. This equation is based on the Bellman equation, which determines the value of a decision made at a certain point in time, given the reward or penalty for making that decision. The Q-learning equation is an adaptation of the Bellman equation. In the Q-learning equation, the most important properties for updating Q-table values are the current state, the action, the next state given the action, and the reward outcome. The learning rate is similar to the learning rate in supervised learning, which determines the extent to which a Q-table value is updated. The discount is used to indicate the importance of possible future rewards, which is used to balance favoring immediate rewards versus long-term rewards:
<!-- #endregion -->

<!-- #region id="W6zT2A5muQmB" -->
<!-- #endregion -->

<!-- #region id="eKxtWPlmuwIs" -->
Next, we explore how to update the Q-table by using the Q-learning equation based on different actions with different reward values. These values will be used for the learning rate (alpha) and discount (gamma):

Learning rate (alpha): 0.1

Discount (gamma): 0.6

Below figure illustrates how the Q-learning equation is used to update the Q-table, if the agent selects the East action from the initial state in the first iteration. Remember that the initial Q-table consists of 0s. The learning rate (alpha), discount (gamma), current action value, reward, and next best state are plugged into the equation to determine the new value for the action that was taken. The action is East, which results in a collision with another car, which yields -100 as a reward. After the new value is calculated, the value of East on state 1 is -10.
<!-- #endregion -->

<!-- #region id="NP_siTZMu085" -->
<!-- #endregion -->

```python id="1zfK4Z7ajrSl" executionInfo={"status": "ok", "timestamp": 1630136033393, "user_tz": -330, "elapsed": 841, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Use the trained Q-table to navigate the map to the goal, to test the training
def execute_with_q_learning(q_table, number_of_episodes):
    total_epochs = 0
    total_penalties_person = 0
    total_penalties_car = 0

    # Repeat for a number of episodes
    for episode in range(number_of_episodes):
        # Initialize a simulator
        simulator = Simulator(DEFAULT_ROAD,
                              DEFAULT_ROAD_SIZE_X, DEFAULT_ROAD_SIZE_Y,
                              DEFAULT_START_X, DEFAULT_START_Y,
                              DEFAULT_GOAL_X, DEFAULT_GOAL_Y)
        state = simulator.get_state()
        epochs = 0
        penalties_person = 0
        penalties_car = 0
        reward = 0
        done = False

        # Choose an action from the Q-table while the simulator is not terminated
        while not done:
            action = np.argmax(q_table[state])

            reward = simulator.move_agent(action)
            state = simulator.get_state()
            done = simulator.is_goal_achieved()

            if reward == ROAD_OBSTACLE_PERSON_REWARD:
                penalties_person += 1
            elif reward == ROAD_OBSTACLE_CAR_REWARD:
                penalties_car += 1

            epochs += 1

        # Calculate the penalties accrued
        total_penalties_person += penalties_person
        total_penalties_car += penalties_car
        total_epochs += epochs

    print('Results after ', number_of_episodes, 'episodes:')
    print('Average time steps per episode: ', total_epochs / number_of_episodes)
    print('Average person penalties per episode: ', total_penalties_person / number_of_episodes)
    print('Average car penalties per episode: ', total_penalties_person / number_of_episodes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="s87qFTVsrfdN" executionInfo={"status": "ok", "timestamp": 1630136044839, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="62d0d033-fd32-48d5-bef0-876f1a6741f7"
execute_happy_path()
```

```python colab={"base_uri": "https://localhost:8080/"} id="O6T20HlCrkJ-" executionInfo={"status": "ok", "timestamp": 1630136062353, "user_tz": -330, "elapsed": 826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="419650d8-9bd9-498d-ae2b-1c1687fa7d89"
execute_sad_path()
```

```python colab={"base_uri": "https://localhost:8080/"} id="853wGnxFrmjL" executionInfo={"status": "ok", "timestamp": 1630136072409, "user_tz": -330, "elapsed": 992, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="563ea7be-ec33-48e2-e8ca-b01e08fd7332"
execute_random_brute_force()
```

```python id="jQ0KtPtMrdla"
LEARNING_RATE = 0.1
DISCOUNT = 0.6
CHANCE_OF_RANDOM_ACTION = 0.1
trained_q_table = train_with_q_learning(4*4*4*4*4*4*4*4, 4, 100, LEARNING_RATE, DISCOUNT, CHANCE_OF_RANDOM_ACTION)
```

```python id="J7bpI0_lrwea"
execute_with_q_learning(trained_q_table, 1)
```
