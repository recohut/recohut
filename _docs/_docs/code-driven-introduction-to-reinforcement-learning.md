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

<!-- #region id="EBLfK_E_BfSd" -->
# Code-Driven Introduction to Reinforcement Learning
<!-- #endregion -->

<!-- #region id="CvZ1HLbBBibe" -->
In the following examples I intentionally use a very simple and visual example. This makes it easier to understand, since that is the main goal. Although there are industrial examples that are similar, real-life implementations are likely to be more complex.
<!-- #endregion -->

<!-- #region id="26JqEZ-ABj2R" -->
## The Markov Decision Process
The Markov decision process (MDP) is a mathematical framework that helps you encapsulate the real-world. Desptite simple and restrictive – the sign of a good interface – a suprising number of situations can be squeezed into the MDP formalism.

### The MDP Entities
An MDP has two “entities”:
1. An agent: An application that is able to observe state and suggest an action. It also receives feedback to let it know whether the action was good, or not.
2. An environment: This is the place where the agent acts within. It accepts an action, which alterts its internal state, and produces a new observation of that state.
In nearly all of the examples that I have seen, these two entities are implemented independently. The agent is often an RL algorithm (although not always) and the environment is either real life or a simulation.

### The MDP Interface
The agent and the environment interact through an interface. You have some control over what goes into that interface and a large amount of effort is typically spent improving the quality of the data that flows through it. You need representations of:
1. State: This is the observation of the environment. You often get to choose what to “show” to the agent. There is a compromise between simplifying the state to speed up learning and preventing overfitting, but often it pays to include as much as you can.
2. Action: Your agent must suggest an action. This mutates the environment in some way. So called “options” or “null-actions” allow you to do nothing, if that’s what you want to do.
3. Reward: You use the reward to fine-tune your action choices.
<!-- #endregion -->

<!-- #region id="GICm97ENDDaS" -->
### Creating a “GridWorld” Environment
To make it easy to understand, I’m going to show you how to create a simulation of a simple grid-based “world”. Many real-life implementations begin with a simulation of the real world, because it’s much easier to iterate and improve your design with a software stub of real-life.

The goal of this environment is to define a “world” in which a “robot” can move. The so-called-world is actually a series of cells inside a 2-dimensional box. The agent can move north, east, south, or west which moves the robot between the cells. The goal of the environment is to reach a goal. There is a reward of -1 for every step, to promote reaching the goal as fast as possible.

### Imports and Definitions
First let me import a few libraries (to enable the autocompletion in later cells) and define a few important definitions. The first is the defacto definition of a “point” object, with x and y coordinates and the second is a direction enumeration. These are use to define the position of the agent in the environment and the direction of movement for an action, respectively. Note that I’m assuming that east moves in a positive x direction and north moves in a positive y direction.
<!-- #endregion -->

```python id="xQaJTkNDDH4T"
from collections import defaultdict, namedtuple
from enum import Enum
from typing import Tuple, List
import random
from IPython.display import clear_output

Point = namedtuple('Point', ['x', 'y'])
class Direction(Enum):
  NORTH = "⬆"
  EAST = "⮕"
  SOUTH = "⬇"
  WEST = "⬅"

  @classmethod
  def values(self):
    return [v for v in self]    
```

<!-- #region id="xR3eHNU1DKOv" -->
### The Environment Class
Next I create a Python class that represents the environment. The first function in the class is the initialisation function in which we can specify the width and height of the environment.

After that I define a helper parameter which encodes the possible actions and then I reset the state of the environment with a reset function.
<!-- #endregion -->

```python id="3_kimiSaDMJG"
class SimpleGridWorld(object):
  def __init__(self, width: int = 5, height: int = 5, debug: bool = False):
    self.width = width
    self.height = height
    self.debug = debug
    self.action_space = [d for d in Direction]
    self.reset()
```

<!-- #region id="QIZz8x5YDPIS" -->
### The Reset Function
Many environments have an implicit “reset”, whereby the environment’s state is moved away from the goal state. In this implementation I reset the environment back to the (0, 0) position, but this isn’t strictly necessary. Many real-life environments reset randomly or have no reset at all.

Here I also set the goal, which is located in the south-eastern corner of the environment.
<!-- #endregion -->

```python id="qVKRPX4sDRb6"
class SimpleGridWorld(SimpleGridWorld):
  def reset(self):
    self.cur_pos = Point(x=0, y=(self.height - 1))
    self.goal = Point(x=(self.width - 1), y=0)
    # If debug, print state
    if self.debug:
      print(self)
    return self.cur_pos, 0, False
```

<!-- #region id="s8EGo29RDTMv" -->
### Taking a Step
Recall that the MDP interface three key components: the state, the action, and the reward. The environment’s step function accepts an action, then produces a new state and reward.

The large amount of code is a consequence of the direction implementation. You can refactor this to use fewer lines of code with some clever indexing. However, I think this level of verbosity helps explain what is going on. In essence, every direction moves the current position by one square. You can see the code incrementing or decrementing the x or y coordinates.

The second part of the function is testing to see if the agent is at the goal. If it is, then it signals that it is at a terminal state.
<!-- #endregion -->

```python id="PRUUslizDU4T"
class SimpleGridWorld(SimpleGridWorld):
  def step(self, action: Direction):
    # Depending on the action, mutate the environment state
    if action == Direction.NORTH:
      self.cur_pos = Point(self.cur_pos.x, self.cur_pos.y + 1)
    elif action == Direction.EAST:
      self.cur_pos = Point(self.cur_pos.x + 1, self.cur_pos.y)
    elif action == Direction.SOUTH:
      self.cur_pos = Point(self.cur_pos.x, self.cur_pos.y - 1)
    elif action == Direction.WEST:
      self.cur_pos = Point(self.cur_pos.x - 1, self.cur_pos.y)
    # Check if out of bounds
    if self.cur_pos.x >= self.width:
      self.cur_pos = Point(self.width - 1, self.cur_pos.y)
    if self.cur_pos.y >= self.height:
      self.cur_pos = Point(self.width, self.cur_pos.y - 1)
    if self.cur_pos.x < 0:
      self.cur_pos = Point(0, self.cur_pos.y)
    if self.cur_pos.y < 0:
      self.cur_pos = Point(self.cur_pos.x, 0)

    # If at goal, terminate
    is_terminal = self.cur_pos == self.goal

    # Constant -1 reward to promote speed-to-goal
    reward = -1

    # If debug, print state
    if self.debug:
      print(self)

    return self.cur_pos, reward, is_terminal
```

<!-- #region id="KCsHpNT1DXwn" -->
### Visualisation
And finally, like all of data science, it is vitally important that you are able to visualise the behaviour and performance of your agent. The first step in this process is being able to visualise the agent within the environment. The next function does this by printing a textual grid, with an x at the agent’s location, a o at the goal, an @ if the agent is on top of the goal, and a _ otherwise.
<!-- #endregion -->

```python id="798-WEEqDZXB"
class SimpleGridWorld(SimpleGridWorld):
  def __repr__(self):
    res = ""
    for y in reversed(range(self.height)):
      for x in range(self.width):
        if self.goal.x == x and self.goal.y == y:
          if self.cur_pos.x == x and self.cur_pos.y == y:
            res += "@"
          else:
            res += "o"
          continue
        if self.cur_pos.x == x and self.cur_pos.y == y:
          res += "x"
        else:
          res += "_"
      res += "\n"
    return res
```

<!-- #region id="0-jjnxPLDazR" -->
### Running the Environment
To run the environment you need to instantiate the class, call reset to move the agent back to the start, then perform a series of actions to move the agent. For now let me move it manually, to make sure it is working, visualising the agent at each step. I also print the result of the step (the new state, reward, and terminal flag) for completeness.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7zHF_idwDcK0" executionInfo={"status": "ok", "timestamp": 1634454072406, "user_tz": -330, "elapsed": 527, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="002a746e-8c19-4bed-da3a-06e016dccb23"
s = SimpleGridWorld(debug=True)
print("☝ This shows a simple visualisation of the environment state.\n")
s.step(Direction.SOUTH)
print(s.step(Direction.SOUTH), "⬅ This displays the state and reward from the environment 𝐀𝐅𝐓𝐄𝐑 moving.\n")
s.step(Direction.SOUTH)
s.step(Direction.SOUTH)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)
```

<!-- #region id="Sw1MEM3RDNog" -->
### Key Takeaways
There are a few key lessons that you should commit to memory:
- The state is an observation of the environment, which contains everything outside of the agent. For example, the agent’s current position within the environment. In real world applications this could be the time of the day, the weather, data from a video camera, literally anything.
- The reward fully specifies the optimal solution to the problem. In real life this might be profit or the number of new customers.
- Every action mutates the state of the environment. This may or may not be observable.
<!-- #endregion -->

<!-- #region id="vYDcvsPpDolZ" -->
## A Reinforcement Learning Solution to the MDP: Implementing the Monte Carlo RL Algorithm
Rather confusingly, RL, like ML, is meant both as a technique and a collection of algorithms. Exactly when an algorithm becomes an RL algorithms is up for debate, but it is generally accepted that there has to be multiple steps (otherwise it would just be a bandit problem) and it attempts to quantify the value of being in a particular state.

An algorithm called the cross-entropy method is an algorithm that attempts to stumble accross the goal. However, once it has then it replicates the same movements again to reach the goal. This is not stricly learning, it is memoising, so it is not an RL algorithm. However, you shouldn’t discount it, because it is a very good and simple baseline. It can easily complete very sophisticated tasks if you give it enough time.

Instead, let me introduce a slight variation to this algorithm called the Monte Carlo (MC) method. This lies at the heart of all modern RL algorithms so it is a great way to start.

In short – you can read more about this in Chapter 2 of the book – MC methods attempt to randomly sample states and judge their value. Once you have sampled the states enough times then you can derive a strategy that follows the path of the next best value.

Let’s give it a try.

### Generating trajectories
Monte Carlo techniques operate by sampling the environment. In general, the idea is that if you can sample the environment enough times, you can begin to build a picture of the output, given any input. We can use this idea in RL. If we capture enough trajectories, where a trajectory is one full pass through an environment, then we can see which states are advantagous.

To begin, I will create a class that is capable of generating trajectories. Here I pass in the environment, then in the run function I repeatedly step in the environment using a random action. I store each step in a list and return it to the user.
<!-- #endregion -->

```python id="xYg_2AJCERm4"
class MonteCarloGeneration(object):
  def __init__(self, env: object, max_steps: int = 1000, debug: bool = False):
    self.env = env
    self.max_steps = max_steps
    self.debug = debug

  def run(self) -> List:
    buffer = []
    n_steps = 0 # Keep track of the number of steps so I can bail out if it takes too long
    state, _, _ = self.env.reset() # Reset environment back to start
    terminal = False
    while not terminal: # Run until terminal state
      action = random.choice(self.env.action_space) # Random action. Try replacing this with Direction.EAST
      next_state, reward, terminal = self.env.step(action) # Take action in environment
      buffer.append((state, action, reward)) # Store the result
      state = next_state # Ready for the next step
      n_steps += 1
      if n_steps >= self.max_steps:
        if self.debug:
          print("Terminated early due to large number of steps")
        terminal = True # Bail out if we've been working for too long
    return buffer
```

<!-- #region id="fNmoXK-kEUGj" -->
### Visualising Trajectories
As before, it’s vitally important to visualise as much as possible, to gain an intuition into your problem. A simple first step is to view the agent’s movement and trajectory. Here I severely limit the amount of exploration to save reams of output. Depending on your random seed you will see the agent stumbling around.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nNQVWC_cEVp5" executionInfo={"status": "ok", "timestamp": 1634454305574, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="186f0cb0-ac02-45a1-c68b-64502da8cb9c"
env = SimpleGridWorld(debug=True) # Instantiate the environment
generator = MonteCarloGeneration(env=env, max_steps=10, debug=True) # Instantiate the generation
trajectory = generator.run() # Generate a trajectory
print([t[1].value for t in trajectory]) # Print chosen actions
print(f"total reward: {sum([t[2] for t in trajectory])}") # Print final reward
```

<!-- #region id="DUGDMw5SEXUK" -->
### Quantifying Value
There’s an important quanity called the action value function. In summary, it is a measure of the value of taking a particular action, given all the experience. In other words, you can look at the previous trajectories, find out which of them lead to the highest values and look to use them again. See Chapter 2 in the book for more details.

To generate an estimate of this value, generate a full trajectory, then look at how far away the agent is from the terminal states at all steps.

So this means we need a class to generate a full trajectory, from start to termination. That code is below. First I create a new class that accepts the generator from before; I’ll use this later to generate the full trajectory.

Then I create two fields to retain the experience observed by the agent. The first is recording the expected value at each state. This is the effectively the distance to the goal. The second is recording the number of times the agent has visited that state.

Then I create a helper function to return a key for the dictionary (a.k.a. map) and an action value function to calculate the value of taking each action in each state. This is simply the average value over all visits.
<!-- #endregion -->

```python id="8Zfe2TbZEa55"
class MonteCarloExperiment(object):
  def __init__(self, generator: MonteCarloGeneration):
    self.generator = generator
    self.values = defaultdict(float)
    self.counts = defaultdict(float)

  def _to_key(self, state, action):
    return (state, action)
  
  def action_value(self, state, action) -> float:
    key = self._to_key(state, action)
    if self.counts[key] > 0:
      return self.values[key] / self.counts[key]
    else:
      return 0.0
```

<!-- #region id="3YWhRUwEEcU2" -->
Next I create a function to store this data after generating a full trajectory. There are several important parts of this function.

The first is that I’m using reversed trajectories. I.e. I’m starting from the end and working backwards.

The second is that I’m averaging the expected return over all visits. So this is reporting the value of an action, on average.
<!-- #endregion -->

```python id="KCjJ9_koEd5b"
class MonteCarloExperiment(MonteCarloExperiment):
  def run_episode(self) -> None:
    trajectory = self.generator.run() # Generate a trajectory
    episode_reward = 0
    for i, t in enumerate(reversed(trajectory)): # Starting from the terminal state
      state, action, reward = t
      key = self._to_key(state, action)
      episode_reward += reward  # Add the reward to the buffer
      self.values[key] += episode_reward # And add this to the value of this action
      self.counts[key] += 1 # Increment counter
```

<!-- #region id="6GIw7j8-EfIF" -->
### Running the Trajectory Generation
Let’s test this by setting some expectations. We’re reporting the value of taking an action on average. So on average, you would expect the value of taking the EAST action when next to the terminal state would be -1, because it’s right there, it’s a single step and therefore a single reward of -1 to get to the terminal state.

However, other directions will not be -1, because the agent will continue to stumble around.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bk4FJyUFEiiA" executionInfo={"status": "ok", "timestamp": 1634454359179, "user_tz": -330, "elapsed": 464, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9df51a7a-63f4-47d5-9887-f4904e9888af"
env = SimpleGridWorld(debug=False) # Instantiate the environment - set the debug to true to see the actual movemen of the agent.
generator = MonteCarloGeneration(env=env, debug=True) # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(4):
  agent.run_episode()
  print(f"Run {i}: ", [agent.action_value(Point(3,0), d) for d in env.action_space])
```

<!-- #region id="0FISlaVNEkeC" -->
So you can see from above that yes, when choosing east from the point to the west of the terminal state the expected return is -1. But notice that the agent (probably) did not observe that result straight away, because it takes time to randomly select it. (Run it a few more times to see what happens, you’ll see random changes)

### Visualising the State Value Function
The state value function is the average expected return for all actions. In general, you should see that the value increases the closer you get to the goal. But because of the random movement, especially far away from the goal, there will be a lot of noise.

Below I create a helper function to plot this.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sd-3lp8TEntv" executionInfo={"status": "ok", "timestamp": 1634454378682, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b00bb11e-5851-4b93-eb47-62cd54fbcd61"
def state_value_2d(env, agent):
    res = ""
    for y in reversed(range(env.height)):
      for x in range(env.width):
        if env.goal.x == x and env.goal.y == y:
          res += "   @  "
        else:
          state_value = sum([agent.action_value(Point(x,y), d) for d in env.action_space]) / len(env.action_space)
          res += f'{state_value:6.2f}'
        res += " | "
      res += "\n"
    return res

print(state_value_2d(env, agent))
```

<!-- #region id="hDnWZSgdEpMu" -->
### Generating Optimal Policies
A policy is a set of rules that an agent should follow. It is a strategy that works for that particular environment. You can now generate thousands of trajectories and track the expected value over time.

With enough averaging, the expected values should present a clear picture of what the optimal policy is. See if you can see what it is?

In the code below I’m instantiating all the previous code and then generating 1000 episodes. Then I print out the state value function for every position.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LzcCLRIJEuzL" executionInfo={"status": "ok", "timestamp": 1634454419548, "user_tz": -330, "elapsed": 10505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8c658baa-cc91-4b76-9644-a35c2af073fe"
env = SimpleGridWorld() # Instantiate the environment
generator = MonteCarloGeneration(env=env) # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(1000):
  clear_output(wait=True)
  agent.run_episode()
  print(f"Iteration: {i}")
  # print([agent.action_value(Point(0,4), d) for d in env.action_space]) # Uncomment this line to see the actual values for a particular state
  print(state_value_2d(env, agent), flush=True)
  # time.sleep(0.1) # Uncomment this line if you want to see every episode
```

<!-- #region id="eo1YIk46Ewuu" -->
### Plotting the Optimal Policy
That’s right! The optimal policy is to choose the action that picks the highest expected return. In other words, you want to move the agent towards regions of higher reward.

Let me create another helper function to visualise where the maximal actions are pointing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WllgY3FNEzq9" executionInfo={"status": "ok", "timestamp": 1634454426586, "user_tz": -330, "elapsed": 590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="428ddb2d-8165-45a0-8a64-96e3f14c9620"
def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])

def next_best_value_2d(env, agent):
    res = ""
    for y in reversed(range(env.height)):
      for x in range(env.width):
        if env.goal.x == x and env.goal.y == y:
          res += "@"
        else:
          # Find the action that has the highest value
          loc = argmax([agent.action_value(Point(x,y), d) for d in env.action_space])
          res += f'{env.action_space[loc].value}'
        res += " | "
      res += "\n"
    return res

print(next_best_value_2d(env, agent))
```

<!-- #region id="wsNqDn1aE062" -->
And there you have it. A policy. The above image spells out what the agent should do in each state. It should move towards regions of higher value. You can see (in general) that the arrows are all pointing towards the goal, as if by magic.

For the arrows that are not, that is a more interesting story. The problem is that the agent is still entirely random at this point. It’s stumbling around until it reachest the goal. The agent started in the top left, so on average, it takes a lot of stumbling to find the goal.

Therefore, for the points at the top left, furthest away from the goal, the agent will probably take many more random steps before it reachest the goal. In essence, it doesn’t matter which way the agent goes. It will still take a long time to get there.

Subsequent Monte Carlo algorithms fix this by updating the action value function every episode and using this knowledge to choose the action. So latter iterations of the agent are far more guided and intelligent.

One Final Run
To wrap this up, let me run the whole thing one more time. I will plot the state value function and the policy for all iteration steps. Watch how it changes over time. Add a sleep to slow it down to see it changing on each step.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mMxEaHIPE3U5" executionInfo={"status": "ok", "timestamp": 1634454457566, "user_tz": -330, "elapsed": 12120, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="38e304a8-df52-4ad3-a48a-8f84a8af35ce"
env = SimpleGridWorld() # Instantiate the environment
generator = MonteCarloGeneration(env=env) # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(1000):
  clear_output(wait=True)
  agent.run_episode()
  print(f"Iteration: {i}")
  print(state_value_2d(env, agent))
  print(next_best_value_2d(env, agent), flush=True)
  # time.sleep(0.1) # Uncomment this line if you want to see the iterations
```

<!-- #region id="p3RF8vjvE5qN" -->
I appreciate that this might be the first time that you have encountered Monte Carlo (MC) techniques or RL. So I have intentionally made this notebook as simple and free of libraries as possible, to gain experience at the coal-face.

This obviously means that the actual algorithm isn’t that intelligent. For example, MC techniques usually go through two phases, policy evaluation, where full trajectories are captured, then policy improvement, where a new policy is derived. This helps stabilise and speed up learning, beacuse you are learning on every episode.

But I’m getting ahead of myself. I really do encourage you to play around with this code and tinker with the results. Here are some things that you can try:
- Increase or decrease the size of the grid.
- Add other terminating states
- Change the reward to a different value
- Change the reward to produce 0 reward per step, and a positive reward for the terminating state
- Add a terminating state with a massive negative reward, to simulate a cliff
- Add a hole in the middle
- Add a wall
- See if you can add the code to use the policy derived by the agent
<!-- #endregion -->
