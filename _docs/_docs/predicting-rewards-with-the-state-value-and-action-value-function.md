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

<!-- #region id="3UFLh0qq8AAs" -->
# Predicting rewards with the state-value and action-value function
<!-- #endregion -->

<!-- #region id="eNUJVfOs8anT" -->
## Setup
<!-- #endregion -->

```python id="riJzjMYl8bpI"
!pip install -q numpy==1.19.2
```

```python id="M7cAHYFO8dYW"
import numpy as np
```

<!-- #region id="Dl_lis7o8RJE" -->
## The environment - a simple grid world
<!-- #endregion -->

<!-- #region id="m5KD8ygN-I-_" -->
To demonstrate, let me use a rediculously simple grid-based environment. This consists of 5 squares, with a cliff on the left-hand side and a goal position on the right. Both are terminating states.
<!-- #endregion -->

```python id="lv_jA3lj-JVI"
starting_position = 1 # The starting position
cliff_position = 0 # The cliff position
end_position = 5 # The terminating state position
reward_goal_state = 5 # Reward for reaching goal
reward_cliff = 0 # Reward for falling off cliff

def reward(current_position) -> int:
    if current_position <= cliff_position:
        return reward_cliff
    if current_position >= end_position:
        return reward_goal_state
    return 0

def is_terminating(current_position) -> bool:
    if current_position <= cliff_position:
        return True
    if current_position >= end_position:
        return True
    return False
```

<!-- #region id="_XiXx656-PQa" -->
## The Agent
In this simple environment, let us define an agent with a simple random strategy. On every step, the agent randomly decides to go left or right.
<!-- #endregion -->

```python id="sTE6A2sT-Qbl"
def strategy() -> int:
    if np.random.random() >= 0.5:
        return 1 # Right
    else:
        return -1 # Left
```

<!-- #region id="Nr6g4kix8FrC" -->
## State-value function
<!-- #endregion -->

<!-- #region id="gmxivI4C8I7B" -->
The state-value function is a view of the expected return with respect to each state.

$V_{\pi}(s) \doteq \mathbb{E}_{\pi}[ G \vert s] = \mathbb{E}_{\pi}\bigg[ \sum^{T}_{k=0} \gamma^k r_{k} \vert s \bigg]$

You could estimate the expectation in a few ways, but the simplest is to simply average over all of the observed rewards. To investigate how this equation works, you can perform the calculation on a simple environment that is easy to validate.
<!-- #endregion -->

<!-- #region id="9qC3wJ_d-RkT" -->
### Experiment
Let’s iterate thousands of times and record what happens.

The key to understanding this algorithm is to truly understand that we want to know the return, from a state, on average. Say that out loud. The return, from a state, on average.

You’re not rewarding on every step. You’re only rewarding when the agent reaches a terminal state. But when you are in the middle of this environment, for example, there is a 50/50 chance of ending up at the goal. You might also end up off the cliff. So in this instance, the expected value of that state is half way between the maximum reward, 5, and the minimum reward, 0.

Note that in this implementation 0 and 5 are terminating states, only 1-4 are valid states, so given four states the mid-point is actually in-between states. This will become clear when you inspect the values later.

If you were to implement this, you need to keep track of which states have been visited and the eventual, final reward. So the implementation below has a simple buffer to keep track of positions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eamzOvzk-igj" executionInfo={"status": "ok", "timestamp": 1634452780091, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bf0df559-7c1b-4602-a07b-7b0045f5be5d"
np.random.seed(42)

# Global buffers to perform averaging later
value_sum = np.zeros(end_position + 1)
n_hits = np.zeros(end_position + 1)

n_iter = 10
for i in range(n_iter):
    position_history = [] # A log of positions in this episode
    current_position = starting_position # Reset
    while True:
        # Append position to log
        position_history.append(current_position)

        if is_terminating(current_position):
            break
        
        # Update current position according to strategy
        current_position += strategy()

    # Now the episode has finished, what was the reward?
    current_reward = reward(current_position)
    
    # Now add the reward to the buffers that allow you to calculate the average
    for pos in position_history:
        value_sum[pos] += current_reward
        n_hits[pos] += 1
        
    # Now calculate the average for this episode and print
    expected_return = ', '.join(f'{q:.2f}' for q in value_sum / n_hits)
    print("[{}] Average reward: [{}]".format(i, expected_return))
```

<!-- #region id="060aqlcZ-i2N" -->
I’ve capped the number of episodes to 10 so you can see the evolution of the value estimates. But I encourage you to run this yourself and change this to 10,000.

Note that I have chosen a random seed that stubles right on the second episode. In general you would expect it to reach the goal every 1-in-5. Try changing the seed to see what happens?

You can see that with each episode the value estimate gets closer and closer to the true value (which should be integer 0 to 5). For example, when you are in the state next to the goal (the box next to the end) the you would expect that the agent should stumble towards the goal more often than not. Indeed, 4 out of 5 times it does, which means that the average return is 5 (the goal) multipled by 4/5.
<!-- #endregion -->

<!-- #region id="kLf_ghKd_A68" -->
### Discussion

It’s worth going through this code line by line. It truly is fundamental. This algorithm allows you to estimate the value of being in each state, purely by experiencing those states.

The key is to remember that the goal is to predict the value FROM each state. The goal is always to reach a point where you are maximizing rewards, so your agent needs to know how far away from optimal it is. This distinction can be tricky to get your head around, but once you have it’s hard to think any other way.

You can even use this in your life. Imagine you wanted to achieve some goal. All you have to do is predict the expected return from being in each new state. For example, say you wanted to get into reinforcement learning. You could go back to university, read the books, or go and watch TV. Each of these have value, but with different costs and lengths of time. The expected return of watching TV is probably very low. The expected return of reading the book is high, but doesn’t guarantee a job. Going back to university still doesn’t guarantee a job, but it might make it easier to get past HR, but it takes years to achieve. Making decisions in this way is known as using the expected value framework and is useful throughout business and life.
<!-- #endregion -->

<!-- #region id="3On6ji1T_Wua" -->
## Action-value function
<!-- #endregion -->

<!-- #region id="WtcqoqhV_j8g" -->
The action-value function is a view of the expected return with respect to a given state and action choice. The action represents an extra dimension over and above the state-value function. The premise is the same, but this time you need to iterate over all actions as well as all states. The equation is also similar, with the extra addition of an action, a:

$ Q_{\pi}(s, a) \doteq \mathbb{E}_{\pi}[ G \vert s, a ] = \mathbb{E}_{\pi}\bigg[ \sum^{T}_{k=0} \gamma^k r_{k} \vert s, a \bigg] $
<!-- #endregion -->

<!-- #region id="Shqg2yyM_qPR" -->
### Experiment

First off, there’s far more exploration to do, because we’re not only iterating over states, but also actions. You’ll need to run this for longer before it converges.

Also, we’re going to have to store both the states and the actions in the buffer.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PeLxEK_NAMj4" executionInfo={"status": "ok", "timestamp": 1634453219318, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="07b65b33-4885-46b3-ae00-94b2c7e5531a"
np.random.seed(42)

# Global buffers to perform averaging later
# Second dimension is the actions
value_sum = np.zeros((end_position + 1, 2))
n_hits = np.zeros((end_position + 1, 2))

# A helper function to map the actions to valid buffer indices
def action_value_mapping(x): return 0 if x == -1 else 1


n_iter = 10
for i in range(n_iter):
    position_history = [] # A log of positions in this episode
    current_position = starting_position # Reset
    current_action = strategy()
    while True:
        # Append position to log
        position_history.append((current_position, current_action))

        if is_terminating(current_position):
            break
        
        # Update current position according to strategy
        current_position += strategy()

    # Now the episode has finished, what was the reward?
    current_reward = reward(current_position)
    
    # Now add the reward to the buffers that allow you to calculate the average
    for pos, act in position_history:
        value_sum[pos, action_value_mapping(act)] += current_reward
        n_hits[pos, action_value_mapping(act)] += 1
        
    # Now calculate the average for this episode and print
    expect_return_0 = ', '.join(
        f'{q:.2f}' for q in value_sum[:, 0] / n_hits[:, 0])
    expect_return_1 = ', '.join(
        f'{q:.2f}' for q in value_sum[:, 1] / n_hits[:, 1])
    print("[{}] Average reward: [{} ; {}]".format(
        i, expect_return_0, expect_return_1))
```

<!-- #region id="OHQFiVP8AOLP" -->
### Discussion

I’ve capped the number of episodes to 10 again. I encourage you to run this yourself and change this to 10,000.

You can see that the results are similar, except for the fact that one of the actions (the action heading towards the cliff) is always zero, as you might expect.

So what’s the point of this if the result is basically the same? The key is that enumerating the action simplifies latter algorithms. With the state-value function your agent has to figure out how to get to better states in order to maximise the expected return. However, if you have the actions at hand, you can simply pick the next best action!
<!-- #endregion -->
