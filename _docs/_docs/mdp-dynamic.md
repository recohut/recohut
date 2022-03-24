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

<!-- #region id="-GOkUXTxSlg_" -->
# MDP with Dynamic Programming in PyTorch
<!-- #endregion -->

<!-- #region id="ABuxj3b_GUKZ" -->
We will evaluate and solve MDPs using dynamic programming (DP). It is worth to note that the Model-based methods such as DP have some drawbacks. They require the environment to be fully known, including the transition matrix and reward matrix. They also have limited scalability, especially for environments with plenty of states.
<!-- #endregion -->

<!-- #region id="-7jlwto6lFuE" -->
## Simple MDP
<!-- #endregion -->

```python id="b3WdqQ8eg9_J"
import torch
```

<!-- #region id="6Ze7whFAaPcP" -->
Our MDP has 3 state (sleep, study and play games), and 2 actions (word, slack). The 3 * 2 * 3 transition matrix T(s, a, s') is as follows:
<!-- #endregion -->

```python id="b9i16FoNgnEz"
T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]]
                 )
```

<!-- #region id="bPzN8YsXh-1b" -->
This means, for example, that when taking the a1 slack action from state s0 study, there is a 60% chance that it will become s1 sleep (maybe getting tired ) and a 30% chance that it will become s2 play games (maybe wanting to relax ), and that there is a 10% chance of keeping on studying (maybe a true workaholic ).
<!-- #endregion -->

```python id="nnzLIGijiGKM"
R = torch.tensor([1., 0, -1.])

gamma = 0.5

action = 0
```

<!-- #region id="xMc9Vb4niFli" -->
We define the reward function as [+1, 0, -1] for three states, to compensate for the hard work. Obviously, the optimal policy, in this case, is choosing a0 work for each step (keep on studying – no pain no gain, right?). Also, we choose 0.5 as the discount factor, to begin with.
<!-- #endregion -->

<!-- #region id="J-_KHL7Di-rz" -->
In this oversimplified study-sleep-game process, the optimal policy, that is, the policy that achieves the highest total reward, is choosing action a0 in all steps. However, it won't be that straightforward in most cases. Also, the actions taken in individual steps won't necessarily be the same. They are usually dependent on states. So, we will have to solve an MDP by finding the optimal policy in real-world cases.

The value function of a policy measures how good it is for an agent to be in each state, given the policy being followed. The greater the value, the better the state.
<!-- #endregion -->

<!-- #region id="WvlFKoqyiPR5" -->
We calculate the value, V, of the optimal policy using the matrix inversion method in the following function:
<!-- #endregion -->

```python id="4G1OnUo_h_TU"
def cal_value_matrix_inversion(gamma, trans_matrix, rewards):
    inv = torch.inverse(torch.eye(rewards.shape[0]) - gamma * trans_matrix)
    V = torch.mm(inv, rewards.reshape(-1, 1))
    return V
```

<!-- #region id="d83__pvujBqg" -->
Above, we calculated the value, V, of the optimal policy using matrix inversion. According to the Bellman Equation, the relationship between the value at step t+1 and that at step t can be expressed as follows:

$$V_{t+1} = R + \gamma*T*V_t$$

When the value converges, which means $V_{t+1} = V_t$, we can derive the value, $V$, as follows:

$$V = R + \gamma*T*V \\ V = (I-\gamma*T)^{-1}*R$$

Here, $I$ is the identity matrix with 1s on the main diagonal.

One advantage of solving an MDP with matrix inversion is that you always get an exact answer. But the downside is its scalability. As we need to compute the inversion of an m * m matrix (where m is the number of possible states), the computation will become costly if there is a large number of states.
<!-- #endregion -->

<!-- #region id="nvzs94x7irK5" -->
We feed all variables we have to the function, including the transition probabilities associated with action a0:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ISBceu4mirhS" executionInfo={"status": "ok", "timestamp": 1634646812886, "user_tz": -330, "elapsed": 392, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d39c1c3-df37-40f6-f055-4d22d07f9d30"
trans_matrix = T[:, action]
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))
```

<!-- #region id="P54H5V4WkT83" -->
We decide to experiment with different values for the discount factor. Let's start with 0, which means we only care about the immediate reward:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vIkafD13iuSX" executionInfo={"status": "ok", "timestamp": 1634647230619, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="faa0b338-f04c-47b5-a33c-d9d70ca93a87"
gamma = 0
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))
```

<!-- #region id="-PB1rtKKkX6B" -->
This is consistent with the reward function as we only look at the reward received in the next move.

As the discount factor increases toward 1, future rewards are considered. Let's take a look at $\gamma$=0.99:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-W95zCZlkP9y" executionInfo={"status": "ok", "timestamp": 1634647268981, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c92da1cb-eff7-4133-bd8b-899675385a88"
gamma = 0.99
V = cal_value_matrix_inversion(gamma, trans_matrix, R)
print("The value function under the optimal policy is:\n{}".format(V))
```

<!-- #region id="J3LASh7fkdif" -->
## Performing policy evaluation

Policy evaluation is an iterative algorithm. It starts with arbitrary policy values and then iteratively updates the values based on the Bellman expectation equation until they converge. In each iteration, the value of a policy, π, for a state, s, is updated as follows:

$$\mathcal{V}_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{\pi}(s'))$$

There are two ways to terminate an iterative updating process. One is by setting a fixed number of iterations, such as 1,000 and 10,000, which might be difficult to control sometimes. Another one involves specifying a threshold (usually 0.0001, 0.00001, or something similar) and terminating the process only if the values of all states change to an extent that is lower than the threshold specified.

Next, we will perform policy evaluation on the study-sleep-game process under the optimal policy and a random policy.
<!-- #endregion -->

```python id="J5X__AXVmTg3"
import torch

T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]]
                 )

R = torch.tensor([1., 0, -1.])

gamma = .5

threshold = 0.0001

policy_optimal = torch.tensor([[1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0]])
```

<!-- #region id="82aZCVjjmiXE" -->
Develop a policy evaluation function that takes in a policy, transition matrix, rewards, discount factor, and a threshold and computes the value function:
<!-- #endregion -->

```python id="hDRzEJXfmdTn"
def policy_evaluation(policy, trans_matrix, rewards, gamma, threshold):
    """
    Perform policy evaluation
    @param policy: policy matrix containing actions and their probability in each state
    @param trans_matrix: transformation matrix
    @param rewards: rewards for each state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy for all possible states
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (R[state] + gamma * torch.dot(trans_matrix[state, action], V))
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V
```

<!-- #region id="aKrlPoaTnIb-" -->
The policy evaluation function does the following tasks:

- Initializes the policy values as all zeros.
- Updates the values based on the Bellman expectation equation.
- Computes the maximal change of the values across all states.
- If the maximal change is greater than the threshold, it keeps updating the values. Otherwise, it terminates the evaluation process and returns the latest values.
<!-- #endregion -->

<!-- #region id="Fa2ngU4_mrny" -->
Now let's plug in the optimal policy and all other variables:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2RtZcWH8mk57" executionInfo={"status": "ok", "timestamp": 1634647862512, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f31ba1c5-efcf-4bf1-cfee-0cc4389fc6c7"
V = policy_evaluation(policy_optimal, T, R, gamma, threshold)
print("The value function under the optimal policy is:\n{}".format(V))
```

<!-- #region id="VMJGAyvxm2ON" -->
This is almost the same as what we got using matrix inversion.
<!-- #endregion -->

<!-- #region id="JtZFYNFim4Om" -->
We now experiment with another policy, a random policy where actions are picked with the same probabilities:
<!-- #endregion -->

<!-- #region id="O1Nrti84m7H0" -->
Plug in the random policy and all other variables:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Iv_O6-C8mqs1" executionInfo={"status": "ok", "timestamp": 1634647911370, "user_tz": -330, "elapsed": 402, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2015944c-a983-4acd-d7d9-a955c7ec19ea"
policy_random = torch.tensor([[0.5, 0.5],
                              [0.5, 0.5],
                              [0.5, 0.5]])

V = policy_evaluation(policy_random, T, R, gamma, threshold)
print("The value function under the random policy is:\n{}".format(V))
```

<!-- #region id="ST4iZiCanWX7" -->
We have just seen how effective it is to compute the value of a policy using policy evaluation. It is a simple convergent iterative approach, in the dynamic programming family, or to be more specific, approximate dynamic programming. It starts with random guesses as to the values and then iteratively updates them according to the Bellman expectation equation until they converge.

Since policy evaluation uses iterative approximation, its result might not be exactly the same as the result of the matrix inversion method, which uses exact computation. In fact, we don't really need the value function to be that precise. Also, it can solve the curses of dimensionality problem, which can result in scaling up the computation to thousands of millions of states. Therefore, we usually prefer policy evaluation over the other.

One more thing to remember is that policy evaluation is used to predict how great a we will get from a given policy; it is not used for control problems.
<!-- #endregion -->

<!-- #region id="CHhh_jRZoYJ4" -->
To take a closer look, we also plot the policy values over the whole evaluation process.

We first need to record the value for each iteration in the policy_evaluation function:
<!-- #endregion -->

```python id="paoyLsQCmy_I"
def policy_evaluation_history(policy, trans_matrix, rewards, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    V_his = [V]
    i = 0
    while True:
        V_temp = torch.zeros(n_state)
        i += 1
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (R[state] + gamma * torch.dot(trans_matrix[state, action], V))
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        V_his.append(V)
        if max_delta <= threshold:
            break
    return V, V_his
```

<!-- #region id="1VUAn4Azobgq" -->
Now we feed the policy_evaluation_history function with the optimal policy, a discount factor of 0.5, and other variables:
<!-- #endregion -->

```python id="EKcgyvQtoagt"
V, V_history = policy_evaluation_history(policy_optimal, T, R, gamma, threshold)
```

<!-- #region id="-ucfo0MWoeQS" -->
We then plot the resulting history of values using the following lines of code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="KmNPSsxZoWz9" executionInfo={"status": "ok", "timestamp": 1634648323920, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e7752285-b210-4c71-a469-21fe769144e0"
import matplotlib.pyplot as plt
s0, = plt.plot([v[0] for v in V_history])
s1, = plt.plot([v[1] for v in V_history])
s2, = plt.plot([v[2] for v in V_history])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy values')
plt.legend([s0, s1, s2],
           ["State s0",
            "State s1",
            "State s2"], loc="upper left")
plt.show()
```

<!-- #region id="PpF1CHnVofIZ" -->
Next, we run the same code but with two different discount factors, 0.2 and 0.99.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="srqgDy3VozPz" executionInfo={"status": "ok", "timestamp": 1634648579971, "user_tz": -330, "elapsed": 480, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="33934ada-1ccf-411d-9715-6e50ee39c6d4"
gamma = 0.2
V, V_history = policy_evaluation_history(policy_optimal, T, R, gamma, threshold)

s0, = plt.plot([v[0] for v in V_history])
s1, = plt.plot([v[1] for v in V_history])
s2, = plt.plot([v[2] for v in V_history])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy values')
plt.legend([s0, s1, s2],
           ["State s0",
            "State s1",
            "State s2"], loc="upper left")
plt.show()
```

<!-- #region id="PcCg-68hpLtY" -->
Comparing the plot with a discount factor of 0.5 with this one, we can see that the smaller the factor, the faster the policy values converge.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="Xk3srMSTo4we" executionInfo={"status": "ok", "timestamp": 1634648590946, "user_tz": -330, "elapsed": 530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62e3f1da-6797-4556-8afa-f515d4d31f03"
gamma = 0.99
V, V_history = policy_evaluation_history(policy_optimal, T, R, gamma, threshold)

s0, = plt.plot([v[0] for v in V_history])
s1, = plt.plot([v[1] for v in V_history])
s2, = plt.plot([v[2] for v in V_history])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy values')
plt.legend([s0, s1, s2],
           ["State s0",
            "State s1",
            "State s2"], loc="upper left")
plt.show()
```

<!-- #region id="lDrUfsYEo-A2" -->
By comparing the plot with a discount factor of 0.5 to the plot with a discount factor of 0.99, we can see that the larger the factor, the longer it takes for policy values to converge. The discount factor is a tradeoff between rewards now and rewards in the future.
<!-- #endregion -->

<!-- #region id="BRnlyH5Zo-oe" -->
## Simulating the FrozenLake environment

The optimal policies for the MDPs we have dealt with so far are pretty intuitive. However, it won't be that straightforward in most cases, such as the FrozenLake environment.

FrozenLake is a typical Gym environment with a discrete state space. It is about moving an agent from the starting location to the goal location in a grid world, and at the same time avoiding traps. The grid is either four by four (https://gym.openai.com/envs/FrozenLake-v0/) or eight by eight.

The grid is made up of the following four types of tiles:

- S: The starting location
- G: The goal location, which terminates an episode
- F: The frozen tile, which is a walkable location
- H: The hole location, which terminates an episode

There are four actions, obviously: moving left (0), moving down (1), moving right (2), and moving up (3). The reward is +1 if the agent successfully reaches the goal location, and 0 otherwise. Also, the observation space is represented in a 16-dimensional integer array, and there are 4 possible actions (which makes sense).

What is tricky in this environment is that, as the ice surface is slippery, the agent won't always move in the direction it intends. For example, it may move to the left or to the right when it intends to move down.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_tILkXrLrO0Z" executionInfo={"status": "ok", "timestamp": 1634649237256, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ca1c8dca-bc67-4af8-8f86-a53f134d8080"
import gym
import torch


env = gym.make("FrozenLake-v0")

n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gZe0KxrjrRFH" executionInfo={"status": "ok", "timestamp": 1634649237691, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f1d3dcee-ce89-4866-e9fa-74a4d0b8d890"
env.reset()

env.render()

new_state, reward, is_done, info = env.step(1)
env.render()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ROZiHbU1rbvY" executionInfo={"status": "ok", "timestamp": 1634649237693, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b0041bc-e1c3-4d23-db9c-d22dea26bfda"
print(new_state)
print(reward)
print(is_done)
print(info)
```

<!-- #region id="664KORsMryyY" -->
To demonstrate how difficult it is to walk on the frozen lake, implement a random policy and calculate the average total reward over 1,000 episodes. First, define a function that simulates a FrozenLake episode given a policy and returns the total reward (we know it is either 0 or 1):
<!-- #endregion -->

```python id="PD8OcDPBrUg2"
def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward
```

<!-- #region id="0wG2LD_Wr2uG" -->
Now run 1000 episodes, and a policy will be randomly generated and will be used in each episode:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OoqGKH0-r0uW" executionInfo={"status": "ok", "timestamp": 1634649239675, "user_tz": -330, "elapsed": 600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="55054ad6-e248-4cb1-d619-5f1a1cdfa6a4"
n_episode = 1000

total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    total_rewards.append(total_reward)

print('Average total reward under random policy: {}'.format(sum(total_rewards) / n_episode))
```

<!-- #region id="UVrozYIir_Xo" -->
This basically means there is only a 1.4% chance on average that the agent can reach the goal if we randomize the actions.

Next, we experiment with a random search policy. In the training phase, we randomly generate a bunch of policies and record the first one that reaches the goal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="V5G7hpCprh4v" executionInfo={"status": "ok", "timestamp": 1634649305003, "user_tz": -330, "elapsed": 407, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b3341093-67d6-48bf-e868-96806394e61a"
while True:
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    if total_reward == 1:
        best_policy = random_policy
        break

print(best_policy)
```

<!-- #region id="AHoesoJdsMQM" -->
Now run 1,000 episodes with the policy we just cherry-picked:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wbxUh-iusLU2" executionInfo={"status": "ok", "timestamp": 1634649316968, "user_tz": -330, "elapsed": 759, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="70280acb-0921-4cde-a44d-bfdba253133d"
total_rewards = []
for episode in range(n_episode):
    total_reward = run_episode(env, best_policy)
    total_rewards.append(total_reward)

print('Average total reward under random search policy: {}'.format(sum(total_rewards) / n_episode))
```

<!-- #region id="7c3oZD_psRpW" -->
Using the random search algorithm, the goal will be reached 11.1% of the time on average.
<!-- #endregion -->

<!-- #region id="nK9yBtAUsoX5" -->
We can look into the details of the FrozenLake environment, including the transformation matrix and rewards for each state and action, by using the P attribute. For example, for state 6, we can do the following:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wkc0_dpfrkt3" executionInfo={"status": "ok", "timestamp": 1634649153387, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6f656151-7c3b-44ee-ee81-8d87802a3a51"
print(env.env.P[6])
```

<!-- #region id="5s2A5ovDssZN" -->
This returns a dictionary with keys 0, 1, 2, and 3, representing four possible actions. The value is a list of movements after taking an action. The movement list is in the following format: (transformation probability, new state, reward received, is done). For instance, if the agent resides in state 6 and intends to take action 1 (down), there is a 33.33% chance that it will land in state 5, receiving a reward of 0 and terminating the episode; there is a 33.33% chance that it will land in state 10 and receive a reward of 0; and there is a 33.33% chance that it will land in state 7, receiving a reward of 0 and terminating the episode.

For state 11, we can do the following:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G06k-5Moro77" executionInfo={"status": "ok", "timestamp": 1634649156788, "user_tz": -330, "elapsed": 420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="853e9427-799e-437b-e6d9-93524e4caad0"
print(env.env.P[11])
```

<!-- #region id="5MExii5erqgP" -->
As stepping on a hole will terminate an episode, it won’t make any movement afterward.

Feel free to check out the other states.
<!-- #endregion -->

<!-- #region id="-22Gshuts9uP" -->
## Solving an MDP with a value iteration algorithm

An MDP is considered solved if its optimal policy is found. In this recipe, we will figure out the optimal policy for the FrozenLake environment using a value iteration algorithm.

The idea behind value iteration is quite similar to that of policy evaluation. It is also an iterative algorithm. It starts with arbitrary policy values and then iteratively updates the values based on the Bellman optimality equation until they converge. So in each iteration, instead of taking the expectation (average) of values across all actions, it picks the action that achieves the maximal policy values:

$$\mathcal{V}_*(s) = \max_{a \in \mathcal{A}} (\mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a {V}_{*}(s')))$$

Once the optimal values are computed, we can easily obtain the optimal policy accordingly.


<!-- #endregion -->

```python id="GS9CViLDxzIp"
import torch
import gym

env = gym.make('FrozenLake-v0')

gamma = 0.99

threshold = 0.0001
```

<!-- #region id="vAzlIdwMy0Wa" -->
Now define the function that computes optimal values based on the value iteration algorithm:
<!-- #endregion -->

```python id="S5Zr9a-CyzDp"
def value_iteration(env, gamma, threshold):
    """
    Solve a given environment with value iteration algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += trans_prob * (reward + gamma * V[new_state])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V
```

<!-- #region id="NfMYsWHUy6P2" -->
Plug in the environment, discount factor, and convergence threshold, then print the optimal values:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0cWU5VlazG76" executionInfo={"status": "ok", "timestamp": 1634651110730, "user_tz": -330, "elapsed": 1275, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6743f5f2-3ef9-4be3-aa38-43f0a5b9c8b8"
V_optimal = value_iteration(env, gamma, threshold)
print('Optimal values:\n{}'.format(V_optimal))
```

<!-- #region id="IZD6S_26zI_p" -->
Now that we have the optimal values, we develop the function that extracts the optimal policy out of them:
<!-- #endregion -->

<!-- #region id="JR1qleofzkl9" -->
We developed our value_iteration function according to the Bellamn Optimality Equation. We perform the following tasks:

- Initialize the policy values as all zeros.
- Update the values based on the Bellman optimality equation.
- Compute the maximal change of the values across all states.
- If the maximal change is greater than the threshold, we keep updating the values. Otherwise, we terminate the evaluation process and return the latest values as the optimal values.
<!-- #endregion -->

```python id="6_JLOGD2y2mM"
def extract_optimal_policy(env, V_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    @param env: OpenAI Gym environment
    @param V_optimal: optimal values
    @param gamma: discount factor
    @return: optimal policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V_optimal[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy
```

<!-- #region id="rTbNG5WezRqA" -->
Plug in the environment, discount factor, and optimal values, then print the optimal policy:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jKam6e14zLad" executionInfo={"status": "ok", "timestamp": 1634651139285, "user_tz": -330, "elapsed": 728, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2d19648-af5b-4cbf-b71e-847fa67136d5"
optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
print('Optimal policy:\n{}'.format(optimal_policy))
```

<!-- #region id="jeHbOJBCzVBV" -->
We want to gauge how good the optimal policy is. So, let's run 1,000 episodes with the optimal policy and check the average reward. Here, we will reuse the run_episode function we defined in the previous recipe:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FvVnF4r7zNpF" executionInfo={"status": "ok", "timestamp": 1634651177122, "user_tz": -330, "elapsed": 1247, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="10a2d9a2-885a-4ca0-d165-07a165f15ae2"
def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward


n_episode = 1000
total_rewards = []
for episode in range(n_episode):
    total_reward = run_episode(env, optimal_policy)
    total_rewards.append(total_reward)

print('Average total reward under the optimal policy: {}'.format(sum(total_rewards) / n_episode))
```

<!-- #region id="VENebdulzZFx" -->
Under the optimal policy, the agent will reach the goal 74% of the time, on average. This is the best we are able to get since the ice is slippery.
<!-- #endregion -->

<!-- #region id="iP1bn49fz2rR" -->
We obtained a success rate of 74% with a discount factor of 0.99. How does the discount factor affect the performance? Let's do some experiments with different factors, including 0, 0.2, 0.4, 0.6, 0.8, 0.99, and 1.:
<!-- #endregion -->

```python id="W8EjEnJhzWOk"
gammas = [0, 0.2, 0.4, 0.6, 0.8, .99, 1.]
n_episode = 10000
avg_reward_gamma = []
for gamma in gammas:
    V_optimal = value_iteration(env, gamma, threshold)
    optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
    total_rewards = []
    for episode in range(n_episode):
        total_reward = run_episode(env, optimal_policy)
        total_rewards.append(total_reward)
    avg_reward_gamma.append(sum(total_rewards) / n_episode)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="ngXBgImGz5eh" executionInfo={"status": "ok", "timestamp": 1634651354955, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d6309e2-3333-43d2-fb66-cf53d9ac0657"
import matplotlib.pyplot as plt
plt.plot(gammas, avg_reward_gamma)
plt.title('Success rate vs discount factor')
plt.xlabel('Discount factor')
plt.ylabel('Average success rate')
plt.show()
```

<!-- #region id="JPFXruXNz6qZ" -->
The result shows that the performance improves when there is an increase in the discount factor. This verifies the fact that a small discount factor values the reward now and a large discount factor values a better reward in the future.
<!-- #endregion -->

<!-- #region id="piuwgedv0cLq" -->
## Solving an MDP with a policy iteration algorithm

Another approach to solving an MDP is by using a policy iteration algorithm, which we will discuss in this section.

A policy iteration algorithm can be subdivided into two components: policy evaluation and policy improvement. It starts with an arbitrary policy. And in each iteration, it first computes the policy values given the latest policy, based on the Bellman expectation equation; it then extracts an improved policy out of the resulting policy values, based on the Bellman optimality equation. It iteratively evaluates the policy and generates an improved version until the policy doesn't change any more.

Let's develop a policy iteration algorithm and use it to solve the FrozenLake environment.
<!-- #endregion -->

```python id="RDHBh4Q70eQM"
import torch
import gym

env = gym.make('FrozenLake-v0')

gamma = 0.99

threshold = 0.0001
```

<!-- #region id="7m1Zc86H0lpe" -->
Now we define the policy_evaluation function that computes the values given a policy:
<!-- #endregion -->

```python id="jCqE4QoS0i88"
def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    @param env: OpenAI Gym environment
    @param policy: policy matrix containing actions and their probability in each state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V
```

<!-- #region id="e5BLuyzy0qaq" -->
Next, we develop the second main component of the policy iteration algorithm, the policy improvement part:
<!-- #endregion -->

```python id="yy47pyCF0msS"
def policy_improvement(env, V, gamma):
    """
    Obtain an improved policy based on the values
    @param env: OpenAI Gym environment
    @param V: policy values
    @param gamma: discount factor
    @return: the policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(v_actions)
    return policy
```

<!-- #region id="tPNRssJH0ubl" -->
This extracts an improved policy from the given policy values, based on the Bellman optimality equation.
<!-- #endregion -->

<!-- #region id="8oWm1TqA00ZP" -->
Now that we have both components ready, we develop the policy iteration algorithm as follows:
<!-- #endregion -->

```python id="yRhebfZw0rOZ"
def policy_iteration(env, gamma, threshold):
    """
    Solve a given environment with policy iteration algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: optimal values and the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved
```

<!-- #region id="slJXdRbu1DHK" -->
The policy_iteration function does the following tasks:

- Initializes a random policy.
- Computes the values of the policy with the policy evaluation algorithm.
- Obtains an improved policy based on the policy values.
- If the new policy is different from the old one, it updates the policy and runs another iteration. Otherwise, it terminates the iteration process and returns the policy values and the policy.
<!-- #endregion -->

<!-- #region id="7y07NgIO04Dj" -->
Plug in the environment, discount factor, and convergence threshold:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JDqzsWVN028H" executionInfo={"status": "ok", "timestamp": 1634651586842, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c15b11a8-487a-41e6-d776-a37572a060ff"
V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))
```

<!-- #region id="qAPnExbp07lR" -->
They are exactly the same as what we got using the value iteration algorithm.
<!-- #endregion -->

<!-- #region id="O9geM33E08As" -->
We have just solved the FrozenLake environment with a policy iteration algorithm. So, you may wonder when it is better to use policy iteration over value iteration and vice versa. There are basically three scenarios where one has the edge over the other:

- If there is a large number of actions, use policy iteration, as it can converge faster.
- If there is a small number of actions, use value iteration.
- If there is already a viable policy (obtained either by intuition or domain knowledge), use policy iteration.

Outside those scenarios, policy iteration and value iteration are generally comparable.

In the next section, we will apply each algorithm to solve the coin-flipping-gamble problem. We will see which algorithm converges faster.
<!-- #endregion -->

<!-- #region id="_AHKWHI7Dokr" -->
## Solving the coin-flipping gamble problem
Gambling on coin flipping should sound familiar to everyone. In each round of the game, the gambler can make a bet on whether a coin flip will show heads. If it turns out heads, the gambler will win the same amount they bet; otherwise, they will lose this amount. The game continues until the gambler loses (ends up with nothing) or wins (wins more than 100 dollars, let's say). Let's say the coin is unfair and it lands on heads 40% of the time. In order to maximize the chance of winning, how much should the gambler bet based on their current capital in each round? This will definitely be an interesting problem to solve.

If the coin lands on heads more than 50% of the time, there is nothing to discuss. The gambler can just keep betting one dollar each round and should win the game most of the time. If it is a fair coin, the gambler could bet one dollar each round and end up winning around 50% of the time. It gets tricky when the probability of heads is lower than 50%; the safe-bet strategy wouldn't work anymore. Nor would a random strategy, either. We need to resort to the reinforcement learning techniques we've learned in this tutorial to make smart bets.

Let's get started by formulating the coin-flipping gamble problem as an MDP. It is basically an undiscounted, episodic, and finite MDP with the following properties:

- The state is the gambler's capital in dollars. There are 101 states: 0, 1, 2, …, 98, 99, and 100+.
- The reward is 1 if the state 100+ is reached; otherwise, the reward is 0.
- The action is the possible amount the gambler bets in a round. Given state s, the possible actions include 1, 2, …, and min(s, 100 - s). For example, when the gambler has 60 dollars, they can bet any amount from 1 to 40. Any amount above 40 doesn't make any sense as it increases the loss and doesn't increase the chance of winning.
- The next state after taking an action depends on the probability of the coin coming up heads. Let's say it is 40%. So, the next state of state s after taking action a will be s+a by 40%, s-a by 60%.
- The process terminates at state 0 and state 100+.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XrsuDt_OD0g5" executionInfo={"status": "ok", "timestamp": 1634655572650, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="703f3121-2c41-4b40-a922-45247ad81831"
import torch

capital_max = 100
n_state = capital_max + 1
rewards = torch.zeros(n_state)
rewards[-1] = 1

print(rewards)

gamma = 1
threshold = 1e-10

head_prob = 0.4

env = {'capital_max': capital_max,
       'head_prob': head_prob,
       'rewards': rewards,
       'n_state': n_state}
```

<!-- #region id="peLsEGtdEJ0L" -->
Now we develop a function that computes optimal values based on the value iteration algorithm:
<!-- #endregion -->

```python id="f6MM2BBPD_2k"
def value_iteration(env, gamma, threshold):
    """
    Solve the coin flipping gamble problem with value iteration algorithm
    @param env: the coin flipping gamble environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the optimal policy for the given environment
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1, capital_max):
            v_actions = torch.zeros(min(state, capital_max - state) + 1)
            for action in range(1, min(state, capital_max - state) + 1):
                v_actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action])
                v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V
```

<!-- #region id="XQzNw5jLET8z" -->
We only need to compute the values for states 1 to 99, as the values for state 0 and state 100+ are 0. And given state s, the possible actions can be anything from 1 up to min(s, 100 - s). We should keep this in mind while computing the Bellman optimality equation.

Next, we develop a function that extracts the optimal policy based on the optimal values:
<!-- #endregion -->

```python id="9eSLvf6ZEL9-"
def extract_optimal_policy(env, V_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    @param env: the coin flipping gamble environment
    @param V_optimal: optimal values
    @param gamma: discount factor
    @return: optimal policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    optimal_policy = torch.zeros(capital_max).int()
    for state in range(1, capital_max):
        v_actions = torch.zeros(n_state)
        for action in range(1, min(state, capital_max - state) + 1):
            v_actions[action] += head_prob * (rewards[state + action] + gamma * V_optimal[state + action])
            v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V_optimal[state - action])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy
```

<!-- #region id="ODB3n4YxEcy8" -->
Finally, we can plug in the environment, discount factor, and convergence threshold to compute the optimal values and optimal policy after . Also, we time how long it takes to solve the gamble MDP with value iteration; we will compare this with the time it takes for policy iteration to complete:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q-NPledPEWd7" executionInfo={"status": "ok", "timestamp": 1634655666302, "user_tz": -330, "elapsed": 3514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f77c48d5-7ef0-4a50-80ba-f6f46ccd4bcc"
import time

start_time = time.time()
V_optimal = value_iteration(env, gamma, threshold)
optimal_policy = extract_optimal_policy(env, V_optimal, gamma)

print("It takes {:.3f}s to solve with value iteration".format(time.time() - start_time))
```

<!-- #region id="5BNyEFbVEjBm" -->
We solved the gamble problem with value iteration in 3.126 seconds.

Take a look at the optimal policy values and the optimal policy we got:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="H1Ux906VEeXa" executionInfo={"status": "ok", "timestamp": 1634655704228, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c8f9e304-1462-4a42-96b8-0654fcec74ed"
print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))
```

<!-- #region id="gCPv70lqEuaN" -->
We can plot the policy value versus state as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="VWctgeDPEnrd" executionInfo={"status": "ok", "timestamp": 1634655728709, "user_tz": -330, "elapsed": 611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0b1bf315-af47-4a77-d75d-3c1fd86b69b9"
import matplotlib.pyplot as plt

plt.plot(V_optimal[:100].numpy())
plt.title('Optimal policy values')
plt.xlabel('Capital')
plt.ylabel('Policy value')
plt.show()
```

<!-- #region id="USgjShgNEyCr" -->

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="GmMklxkjEsdU" executionInfo={"status": "ok", "timestamp": 1634655754449, "user_tz": -330, "elapsed": 596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d4e708cb-ed47-4080-893a-84f54c630a9c"
plt.bar(range(1, capital_max), optimal_policy[1:capital_max].numpy())
plt.title('Optimal policy')
plt.xlabel('Capital')
plt.ylabel('Optimal action')
plt.show()
```

<!-- #region id="Y7WTKF7LE3Nq" -->
Now that we've solved the gamble problem with value iteration, how about policy iteration? Let's see.

We start by developing the policy_evaluation function that computes the values given a policy:
<!-- #endregion -->

```python id="nkgjqb9bEzol"
def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    @param env: the coin flipping gamble environment
    @param policy: policy tensor containing actions taken for individual state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1, capital_max):
            action = policy[state].item()
            V_temp[state] += head_prob * (rewards[state + action] + gamma * V[state + action])
            V_temp[state] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V
```

<!-- #region id="CjYA9TSqE6zM" -->
Next, we develop another main component of the policy iteration algorithm, the policy improvement part:
<!-- #endregion -->

```python id="zku1qocsE4vH"
def policy_improvement(env, V, gamma):
    """
    Obtain an improved policy based on the values
    @param env: the coin flipping gamble environment
    @param V: policy values
    @param gamma: discount factor
    @return: the policy
    """
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    policy = torch.zeros(n_state).int()
    for state in range(1, capital_max):
        v_actions = torch.zeros(min(state, capital_max - state) + 1)
        for action in range(1, min(state, capital_max - state) + 1):
            v_actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action])
            v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V[state - action])
        policy[state] = torch.argmax(v_actions)
    return policy
```

<!-- #region id="2G7oR_MLE-IZ" -->
With both components ready, we can develop the main entry to the policy iteration algorithm as follows:
<!-- #endregion -->

```python id="p9_0d0hjE8Io"
def policy_iteration(env, gamma, threshold):
    """
    Solve the coin flipping gamble problem with policy iteration algorithm
    @param env: the coin flipping gamble environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: optimal values and the optimal policy for the given environment
    """
    n_state = env['n_state']
    policy = torch.zeros(n_state).int()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved
```

<!-- #region id="AQAcAzPuFD5T" -->
Finally, we plug in the environment, discount factor, and convergence threshold to compute the optimal values and the optimal policy. We record the time spent solving the MDP as well:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yuB6pLUJE_hN" executionInfo={"status": "ok", "timestamp": 1634655808965, "user_tz": -330, "elapsed": 3445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dfc739c4-fd15-4cb2-bcf4-3580bb591644"
start_time = time.time()
V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)

print("It takes {:.3f}s to solve with policy iteration".format(time.time() - start_time))
```

<!-- #region id="1VyHJzgiFGyV" -->
Check out the optimal values and optimal policy we just obtained:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZQQjUT8TFBJt" executionInfo={"status": "ok", "timestamp": 1634655833161, "user_tz": -330, "elapsed": 426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7584bbfa-37cb-4373-e57e-b155f989ceb8"
print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))
```

<!-- #region id="XgSjDQKMFef7" -->
The results from the two approaches, value iteration and policy iteration, are consistent.

We have solved the gamble problem by using value iteration and policy iteration. To deal with a reinforcement learning problem, one of the trickiest tasks is to formulate the process into an MDP. In our case, the policy is transformed from the current capital (states) to the new capital (new states) by betting certain stakes (actions). The optimal policy maximizes the probability of winning the game (+1 reward), and evaluates the probability of winning under the optimal policy.

Another interesting thing to note is how the transformation probabilities and new states are determined in the Bellman equation in our example. Taking action a in state s (having capital s and making a bet of 1 dollar) will have two possible outcomes:

- Moving to new state s+a, if the coin lands on heads. Hence, the transformation probability is equal to the probability of heads.
- Moving to new state s-a, if the coin lands on tails. Therefore, the transformation probability is equal to the probability of tails.
This is quite similar to the FrozenLake environment, where the agent lands on the intended tile only by a certain probability.

We also verified that policy iteration converges faster than value iteration in this case. This is because there are up to 50 possible actions, which is more than the 4 actions in FrozenLake. For MDPs with a large number of actions, solving with policy iteration is more efficient than doing so with value iteration.
<!-- #endregion -->

<!-- #region id="uzjauh-GFoBV" -->
You may want to know whether the optimal policy really works. Let's act like smart gamblers and play 10,000 episodes of the game. We are going to compare the optimal policy with two other strategies: conservative (betting one dollar each round) and random (betting a random amount):
<!-- #endregion -->

```python id="GDQtgCKuFIBp"
def run_random_episode(head, capital):
    while capital > 0:
        # print(capital)
        # bet = torch.randint(1, capital + 1, (1,)).item()
        bet = 1
        if torch.rand(1).item() < head:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    return 0


def run_optimal_episode(head, capital, optimal_policy):
    while capital > 0:
        bet = optimal_policy[capital].item()
        if torch.rand(1).item() < head:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    return 0


capital = 50

n_episode = 5000
total_rewards_random = []
total_rewards_opt = []

for episode in range(n_episode):
    total_reward_random = run_random_episode(0.48, capital)
    total_reward_opt = run_optimal_episode(0.4, capital, optimal_policy)
    total_rewards_random.append(total_reward_random)
    total_rewards_opt.append(total_reward_opt)
```

```python colab={"base_uri": "https://localhost:8080/"} id="oMUJZgqMF7ne" executionInfo={"status": "ok", "timestamp": 1634656050658, "user_tz": -330, "elapsed": 824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1be9ee9c-d667-4e72-a184-964e481af088"
print('Average total reward under the random policy: {}'.format(sum(total_rewards_random) / n_episode))
print('Average total reward under the optimal policy: {}'.format(sum(total_rewards_opt) / n_episode))
```

<!-- #region id="MhnDaUmtFsfL" -->
Our optimal policy is clearly the winner!
<!-- #endregion -->
