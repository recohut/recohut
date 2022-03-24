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

<!-- #region id="nG-P5CDvH7WG" -->
# Contextual Recommender with Vowpal Wabbit
<!-- #endregion -->

<!-- #region id="4aI3KjKLVsnn" -->
We will simulate the scenario of personalizing news content on a site, using CB, to users. The goal is to maximize user engagement quantified by measuring click through rate (CTR).

Let's recall that in a CB setting, a data point has four components,
- Context
- Action
- Probability of choosing action
- Reward/cost for chosen action

We will need to generate a context, get an action/decision for the given context and also simulate generating a reward. We have two website visitors: 'Tom' and 'Anna' Each of them may visit the website either in the morning or in the afternoon. The context is therefore (user, time_of_day). We have the option of recommending a variety of articles to Tom and Anna. Therefore, actions are the different choices of articles: "politics", "sports", "music", "food", "finance", "health", "cheese". The reward is whether they click on the article or not: 'click' or 'no click'.
<!-- #endregion -->

<!-- #region id="0X6NWsRBjT5f" -->
## Setup
<!-- #endregion -->

<!-- #region id="y0uVpw69jU4F" -->
### Installations
<!-- #endregion -->

```python id="YqOCKRRAjR6T"
!pip install vowpalwabbit
```

<!-- #region id="NtQy4OVzjWh3" -->
### Imports
<!-- #endregion -->

```python id="_m3opUkTjX7M"
from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import product
import numpy as np
import scipy
import scipy.stats as stats
```

<!-- #region id="yBX5eCYOOc_8" -->
## Simulate reward

In the real world, we will have to learn Tom and Anna's preferences for articles as we observe their interactions. Since this is a simulation, we will have to define Tom and Anna's preference profile. The reward that we provide to the learner will follow this preference profile. Our hope is to see if the learner can take better and better decisions as we see more samples which in turn means we are maximizing the reward.

We will also modify the reward function in a few different ways and see if the CB learner picks up the changes. We will compare the CTR with and without learning.

VW optimizes to minimize **cost which is negative of reward**. Therefore, we will always pass negative of reward as cost to VW.
<!-- #endregion -->

```python id="NOzs-MMwOc_9"
# VW tries to minimize loss/cost, therefore we will pass cost as -reward
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0
```

<!-- #region id="XQTwsgNMOc_-" -->
The reward function below specifies that Tom likes politics in the morning and music in the afternoon whereas Anna likes sports in the morning and politics in the afternoon. It looks dense but we are just simulating our hypothetical world in the format of the feedback the learner understands: cost. If the learner recommends an article that aligns with the reward function, we give a positive reward. In our simulated world this is a click.
<!-- #endregion -->

```python id="xpVW_5juOc__"
def get_cost(context,action):
    if context['user'] == "Tom":
        if context['time_of_day'] == "morning" and action == 'politics':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'music':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context['user'] == "Anna":
        if context['time_of_day'] == "morning" and action == 'sports':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'politics':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
```

<!-- #region id="Hh0MYzfeOdAB" -->

## Understanding VW format

There are some things we need to do to get our input into a format VW understands. This function handles converting from our context as a dictionary, list of articles and the cost if there is one into the text format VW understands.

<!-- #endregion -->

```python id="Wn_sVCzxOdAC"
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label = None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User user={} time_of_day={}\n".format(context["user"], context["time_of_day"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action article={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]
```

<!-- #region id="el20qcr1OdAD" -->
To understand what's going on here let's go through an example. Here, it's the morning and the user is Tom. There are four possible articles. So in the VW format there is one line that starts with shared, this is the shared context, followed by four lines each corresponding to an article.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_Yk4gaXWOdAE" executionInfo={"status": "ok", "timestamp": 1634747734282, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0bcbd55e-95d5-4c70-b4a2-cf3c11a63638"
context = {"user":"Tom","time_of_day":"morning"}
actions = ["politics", "sports", "music", "food"]

print(to_vw_example_format(context,actions))
```

<!-- #region id="XNOSmbNgOdAE" -->
## Getting a decision

When we call VW we get a _pmf_, [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function), as the output. Since we are incorporating exploration into our strategy, VW will give us a list of probabilities over the set of actions. This means that the probability at a given index in the list corresponds to the likelihood of picking that specific action. In order to arrive at a decision/action, we will have to sample from this list.

So, given a list `[0.7, 0.1, 0.1, 0.1]`, we would choose the first item with a 70% chance. `sample_custom_pmf` takes such a list and gives us the index it chose and what the probability of choosing that index was.
<!-- #endregion -->

```python id="TFpPKrU4OdAF"
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1/total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob
```

<!-- #region id="FGyYUeeVOdAG" -->
We have all of the information we need to choose an action for a specific user and context. To use VW to achieve this, we will do the following:

1. We convert our context and actions into the text format we need
2. We pass this example to vw and get the pmf out
3. Now, we sample this pmf to get what article we will end up showing
4. Finally we return the article chosen, and the probability of choosing it (we are going to need the probability when we learn form this example)
<!-- #endregion -->

```python id="W9GHrLJjOdAG"
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob
```

<!-- #region id="pRt4MxDnOdAH" -->


## Simulation set up

Now that we have done all of the setup work and know how to interface with VW, let's simulate the world of Tom and Anna. The scenario is they go to a website and are shown an article. Remember that the reward function allows us to define the worlds reaction to what VW recommends.


We will choose between Tom and Anna uniformly at random and also choose their time of visit uniformly at random. You can think of this as us tossing a coin to choose between Tom and Anna (Anna if heads and Tom if tails) and another coin toss for choosing time of day.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="fd_g9N5UOdAH" executionInfo={"status": "ok", "timestamp": 1634747751750, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9d5106ce-b0ff-4401-968c-46a99915693f"
users = ['Tom', 'Anna']
times_of_day = ['morning', 'afternoon']
actions = ["politics", "sports", "music", "food", "finance", "health", "camping"]

def choose_user(users):
    return random.choice(users)

def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)

# display preference matrix
def get_preference_matrix(cost_fun):
    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    df = expand_grid({'users':users, 'times_of_day': times_of_day, 'actions': actions})
    df['cost'] = df.apply(lambda r: cost_fun({'user': r[0], 'time_of_day': r[1]}, r[2]), axis=1)

    return df.pivot_table(index=['users', 'times_of_day'], 
            columns='actions', 
            values='cost')

get_preference_matrix(get_cost)
```

<!-- #region id="0Z4F7pRzOdAI" -->
We will instantiate a CB learner in VW and then simulate Tom and Anna's website visits `num_iterations` number of times. In each visit, we:

1. Decide between Tom and Anna
2. Decide time of day
3. Pass context i.e. (user, time of day) to learner to get action i.e. article recommendation and probability of choosing action
4. Receive reward i.e. see if user clicked or not. Remember that cost is just negative reward.
5. Format context, action, probability, reward in VW format
6. Learn from the example
    - VW _reduces_ a CB problem to a cost sensitive multiclass classification problem.

This is the same for every one of our simulations, so we define the process in the `run_simulation` function. The cost function must be supplied as this is essentially us simulating how the world works.

<!-- #endregion -->

```python id="7Vd-9I-KOdAI"
def run_simulation(vw, num_iterations, users, times_of_day, actions, cost_function, do_learn = True):
    cost_sum = 0.
    ctr = []

    for i in range(1, num_iterations+1):
        # 1. In each simulation choose a user
        user = choose_user(users)
        # 2. Choose time of day for a given user
        time_of_day = choose_time_of_day(times_of_day)

        # 3. Pass context to vw to get an action
        context = {'user': user, 'time_of_day': time_of_day}
        action, prob = get_action(vw, context, actions)

        # 4. Get cost of the action we chose
        cost = cost_function(context, action)
        cost_sum += cost

        if do_learn:
            # 5. Inform VW of what happened so we can learn from it
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
            # 6. Learn
            vw.learn(vw_format)

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        ctr.append(-1*cost_sum/i)

    return ctr
```

<!-- #region id="STiQn3KWOdAJ" -->

We want to be able to visualize what is occurring, so we are going to plot the click through rate over each iteration of the simulation. If VW is showing actions the get rewards the ctr will be higher. Below is a little utility function to make showing the plot easier.

<!-- #endregion -->

```python id="wJxpv72YOdAK"
def plot_ctr(num_iterations, ctr):
    plt.plot(range(1,num_iterations+1), ctr)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('ctr', fontsize=14)
    plt.ylim([0,1])
```

<!-- #region id="AVR_H4StOdAK" -->
## Scenario 1

We will use the first reward function `get_cost` and assume that Tom and Anna do not change their preferences over time and see what happens to user engagement as we learn. We will also see what happens when there is no learning. We will use the "no learning" case as our baseline to compare to.

### With learning

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="KiWbwzoXOdAL" executionInfo={"status": "ok", "timestamp": 1634747758919, "user_tz": -330, "elapsed": 1096, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ceed5a34-03a8-44c1-eac4-2831bec215f2"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost)

plot_ctr(num_iterations, ctr)
```

<!-- #region id="M-s1_N1fOdAL" -->
Aside: interactions

You'll notice in the arguments we supply to VW, **we include `-q UA`**. This is telling VW to create additional features which are the features in the (U)ser namespace and (A)ction namespaces multiplied together. This allows us to learn the interaction between when certain actions are good in certain times of days and for particular users. If we didn't do that, the learning wouldn't really work. We can see that in action below.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="voXxj6dwOdAM" executionInfo={"status": "ok", "timestamp": 1634747763631, "user_tz": -330, "elapsed": 1765, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22ddf753-545e-4238-fb6f-be63e5d4c8fc"
# Instantiate learner in VW but without -q
vw = pyvw.vw("--cb_explore_adf --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost)

plot_ctr(num_iterations, ctr)
```

<!-- #region id="7RpLurdyOdAN" -->

### Without learning
Let's do the same thing again (but with `-q`, but this time show the effect if we don't learn from what happens. The ctr never improves are we just hover around 0.2.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="-agneYrYOdAN" executionInfo={"status": "ok", "timestamp": 1634747765089, "user_tz": -330, "elapsed": 696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="046fe1b2-0801-4db6-d15c-ab9609b60b08"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost, do_learn=False)

plot_ctr(num_iterations, ctr)
```

<!-- #region id="I6ZyisDnOdAO" -->
## Scenario 2

In the real world people's preferences change over time. So now in the simulation we are going to incorporate two different cost functions, and swap over to the second one halfway through. Below is a a table of the new reward function we are going to use, `get_cost_1`:

### Tom

| | `get_cost` | `get_cost_new1` |
|:---|:---:|:---:|
| **Morning** | Politics | Politics |
| **Afternoon** | Music | Sports |

### Anna

| | `get_cost` | `get_cost_new1`  |
|:---|:---:|:---:|
| **Morning** | Sports | Sports |
| **Afternoon** | Politics | Sports |

This reward function is still working with actions that the learner has seen previously.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="5EY5x6FGOdAO" executionInfo={"status": "ok", "timestamp": 1634747766985, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="434a2190-9dc9-4eae-8e15-c22555ea2ec1"
def get_cost_new1(context,action):
    if context['user'] == "Tom":
        if context['time_of_day'] == "morning" and action == 'politics':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'sports':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context['user'] == "Anna":
        if context['time_of_day'] == "morning" and action == 'sports':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'sports':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
        
get_preference_matrix(get_cost_new1)
```

<!-- #region id="sap2wFagOdAP" -->
To make it easy to show the effect of the cost function changing we are going to modify the `run_simulation` function. It is a little less readable now, but it supports accepting a list of cost functions and it will operate over each cost function in turn. This is perfect for what we need.
<!-- #endregion -->

```python id="nVmE_UkFOdAP"
def run_simulation_multiple_cost_functions(vw, num_iterations, users, times_of_day, actions, cost_functions, do_learn = True):
    cost_sum = 0.
    ctr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
            # 1. in each simulation choose a user
            user = choose_user(users)
            # 2. choose time of day for a given user
            time_of_day = choose_time_of_day(times_of_day)

            # Construct context based on chosen user and time of day
            context = {'user': user, 'time_of_day': time_of_day}

            # 3. Use the get_action function we defined earlier
            action, prob = get_action(vw, context, actions)

            # 4. Get cost of the action we chose
            cost = cost_function(context, action)
            cost_sum += cost

            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
                # 6. Learn
                vw.learn(vw_format)

            # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
            ctr.append(-1*cost_sum/i)
        start_counter = end_counter
        end_counter = start_counter + num_iterations

```

```python id="0O--0JpdOdAQ"
def run_simulation_multiple_cost_functions(vw, num_iterations, users, times_of_day, actions, cost_functions, do_learn = True):
    cost_sum = 0.
    ctr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
            # 1. in each simulation choose a user
            user = choose_user(users)
            # 2. choose time of day for a given user
            time_of_day = choose_time_of_day(times_of_day)

            # Construct context based on chosen user and time of day
            context = {'user': user, 'time_of_day': time_of_day}

            # 3. Use the get_action function we defined earlier
            action, prob = get_action(vw, context, actions)

            # 4. Get cost of the action we chose
            cost = cost_function(context, action)
            cost_sum += cost

            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
                # 6. Learn
                vw.learn(vw_format)

            # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
            ctr.append(-1*cost_sum/i)
        start_counter = end_counter
        end_counter = start_counter + num_iterations

    return ctr
```

<!-- #region id="FqGuu9aiOdAQ" -->
### With learning
Let us now switch to the second reward function after a few samples (running the first reward function). Recall that this reward function changes the preferences of the web users but it is still working with the same action space as before. We should see the learner pick up these changes and optimize towards the new preferences.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="F1pAQ0ecOdAR" executionInfo={"status": "ok", "timestamp": 1634747775296, "user_tz": -330, "elapsed": 2310, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="645f59c4-a4e4-41a2-a3c4-d44bb46795ef"
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times_of_day, actions, cost_functions)

plot_ctr(total_iterations, ctr)
```

<!-- #region id="rv_kVQU_OdAR" -->
**Note:** The initial spike in CTR depends on the rewards received for the first few examples. When you run on your own, you may see something different initially because our simulator is designed to have randomness.

### Without learning
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="KPXXayj8OdAR" executionInfo={"status": "ok", "timestamp": 1634747776303, "user_tz": -330, "elapsed": 1016, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="256a29bd-17bd-4e94-ad02-aa79b525f293"
# Do not learn
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times_of_day, actions, cost_functions, do_learn=False)
plot_ctr(total_iterations, ctr)
```

<!-- #region id="O8zluZecOdAS" -->
## Scenario 3
In this scenario we are going to start rewarding actions that have never seen a reward previously when we change the cost function.

### Tom

| | `get_cost` | `get_cost_new2` |
|:---|:---:|:---:|
| **Morning** | Politics |  Politics|
| **Afternoon** | Music |   Food |

### Anna

| | `get_cost` | `get_cost_new2` |
|:---|:---:|:---:|
| **Morning** | Sports | Food|
| **Afternoon** | Politics |  Food |

<!-- #endregion -->

```python id="AzmkXtSOOdAS"
def get_cost_new2(context,action):
    if context['user'] == "Tom":
        if context['time_of_day'] == "morning" and action == 'politics':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'food':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context['user'] == "Anna":
        if context['time_of_day'] == "morning" and action == 'food':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'food':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
```

<!-- #region id="hyqhA3WFOdAS" -->

### With learning
Let us now switch to the third reward function after a few samples (running the first reward function). Recall that this reward function changes the preferences of the users and is working with a **different** action space than before. We should see the learner pick up these changes and optimize towards the new preferences

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="VIh8qkMcOdAT" executionInfo={"status": "ok", "timestamp": 1634747780582, "user_tz": -330, "elapsed": 1899, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9666899d-9a7c-4cb5-aad3-f0c2de5c8918"
# use first reward function initially and then switch to third reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new2]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times_of_day, actions, cost_functions)

plot_ctr(total_iterations, ctr)
```

<!-- #region id="a-PpzgP_OdAT" -->
### Without Learning
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="XuE1kVH6OdAU" executionInfo={"status": "ok", "timestamp": 1634747782889, "user_tz": -330, "elapsed": 830, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcc22ee6-f8e0-40f3-97e9-ec24e9ab7b43"
# Do not learn
# use first reward function initially and then switch to third reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new2]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times_of_day, actions, cost_functions, do_learn=False)

plot_ctr(total_iterations, ctr)
```

<!-- #region id="wDoMeJ8rOdAU" -->
This section aimed at showcasing a real world scenario where contextual bandit algorithms can be used. We were able to take a context and set of actions and learn what actions worked best for a given context. We saw that the learner was able to respond rapidly to changes in the world.  We showed that allowing the learner to interact with the world resulted in higher rewards than the no learning baseline. We worked with simplistic features. VW supports high dimensional sparse features, different exploration algorithms and policy evaluation approaches.
<!-- #endregion -->

<!-- #region id="CJk-yZDQjZMX" -->
## Contextual bandit with changing context

> Customizing the context and changing it midway to see how fast the agent can adapt to the new context and start recommending better products as per the context
<!-- #endregion -->

<!-- #region id="V9D8ePsikWAv" -->
### Setting the context
<!-- #endregion -->

<!-- #region id="-9UvUHeflfz5" -->
We have 3 users and 6 items. Context 1 is time of day - morning and evening. Context 2 is season - summer and winter.

Ground truth rules:

1. User 1 likes Item 1 in morning, and Item 6 in summer
2. User 2 likes Item 2 in winter, and Item 5 in summer morning
3. User 3 likes Item 2 in morning, Item 3 in evening, and item 4 in winter morning
<!-- #endregion -->

```python id="uP6Qy9upkX9x"
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0
```

```python id="lyLnQ1ERkZhh"
users = ['A','B','C']
items = ['Item1','Item2','Item3','Item4','Item5','Item6']
context1 = ['morning','evening']
context2 = ['summer','winter']

context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = 0

#user 1 likes Item 1 in morning, and Item 6 in summer
context.loc[(context.users=='A') & \
            (context.context1=='morning') & \
            (context['items']=='Item1'), \
            'reward'] = 1
context.loc[(context.users=='A') & \
            (context.context2=='summer') & \
            (context['items']=='Item6'), \
            'reward'] = 1

#user 2 likes Item 2 in winter, and Item 5 in summer morning
context.loc[(context.users=='B') & \
            (context.context2=='winter') & \
            (context['items']=='Item2'), \
            'reward'] = 1
context.loc[(context.users=='B') & \
            (context.context1=='morning') & \
            (context.context2=='summer') & \
            (context['items']=='Item5'), \
            'reward'] = 1


#user 3 likes Item 2 in morning, Item 3 in evening, and item 4 in winter morning
context.loc[(context.users=='C') & \
            (context.context1=='morning') & \
            (context['items']=='Item2'), \
            'reward'] = 1
context.loc[(context.users=='C') & \
            (context.context1=='evening') & \
            (context['items']=='Item3'), \
            'reward'] = 1
context.loc[(context.users=='C') & \
            (context.context1=='morning') & \
            (context.context2=='winter') & \
            (context['items']=='Item4'), \
            'reward'] = 1

context['cost'] = context['reward']*-1

contextdf = context.copy()
```

```python colab={"base_uri": "https://localhost:8080/"} id="gGiJWfhtkc9_" executionInfo={"status": "ok", "timestamp": 1634747969054, "user_tz": -330, "elapsed": 1929, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8740bf12-79d9-42d2-a96f-494395009429"
contextdf.cost.value_counts()
```

<!-- #region id="Jodzh1ofkl5X" -->
### Cost function util
<!-- #endregion -->

```python id="MSfJOXwMktkA"
def get_cost(context,action):
    return contextdf.loc[(contextdf['users']==context['user']) & \
            (contextdf.context1==context['context1']) & \
            (contextdf.context2==context['context2']) & \
            (contextdf['items']==action), \
            'cost'].values[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Yy89o4afkvYy" executionInfo={"status": "ok", "timestamp": 1634748013910, "user_tz": -330, "elapsed": 702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="36cc4b90-f560-4dcc-b41e-8b664437cec6"
get_cost({'user':'A','context1':'morning','context2':'summer'},'Item2')
```

<!-- #region id="apj4zt0_kw00" -->
### Vowpalwabbit format util
<!-- #endregion -->

```python id="xJ9laRJGkzSD"
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label = None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User users={} context1={} context2={}\n".format(context["user"], context["context1"], context["context2"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action items={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="BrASk6IPk007" executionInfo={"status": "ok", "timestamp": 1634748035616, "user_tz": -330, "elapsed": 693, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ab6dbbbd-b935-4e9a-846b-c572b5906279"
context = {"user":"A","context1":"morning","context2":"summer"}

print(to_vw_example_format(context,items))
```

```python id="ez2lyHNCk2u2"
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob
```

```python id="ZflJA3J0k8hP"
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob


def choose_user(users):
    return random.choice(users)

def choose_context1(context1):
    return random.choice(context1)

def choose_context2(context2):
    return random.choice(context2)


def run_simulation(vw, num_iterations, users, contexts1, contexts2, actions, cost_function, do_learn = True):
    cost_sum = 0.
    ctr = []

    for i in range(1, num_iterations+1):
        user = choose_user(users)
        context1 = choose_context1(contexts1)
        context2 = choose_context2(contexts2)

        context = {'user': user, 'context1': context1, 'context2': context2}
        # print(context)
        action, prob = get_action(vw, context, actions)
        # print(action, prob)

        cost = cost_function(context, action)
        # print(cost)
        cost_sum += cost

        if do_learn:
            # 5. Inform VW of what happened so we can learn from it
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
            # 6. Learn
            vw.learn(vw_format)
            # 7. Let VW know you're done with these objects
            vw.finish_example(vw_format)

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        ctr.append(-1*cost_sum/i)

    return ctr


def plot_ctr(num_iterations, ctr):
    plt.plot(range(1,num_iterations+1), ctr)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('ctr', fontsize=14)
    plt.ylim([0,1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="LFnJ5I2XlFR0" executionInfo={"status": "ok", "timestamp": 1634748123342, "user_tz": -330, "elapsed": 5744, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0e809d4d-f955-46af-aec6-d4ae11bb0e73"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost)

plot_ctr(num_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="UIhoP-rjlK0G" executionInfo={"status": "ok", "timestamp": 1634748136550, "user_tz": -330, "elapsed": 6353, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11dc0681-d65d-4e14-b8a9-56a48ec7c5ed"
# Instantiate learner in VW but without -q
vw = pyvw.vw("--cb_explore_adf --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost)

plot_ctr(num_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="11Z97qY4lN90" executionInfo={"status": "ok", "timestamp": 1634748145725, "user_tz": -330, "elapsed": 2205, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5999b5bc-b17a-4d65-e1b0-fb2017be1728"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost, do_learn=False)

plot_ctr(num_iterations, ctr)
```

<!-- #region id="RwaRHglwlow_" -->
### Changing the context

Updated ground truth rules:

1. User 1 likes Item 2 in morning, and Item 5 in summer
2. User 2 likes Item 2 in summer, and Item 5 in morning
3. User 3 likes Item 4 in morning, Item 3 in evening, and item 4 in winter evening
<!-- #endregion -->

```python id="y_PbP3ZMlQQR"
users = ['A','B','C']
items = ['Item1','Item2','Item3','Item4','Item5','Item6']
context1 = ['morning','evening']
context2 = ['summer','winter']

context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = 0

#user 1 likes Item 2 in morning, and Item 5 in summer
context.loc[(context.users=='A') & \
            (context.context1=='morning') & \
            (context['items']=='Item2'), \
            'reward'] = 1
context.loc[(context.users=='A') & \
            (context.context2=='summer') & \
            (context['items']=='Item5'), \
            'reward'] = 1

#user 2 likes Item 2 in summer, and Item 5 in morning
context.loc[(context.users=='B') & \
            (context.context2=='summer') & \
            (context['items']=='Item2'), \
            'reward'] = 1
context.loc[(context.users=='B') & \
            (context.context1=='morning') & \
            (context['items']=='Item5'), \
            'reward'] = 1


#user 3 likes Item 4 in morning, Item 3 in evening, and item 4 in winter evening
context.loc[(context.users=='C') & \
            (context.context1=='morning') & \
            (context['items']=='Item4'), \
            'reward'] = 1
context.loc[(context.users=='C') & \
            (context.context1=='evening') & \
            (context['items']=='Item3'), \
            'reward'] = 1
context.loc[(context.users=='C') & \
            (context.context1=='evening') & \
            (context.context2=='winter') & \
            (context['items']=='Item4'), \
            'reward'] = 1

context['cost'] = context['reward']*-1

contextdf_new = context.copy()

def get_cost_new1(context,action):
    return contextdf_new.loc[(contextdf_new['users']==context['user']) & \
            (contextdf_new.context1==context['context1']) & \
            (contextdf_new.context2==context['context2']) & \
            (contextdf_new['items']==action), \
            'cost'].values[0]
```

```python id="giMJGBc_lTPH"
def run_simulation_multiple_cost_functions(vw, num_iterations, users, contexts1, contexts2, actions, cost_functions, do_learn = True):
    cost_sum = 0.
    ctr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
          user = choose_user(users)
          context1 = choose_context1(contexts1)
          context2 = choose_context2(contexts2)

          context = {'user': user, 'context1': context1, 'context2': context2}
          
          action, prob = get_action(vw, context, actions)
          cost = cost_function(context, action)
          cost_sum += cost

          if do_learn:
              vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
              vw.learn(vw_format)

          ctr.append(-1*cost_sum/i)
        start_counter = end_counter
        end_counter = start_counter + num_iterations

    return ctr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="oer_Vp8SlUsu" executionInfo={"status": "ok", "timestamp": 1634748177369, "user_tz": -330, "elapsed": 12390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c77cba28-8ad5-40e7-8edb-6d22a46986ab"
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, context1, context2, items, cost_functions)

plot_ctr(total_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="p7LbRqzElWgc" executionInfo={"status": "ok", "timestamp": 1634748188707, "user_tz": -330, "elapsed": 11393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8fc30f58-d9ab-4eac-ab68-a02c050c0d1b"
# Do not learn
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, context1, context2, items, cost_functions, do_learn=False)
plot_ctr(total_iterations, ctr)
```

```python id="vNgiCkfGlYAi"
mapping_users = {
    'Alex':'usera',
    'Ben':'userb',
    'Cindy': 'userc'
}
    
mapping_context1 = {
    'Morning':'ctx11',
    'Evening':'ctx12',
}

mapping_context2 = {
    'Summer':'ctx21',
    'Winter':'ctx22'
}

mapping_items = {
    'Politics':'item1',
    'Economics':'item2',
    'Technology':'item3',
    'Movies':'item4',
    'Business':'item5',
    'History':'item6'
}
```

```python id="UTu524YinWig"
users = list(mapping_users.values())
items = list(mapping_items.values())
context1 = list(mapping_context1.values())
context2 = list(mapping_context2.values())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="Fj5L4k_UnZkr" executionInfo={"status": "ok", "timestamp": 1634748746746, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8802dc91-faa9-401c-db4d-1678e3c5b1fb"
context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context))
context['cost'] = context['reward']*-1
contextdf = context.copy()
contextdf
```

```python id="wZ66OoF_nb5c"
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User users={} context1={} context2={}\n".format(context["user"], context["context1"], context["context2"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action items={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]


def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob


def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob


def choose_user(users):
    return random.choice(users)


def choose_context1(context1):
    return random.choice(context1)

    
def choose_context2(context2):
    return random.choice(context2)
```

```python id="xIvHNcNmnmUa"
class VWCSimulation():
    def __init__(self, vw, ictxt, n=100000):
        self.vw = vw
        self.users = ictxt['users'].unique().tolist()
        self.contexts1 = ictxt['context1'].unique().tolist()
        self.contexts2 = ictxt['context2'].unique().tolist()
        self.actions = ictxt['items'].unique().tolist()
        self.contextdf = ictxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
        
    def get_cost(self, context, action):
        return self.contextdf.loc[(self.contextdf['users']==context['user']) & \
                (self.contextdf.context1==context['context1']) & \
                (self.contextdf.context2==context['context2']) & \
                (self.contextdf['items']==action), \
                'cost'].values[0]
    
    def update_context(self, new_ctxt):
        self.contextdf = new_ctxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
    
    def step(self):
        user = choose_user(self.users)
        context1 = choose_context1(self.contexts1)
        context2 = choose_context2(self.contexts2)
        context = {'user': user, 'context1': context1, 'context2': context2}
        action, prob = get_action(self.vw, context, self.actions)
        cost = self.get_cost(context, action)
        vw_format = self.vw.parse(to_vw_example_format(context, self.actions, (action, cost, prob)), pyvw.vw.lContextualBandit)
        self.vw.learn(vw_format)
        self.vw.finish_example(vw_format)
        return (context['user'], context['context1'], context['context2'], action, cost, prob)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R3--k-qNntVk" executionInfo={"status": "ok", "timestamp": 1634748792419, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b1bed0b-6042-480e-c1b5-a9ca7b6da1dc"
context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context),p=[0.8,0.2])
contextdf = context.copy()
contextdf.reward.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Jr39uZXOnvVV" executionInfo={"status": "ok", "timestamp": 1634748808175, "user_tz": -330, "elapsed": 559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="18ab736c-93b8-464c-970c-7958413cae4a"
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
vws = VWCSimulation(vw, contextdf)

vws.step()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="nrcPxEkynzPC" executionInfo={"status": "ok", "timestamp": 1634748831964, "user_tz": -330, "elapsed": 5874, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="73616137-7c78-4f3b-8818-ba55a91ea67c"
_temp = []
for i in range(5000):
    _temp.append(vws.step())

x = pd.DataFrame.from_records(_temp, columns=['user','context1','context2','item','cost','prob'])


xx = x.copy()
xx['ccost'] = xx['cost'].cumsum()
xx = xx.fillna(0)
xx = xx.rename_axis('iter').reset_index()
xx['ctr'] = -1*xx['ccost']/xx['iter']
xx.sample(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="W2wHOA4wn3rr" executionInfo={"status": "ok", "timestamp": 1634748834940, "user_tz": -330, "elapsed": 1025, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="30b46d13-41f2-4a8b-fcd2-957809c50871"
xx['ccost'].plot()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="x7A81yOWn5yF" executionInfo={"status": "ok", "timestamp": 1634748842396, "user_tz": -330, "elapsed": 706, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="173e5ecc-b7a7-4c06-b224-365ae10cde79"
xx['ctr'].plot()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="JtF5tbhOn7jf" executionInfo={"status": "ok", "timestamp": 1634748887359, "user_tz": -330, "elapsed": 7447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a531cd9b-56d9-4cc8-c804-80ff4df2d0cf"
tempdf1 = xx.copy()

context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = 0
X = context.copy()
X.loc[(X['users']=='usera')&(X['items']=='item1'),'reward']=1
X.loc[(X['users']=='userb')&(X['items']=='item2'),'reward']=1
X.loc[(X['users']=='userc')&(X['items']=='item3'),'reward']=1
X.reward.value_counts()

vws.update_context(X)

_temp = []
for i in range(5000):
    _temp.append(vws.step())

x = pd.DataFrame.from_records(_temp, columns=['user','context1','context2','item','cost','prob'])
xx = x.copy()
xx['ccost'] = xx['cost'].cumsum()
xx = xx.fillna(0)
xx = xx.rename_axis('iter').reset_index()
xx['ctr'] = -1*xx['ccost']/xx['iter']
xx.sample(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="m6Ej5cMHoE-O" executionInfo={"status": "ok", "timestamp": 1634748887362, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f3d2ce8f-a9ba-48c9-c264-37ab4ddd7948"
tempdf2 = tempdf1.append(xx, ignore_index=True)
tempdf2.sample(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="WluzTjjgoGgJ" executionInfo={"status": "ok", "timestamp": 1634748894913, "user_tz": -330, "elapsed": 55, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="766bf74f-6313-4f7c-eba3-1ef7709493b2"
tempdf2['ccost'].plot()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="264SEH2zoH1M" executionInfo={"status": "ok", "timestamp": 1634748915710, "user_tz": -330, "elapsed": 998, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e15f0fcd-49fb-49d2-d481-1d4dc24dd0b2"
tempdf2['ctr'].plot()
```

<!-- #region id="Kxcubne-oNic" -->
## Contextual bandit dash app

> Building a dash app of contextual bandit based recommender system
<!-- #endregion -->

<!-- #region id="pI_Y3hYnorHh" -->
The objective of this app is to apply the contextual bandit algorithms to recommendation problem under a simulated environment. The recommender agent is able to quickly adapt the changing behavior of users and change the recommendation strategy accordingly.

There are 3 users: Alex, Ben and Cindy. There are 6 news topics and 2 types of context. That means, Alex, Ben, and Cindy might prefer to read news reated to any of the 6 topics on morning/evening and weekday/weekends. Eg. Alex might prefer business related news on weekday mornings and entertainment related news on weekend evenings. And it is also possible that in future, Alex starts reading politics on weekday mornings. These situations reflect the real-world scenarios and the job of our contextual agent is to automatically detect these preferences and changes and recommend the items accordingly to maximize the reward like user satisfaction.

[https://www.youtube.com/watch?v=9t0-FZIWMRQ](https://www.youtube.com/watch?v=9t0-FZIWMRQ)

In the example, agent initialized with random preferences and starts recommending news to the users. We added 2 context: "Cindy prefers economy news on weekday mornings" and "Ben prefers weather news on weekday mornings" and starts rewarding agent for correctly recommending as per these preferences. At the moment, agent knows that Ben prefers business news and Cindy prefers history news. With time, agent started recommending weather news to Ben. Similar case we will see for Cindy and in fact, for all users.

Note: Interval is 1 sec but we are not seeing updates every second because we are looking at a particular context only: Weekday mornings but agent is recommending globally.

It is important to note that agent do not know the ground truth. It just taking action and receiving reward and the objective is to estimate this ground truth preferences.
<!-- #endregion -->

```python id="YVP4fTkR3duW"
!pip install -q dash dash-html-components dash-core-components dash_bootstrap_components jupyter-dash
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wq_hAT_t5AGN" executionInfo={"status": "ok", "timestamp": 1634749093998, "user_tz": -330, "elapsed": 831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62d27a73-ab02-48b1-c246-d6fb66bcfbd8"
!mkdir -p assets
!wget -q --show-progress -O assets/image.jpg https://moodle.com/wp-content/uploads/2020/04/Moodle_General_news.png
```

```python id="pcv55-nd3WWy"
import dash
from dash import dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import plotly.graph_objs as go
import plotly.express as px

from vowpalwabbit import pyvw

import numpy as np
import pandas as pd
import itertools
import pathlib
from copy import deepcopy
from itertools import product
import scipy
import scipy.stats as stats
import random
```

```python id="8mLLJQJn5Wws"
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User users={} context1={} context2={}\n".format(context["user"], context["context1"], context["context2"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action items={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]


def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob


def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob


def choose_user(users):
    return random.choice(users)


def choose_context1(context1):
    return random.choice(context1)


def choose_context2(context2):
    return random.choice(context2)
    

class VWCSimulation():
    def __init__(self, vw, ictxt):
        self.vw = vw
        self.users = ictxt['users'].unique().tolist()
        self.contexts1 = ictxt['context1'].unique().tolist()
        self.contexts2 = ictxt['context2'].unique().tolist()
        self.actions = ictxt['items'].unique().tolist()
        self.contextdf = ictxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
        
    def get_cost(self, context, action):
        return self.contextdf.loc[(self.contextdf['users']==context['user']) & \
                (self.contextdf.context1==context['context1']) & \
                (self.contextdf.context2==context['context2']) & \
                (self.contextdf['items']==action), \
                'cost'].values[0]
    
    def update_context(self, new_ctxt):
        self.contextdf = new_ctxt.copy()
        self.contextdf['cost'] = self.contextdf['reward']*-1
    
    def step(self):
        user = choose_user(self.users)
        context1 = choose_context1(self.contexts1)
        context2 = choose_context2(self.contexts2)
        context = {'user': user, 'context1': context1, 'context2': context2}
        action, prob = get_action(self.vw, context, self.actions)
        cost = self.get_cost(context, action)
        vw_format = self.vw.parse(to_vw_example_format(context, self.actions, (action, cost, prob)), pyvw.vw.lContextualBandit)
        self.vw.learn(vw_format)
        self.vw.finish_example(vw_format)
        return (context['user'], context['context1'], context['context2'], action, cost, prob)
```

```python id="IEd6IKmv3NLq"
app = JupyterDash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

def generate_input_cards(preference='Random'):
    card_content = [
    dbc.CardImg(src="assets/image.jpg", top=True),
    dbc.CardBody([html.P(preference, className="card-title")])
    ]
    card = dbc.Card(card_content, color="primary", outline=True)
    return dbc.Col([card], width={"size": 2})

pref_grid = []

mapping_users = {
    'Alex':'usera',
    'Ben':'userb',
    'Cindy': 'userc'
}
    
mapping_context1 = {
    'Morning':'ctx11',
    'Evening':'ctx12',
}

mapping_context2 = {
    'Weekday':'ctx21',
    'Weekend':'ctx22'
}

mapping_items = {
    'Politics':'item1',
    'Economics':'item2',
    'Technology':'item3',
    'Weather':'item4',
    'Business':'item5',
    'History':'item6'
}

mapping_users_reverse = {v:k for k,v in mapping_users.items()}
mapping_context1_reverse = {v:k for k,v in mapping_context1.items()}
mapping_context2_reverse = {v:k for k,v in mapping_context2.items()}
mapping_items_reverse = {v:k for k,v in mapping_items.items()}

users = list(mapping_users.values())
items = list(mapping_items.values())
context1 = list(mapping_context1.values())
context2 = list(mapping_context2.values())

context = pd.DataFrame(list(product(users, context1, context2, items)),
                       columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context),p=[0.8,0.2])

vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
vws = VWCSimulation(vw, context)
last_update = vws.step()

contextdf = context.copy()
countDF = contextdf.copy()
countDF['prob'] = 0

def generate_input_boxes():
    dropdown_users = dcc.Dropdown(
        id='ddown_users',
        options=[{"label":k, "value":v} for k,v in mapping_users.items()],
        clearable=False,
        value="usera",
        className="m-1",
    )
    dropdown_context1 = dcc.Dropdown(
        id='ddown_ctx1',
        options=[{"label":k, "value":v} for k,v in mapping_context1.items()],
        clearable=False,
        value="ctx11",
        className="m-1",
    )
    dropdown_context2 = dcc.Dropdown(
        id='ddown_ctx2',
        options=[{"label":k, "value":v} for k,v in mapping_context2.items()],
        clearable=False,
        value="ctx21",
        className="m-1",
    )
    dropdown_items = dcc.Dropdown(
        id='ddown_items',
        options=[{"label":k, "value":v} for k,v in mapping_items.items()],
        clearable=False,
        value="item1",
        className="m-1",
    )
    return html.Div(
        [
            dropdown_users,
            dropdown_context1,
            dropdown_context2,
            dropdown_items,
        ],
        style={"display": "flex", "flex-direction": "column"},
    )

def generate_context_boxes():
    dropdown_outcontext1 = dcc.Dropdown(
        id='ddown_outctx1',
        options=[{"label":k, "value":v} for k,v in mapping_context1.items()],
        clearable=False,
        value="ctx11",
        className="m-1",
    )
    dropdown_outcontext2 = dcc.Dropdown(
        id='ddown_outctx2',
        options=[{"label":k, "value":v} for k,v in mapping_context2.items()],
        clearable=False,
        value="ctx21",
        className="m-1",
    )
    return html.Div(
        [
            dropdown_outcontext1,
            dropdown_outcontext2
        ],
        style={"display": "flex", "flex-direction": "column"},
    )

app.layout = html.Div([
        generate_input_boxes(),
        dbc.Button("Register your Preference", color="primary", className="m-1", 
                   id='pref-button'),
        html.Div(id='pref-grid'),
        dbc.Button("Clear the context", color="secondary", 
                   className="m-1", id='clr-button'),
        dbc.Button("Start rewarding Agent for these Preferences", color="success", 
                   className="m-1", id='updt-button'),
        generate_context_boxes(),
        dcc.Interval(
            id='interval-component',
            interval=100, # in milliseconds
            n_intervals=0),
        html.Div(id='placeholder'),
        html.Div(id='placeholder2'),

])

@app.callback(
    Output("pref-grid", "children"),
    Input("pref-button", "n_clicks"),   
    Input("clr-button", "n_clicks"),
    State('ddown_users', 'value'),
    State('ddown_items', 'value'),
    State('ddown_ctx1', 'value'), 
    State('ddown_ctx2', 'value'),
)
def update_pref_grid(nclick_pref, nclick_clr, pref_user, pref_item, pref_ctx1, pref_ctx2):
    global pref_grid
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "pref-button" in changed_id:
        global contextdf
        card_text = '{} prefers {} related news in {} {}s'.format(mapping_users_reverse[pref_user],
                                                  mapping_items_reverse[pref_item],
                                                  mapping_context2_reverse[pref_ctx2],
                                                  mapping_context1_reverse[pref_ctx1])
        
        contextdf.loc[(contextdf.users==pref_user) & \
            (contextdf.context1==pref_ctx1) & \
            (contextdf.context2==pref_ctx2), \
            'reward'] = 0
        contextdf.loc[(contextdf.users==pref_user) & \
            (contextdf.context1==pref_ctx1) & \
            (contextdf.context2==pref_ctx2) & \
            (contextdf['items']==pref_item), \
            'reward'] = 1
        pref_grid.append(generate_input_cards(card_text))
        return dbc.Row(children=pref_grid,
                      style={'max-width': '100%',
                             'display': 'flex',
                             'align-items': 'center',
                             'padding': '2rem 5rem',
                             'overflow': 'auto',
                             'height': 'fit-content',
                             'flex-direction': 'row',
                            })
    elif "clr-button" in changed_id:
        pref_grid = []
        return dbc.Row(children=pref_grid)

@app.callback(
    Output("placeholder2", "children"),
    Input("updt-button", "n_clicks")
)
def update_context(nclick):
    if nclick:
        global vws
        global contextdf
        vws.update_context(contextdf)
    return ''


@app.callback(
    Output("placeholder", "children"),
    Input('interval-component', 'n_intervals'),
    Input('ddown_outctx1', 'value'), 
    Input('ddown_outctx2', 'value'),
)
def update_metrics(n, octx1, octx2):
    global countDF
    countDF = countDF.append(pd.Series(vws.step(),countDF.columns),ignore_index=True)
    _x = countDF.copy()
    _x = _x[(_x.context1==octx1) & (_x.context2==octx2)]
    _x['reward']*=-1
    pv = pd.pivot_table(_x, index=['users'], columns=["items"], values=['reward'], aggfunc=sum, fill_value=0)
    pv.index = [mapping_users_reverse[x] for x in pv.index]
    pv.columns = pv.columns.droplevel(0)
    pv = pv.rename_axis('User').reset_index().rename_axis(None, axis=1).set_index('User').T.reset_index()
    pv['index'] = pv['index'].map(mapping_items_reverse)
    pv = pv.rename(columns={"index": "Preferences"})
    out = html.Div([
        dbc.Table.from_dataframe(pv, striped=True, bordered=True, hover=True, responsive=True)
    ])
    return out
```

```python colab={"base_uri": "https://localhost:8080/", "height": 671} id="Cm6QckQy4BeO" executionInfo={"status": "ok", "timestamp": 1634749335337, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8c426484-039f-4403-8546-b017fa284e4f"
app.run_server(mode='inline', port=8081)
```

```python id="Lg14PTYT4CuG"
# !kill -9 $(lsof -t -i:8081) # command to kill the dash once done
```
