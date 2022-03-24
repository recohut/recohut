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
    language: python
    name: python3
---

<!-- #region id="uIdN0EwSPPEG" -->
# Simulating a news personalization scenario using Contextual Bandits
> A tutorial to understand the process of building a contextual recommender system using vowpalwabbit

- toc: true
- badges: true
- comments: true
- categories: [reinforcement, contextual]
- image: 
<!-- #endregion -->

<!-- #region id="8u_FxzhGOc_s" -->
In the Contextual Bandit(CB) introduction tutorial, we learnt about CB and different CB algorithms. In this tutorial we will simulate the scenario of personalizing news content on a site, using CB, to users. The goal is to maximize user engagement quantified by measuring click through rate (CTR).

Let's recall that in a CB setting, a data point has four components,

- Context
- Action
- Probability of choosing action
- Reward/cost for chosen action

In our simulator, we will need to generate a context, get an action/decision for the given context and also simulate generating a reward.

In our simulator, our goal is to maximize reward (click through rate/CTR) or minimize loss (-CTR)

- We have two website visitors: 'Tom' and 'Anna'
- Each of them may visit the website either in the morning or in the afternoon

The **context** is therefore (user, time_of_day)

We have the option of recommending a variety of articles to Tom and Anna. Therefore, **actions** are the different choices of articles: "politics", "sports", "music", "food", "finance", "health", "cheese"

The **reward** is whether they click on the article or not: 'click' or 'no click'

Let's first start with importing the necessary packages:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5231, "status": "ok", "timestamp": 1619089424423, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="rDOJOTuWP8Q3" outputId="54ba15b6-9f44-4e8d-f277-e2ad7fc34ff2"
#hide
!pip install vowpalwabbit
```

```python executionInfo={"elapsed": 1064, "status": "ok", "timestamp": 1619089432057, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Ovre-lfCOc_6"
#hide
from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
```

<!-- #region id="yBX5eCYOOc_8" -->
## Simulate reward

In the real world, we will have to learn Tom and Anna's preferences for articles as we observe their interactions. Since this is a simulation, we will have to define Tom and Anna's preference profile. The reward that we provide to the learner will follow this preference profile. Our hope is to see if the learner can take better and better decisions as we see more samples which in turn means we are maximizing the reward.

We will also modify the reward function in a few different ways and see if the CB learner picks up the changes. We will compare the CTR with and without learning.

VW optimizes to minimize **cost which is negative of reward**. Therefore, we will always pass negative of reward as cost to VW.
<!-- #endregion -->

```python executionInfo={"elapsed": 1249, "status": "ok", "timestamp": 1619089448955, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="NOzs-MMwOc_9"
# VW tries to minimize loss/cost, therefore we will pass cost as -reward
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0
```

<!-- #region id="XQTwsgNMOc_-" -->
The reward function below specifies that Tom likes politics in the morning and music in the afternoon whereas Anna likes sports in the morning and politics in the afternoon. It looks dense but we are just simulating our hypothetical world in the format of the feedback the learner understands: cost. If the learner recommends an article that aligns with the reward function, we give a positive reward. In our simulated world this is a click.
<!-- #endregion -->

```python executionInfo={"elapsed": 1101, "status": "ok", "timestamp": 1619089454882, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="xpVW_5juOc__"
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

```python executionInfo={"elapsed": 742, "status": "ok", "timestamp": 1619089458967, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="Wn_sVCzxOdAC"
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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1292, "status": "ok", "timestamp": 1619089464316, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="_Yk4gaXWOdAE" outputId="5e5e1179-3fe6-4c16-9eec-da536ecb0a01"
#collapse-input
context = {"user":"Tom","time_of_day":"morning"}
actions = ["politics", "sports", "music", "food"]

print(to_vw_example_format(context,actions))
```

<!-- #region id="XNOSmbNgOdAE" -->
## Getting a decision

When we call VW we get a _pmf_, [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function), as the output. Since we are incorporating exploration into our strategy, VW will give us a list of probabilities over the set of actions. This means that the probability at a given index in the list corresponds to the likelihood of picking that specific action. In order to arrive at a decision/action, we will have to sample from this list.

So, given a list `[0.7, 0.1, 0.1, 0.1]`, we would choose the first item with a 70% chance. `sample_custom_pmf` takes such a list and gives us the index it chose and what the probability of choosing that index was.
<!-- #endregion -->

```python executionInfo={"elapsed": 1383, "status": "ok", "timestamp": 1619089588369, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="TFpPKrU4OdAF"
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

```python executionInfo={"elapsed": 1149, "status": "ok", "timestamp": 1619089589841, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="W9GHrLJjOdAG"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1266, "status": "ok", "timestamp": 1619089595147, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="fd_g9N5UOdAH" outputId="7761ec5d-6db9-4d6b-c6ef-a830e8985f2b"
#hide-input
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

```python executionInfo={"elapsed": 1242, "status": "ok", "timestamp": 1619089610256, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="7Vd-9I-KOdAI"
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

```python executionInfo={"elapsed": 1267, "status": "ok", "timestamp": 1619089613434, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="wJxpv72YOdAK"
#hide
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2313, "status": "ok", "timestamp": 1619089632070, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="KiWbwzoXOdAL" outputId="f901b148-22ef-4898-dbf4-317a00fa467d"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2088, "status": "ok", "timestamp": 1619089637753, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="voXxj6dwOdAM" outputId="2302e6d8-4be2-4f84-fd90-2e56cc211a85"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2244, "status": "ok", "timestamp": 1619089642926, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="-agneYrYOdAN" outputId="a8eb3ae9-0fb9-4bc2-ff53-0b05de29932e"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1122, "status": "ok", "timestamp": 1619089653766, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="5EY5x6FGOdAO" outputId="c7ca5138-07f5-4009-88ec-4f92698cd53b"
#hide
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

```python executionInfo={"elapsed": 1038, "status": "ok", "timestamp": 1619089662109, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="nVmE_UkFOdAP"
#hide
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

```python executionInfo={"elapsed": 1055, "status": "ok", "timestamp": 1619089668232, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="0O--0JpdOdAQ"
#hide
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2891, "status": "ok", "timestamp": 1619089672602, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="F1pAQ0ecOdAR" outputId="fd5a04dd-5ac7-4d47-ac0a-f773785eae07"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2034, "status": "ok", "timestamp": 1619089674430, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="KPXXayj8OdAR" outputId="3924efe5-b828-4f69-c90e-862730a763be"
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

```python executionInfo={"elapsed": 1039, "status": "ok", "timestamp": 1619089687734, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="AzmkXtSOOdAS"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2497, "status": "ok", "timestamp": 1619089691573, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="VIh8qkMcOdAT" outputId="4dcb0c31-de8e-46c6-a412-3e8ad457c422"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} executionInfo={"elapsed": 2190, "status": "ok", "timestamp": 1619089692853, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}, "user_tz": -330} id="XuE1kVH6OdAU" outputId="b12514e1-e24e-488e-d6ee-b64d8242a138"
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
## Summary

This tutorial aimed at showcasing a real world scenario where contextual bandit algorithms can be used. We were able to take a context and set of actions and learn what actions worked best for a given context. We saw that the learner was able to respond rapidly to changes in the world.  We showed that allowing the learner to interact with the world resulted in higher rewards than the no learning baseline.

This tutorial worked with simplistic features. VW supports high dimensional sparse features, different exploration algorithms and policy evaluation approaches.
<!-- #endregion -->

<!-- #region id="g6TazTXRXiFl" -->
[credits](https://vowpalwabbit.org/tutorials/cb_simulation.html)
<!-- #endregion -->

```python id="OJxfTTT4Xin6"

```
