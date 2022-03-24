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

<!-- #region id="jRc-QwQHUve-" -->
# Probability Theory and the Sample Space Analysis
<!-- #endregion -->

<!-- #region id="hquHcU8cZFob" -->
Few things in life are certain; most things are driven by chance. Whenever we cheer for our favorite sports team, or purchase a lottery ticket, or make an investment in the stock market, we hope for some particular outcome, but that outcome cannot ever be guaranteed. Randomness permeates our day-to-day experiences. Fortunately, that randomness can still be mitigated and controlled. We know that some unpredictable events occur more rarely than others and that certain decisions carry less uncertainty than other much-riskier choices. Driving to work in a car is safer than riding a motorcycle. Investing part of your savings in a retirement account is safer than betting it all on a single hand of blackjack. We can intrinsically sense these trade-offs in certainty because even the most unpredictable systems still show some predictable behaviors. These behaviors have been rigorously studied using *probability theory*. Probability theory is an inherently complex branch of math. However, aspects of the theory can be understood without knowing the mathematical underpinnings. In fact, difficult probability problems can be solved in Python without needing to know a single math equation. Such an equation-free approach to probability requires a baseline understanding of what mathematicians call a *sample space*.

Certain actions have measurable outcomes. A *sample space* is the set of all the possible outcomes an action could produce. Let’s take the simple action of flipping a coin. The coin will land on either heads or tails. Thus, the coin flip will produce one of two measurable outcomes: *heads* or *tails*. By storing these outcomes in a Python set, we can create a sample space of coin flips.
<!-- #endregion -->

<!-- #region id="RWImfNkJ3CHg" -->
## Unbiased Coin
<!-- #endregion -->

```python id="8Ji6E6MJ3GpR"
sample_space = {'Heads', 'Tails'}
```

```python colab={"base_uri": "https://localhost:8080/"} id="KQ56sMsR3IIQ" executionInfo={"status": "ok", "timestamp": 1637252642083, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cb491ed4-b197-4a2c-dc6a-a21651d9e441"
probability_heads = 1 / len(sample_space)
print(f"Probability of choosing heads is {probability_heads}")
```

<!-- #region id="chd76aZ43anP" -->
<!-- #endregion -->

<!-- #region id="bMlY08hi3fIL" -->
*figure: Four event conditions applied to a sample space. The sample space contains two outcomes: heads and tails. Arrows represent the event conditions. Every event condition is a yes-or-no function. Each function filters out those outcomes that do not satisfy its terms. The remaining outcomes form an event. Each event contains a subset of the outcomes found in the sample space. Four events are possible: heads, tails, heads or tails, and neither heads nor tails.*
<!-- #endregion -->

```python id="mNlRBK_C3Kwf"
def is_heads_or_tails(outcome): return outcome in sample_space
def is_neither(outcome): return not is_heads_or_tails(outcome)
def is_heads(outcome): return outcome == "Heads"
def is_tails(outcome): return outcome == "Tails"
```

```python id="KIG-UCEe3v4A"
def get_event(event_condition, sample_space):
  return set([outcome for outcome in sample_space if event_condition(outcome)])
```

```python colab={"base_uri": "https://localhost:8080/"} id="85zVHfi53nXt" executionInfo={"status": "ok", "timestamp": 1637252869285, "user_tz": -330, "elapsed": 468, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4db3f15a-e701-47c1-bf80-85fae6bff039"
event_conditions = [is_heads_or_tails, is_heads, is_tails, is_neither]

for event_condition in event_conditions:
    print(f"Event Condition: {event_condition.__name__}")
    event = get_event(event_condition, sample_space)
    print(f'Event: {event}\n')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hsf8Jqry36nH" executionInfo={"status": "ok", "timestamp": 1637252918423, "user_tz": -330, "elapsed": 692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="671a2bbe-7ee8-4d5b-bbe3-a479d58983cf"
def compute_probability(event_condition, sample_space):
  event = get_event(event_condition, sample_space)
  return len(event) / len(sample_space)

for event_condition in event_conditions:
    print(f"Event Condition: {event_condition.__name__}")
    probability = compute_probability(event_condition, sample_space)
    print(f'Probability: {probability}\n')
```

<!-- #region id="MVD3YMdg33Hc" -->
## Biased Coin
<!-- #endregion -->

```python id="FVYk7dcQ4Q-x"
weighted_sample_space = {'Heads': 4, 'Tails': 1}
```

```python id="eVJ7zsBr4WH1"
sample_space_size = sum(weighted_sample_space.values())
assert sample_space_size == 5
```

```python id="JsBW__Fn4pkY"
# Checking the weighted event size

event = get_event(is_heads_or_tails, weighted_sample_space)
event_size = sum(weighted_sample_space[outcome] for outcome in event)
assert event_size == 5
```

```python id="tFc4NPEJ4qAc"
def compute_event_probability(event_condition, generic_sample_space):
  event = get_event(event_condition, generic_sample_space)
  if type(generic_sample_space) == type(set()):
    return len(event) / len(generic_sample_space)
  event_size = sum(generic_sample_space[outcome] for outcome in event)
  return event_size / sum(generic_sample_space.values())
```

```python colab={"base_uri": "https://localhost:8080/"} id="S1Kg34eM4_Yo" executionInfo={"status": "ok", "timestamp": 1637253131780, "user_tz": -330, "elapsed": 481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8409d865-6ed3-4710-c50c-50a127347e4a"
for event_condition in event_conditions:
    prob = compute_event_probability(event_condition, weighted_sample_space)
    name = event_condition.__name__
    print(f"Probability of event arising from '{name}' is {prob}")
```

<!-- #region id="INX2w3X35CiQ" -->
## Analyzing a family with four children
<!-- #endregion -->

<!-- #region id="6cCseJUb5RdR" -->
Suppose a family has four children. What is the probability that exactly two of the children are boys? We’ll assume that each child is equally likely to be either a boy or a girl. Thus we can construct an unweighted sample space where each outcome represents one possible sequence of four children, as shown in figure below:
<!-- #endregion -->

<!-- #region id="W83_D_Gh5UeE" -->
<!-- #endregion -->

<!-- #region id="4l6hbOVb5TN8" -->
*fgiure: The sample space for four sibling children. Each row in the sample space contains 1 of 16 possible outcomes. Every outcome represents a unique combination of four children. The sex of each child is indicated by a letter: B for boy and G for girl. Outcomes with two boys are marked by an arrow. There are six such arrows; thus, the probability of two boys equals 6 / 16.*
<!-- #endregion -->

```python id="AmXqH2Au5ZQG"
from itertools import product

possible_children = ['Boy', 'Girl']

# Naive approach
sample_space = set()
for child1 in possible_children:
    for child2 in possible_children:
        for child3 in possible_children:
            for child4 in possible_children:
                outcome = (child1, child2, child3, child4)
                sample_space.add(outcome)

# Better approach
all_combinations = product(*(4 * [possible_children]))
assert set(all_combinations) == sample_space

# Best approach
sample_space_efficient = set(product(possible_children, repeat=4))
assert sample_space == sample_space_efficient
```

```python colab={"base_uri": "https://localhost:8080/"} id="pmjwZCXI9afU" executionInfo={"status": "ok", "timestamp": 1637254279656, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="50ef59ce-94e0-4c8b-cbe3-e0105bbccf74"
sample_space
```

```python colab={"base_uri": "https://localhost:8080/"} id="byz_MDHp5r8V" executionInfo={"status": "ok", "timestamp": 1637253340652, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b3e3da91-3d54-4908-fd1d-5ff7754e0a00"
def has_two_boys(outcome): 
  num = len([child for child in outcome if child == 'Boy']) == 2
  return num

prob = compute_probability(has_two_boys, sample_space)
print(f"Probability of 2 boys: {prob}")
```

<!-- #region id="NsCBdutI54rD" -->
## Analyzing multiple die rolls
<!-- #endregion -->

<!-- #region id="ZUoZ5eDW55P8" -->
Suppose we’re shown a fair six-sided dice whose faces are numbered from 1 to 6. The die is rolled six times. What is the probability that these six die rolls add up to 21?

We begin by defining the possible values of any single roll. These are integers that range from 1 to 6.
<!-- #endregion -->

```python id="o663SyNI6TkQ"
numbers_on_die = list(range(1,7))

sample_space_die = set(product(numbers_on_die, repeat=6))
```

```python colab={"base_uri": "https://localhost:8080/"} id="MKEbMIvy5--6" executionInfo={"status": "ok", "timestamp": 1637253435527, "user_tz": -330, "elapsed": 502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="43e12359-dd00-499b-c176-916819d17284"
def add_up_to_21(outcome):
  return sum(outcome) == 21

prob = compute_event_probability(add_up_to_21, sample_space_die)
print(f"Probability that a dice-roll out of 6 dice rolls add up to 21: {prob:2f}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="njOXhlop6Mtw" executionInfo={"status": "ok", "timestamp": 1637253467730, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="febf923a-6335-4d68-dee6-1fa4c256c0c9"
prob = compute_event_probability(lambda x: sum(x) == 21, sample_space_die)
print(f"Probability that a dice-roll out of 6 dice rolls add up to 21: {prob:2f}")
```

<!-- #region id="eyxakyPT6evS" -->
## Computing die-roll probabilities using weighted sample spaces

We’ve just computed the likelihood of six die rolls summing to 21. Now, let’s recompute that probability using a weighted sample space. We need to convert our unweighted sample space set into a weighted sample space dictionary; this will require us to identify all possible die-roll sums. Then we must count the number of times each sum appears across all possible die-roll combinations. These combinations are already stored in our computed sample_space set. By mapping the die-roll sums to their occurrence counts, we will produce a weighted_sample_space result.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2H4_mqIi6k5C" executionInfo={"status": "ok", "timestamp": 1637253558941, "user_tz": -330, "elapsed": 567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2146fc83-5bd9-40a0-c7ec-88c90e753e34"
from collections import defaultdict

weighted_sample_space = defaultdict(int)

for outcome in sample_space_die:
  total = sum(outcome)
  weighted_sample_space[total] += 1

weighted_sample_space
```

```python colab={"base_uri": "https://localhost:8080/"} id="CXy9hjvm6qxs" executionInfo={"status": "ok", "timestamp": 1637253636155, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dd0f23dc-5d10-42bb-d35a-1795cd15b4eb"
num_combinations = weighted_sample_space.get(21)
print(f"There are {num_combinations } ways for six rolled dice to sum to 21")

prob = compute_event_probability(lambda x: x == 21, weighted_sample_space)
print(f"Probability that a dice-roll out of 6 dice rolls add up to 21: {prob:2f}")

print('Number of Elements in Unweighted Sample Space:')
print(len(sample_space_die))

print('Number of Elements in Weighted Sample Space:')
print(len(weighted_sample_space))
```

<!-- #region id="nQsE23eC69tJ" -->
## Computing probabilities over interval ranges
<!-- #endregion -->

<!-- #region id="tk21A6Bv7cy4" -->
So far, we’ve only analyzed event conditions that satisfy some single value. Now we’ll analyze event conditions that span intervals of values. An interval is the set of all the numbers between and including two boundary cutoffs. Let’s define an is_in_interval function that checks whether a number falls within a specified interval. We’ll control the interval boundaries by passing a minimum and a maximum parameter.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nWxbs-Zm7zb8" executionInfo={"status": "ok", "timestamp": 1637253880332, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b003d388-6579-4d22-d7d0-bb24fa3a4b5e"
def is_in_interval(number, minimum, maximum):
  return minimum <= number <= maximum
  
prob = compute_event_probability(lambda x: is_in_interval(x, 10,21), weighted_sample_space)
print(f"Probability of the interval: {prob:2f}")
```

<!-- #region id="T1k4o_AT75Ua" -->
Interval analysis is critical to solving a whole class of very important problems in probability and statistics. One such problem involves the evaluation of extremes, the problem boils down to whether obeserved data is too extreme to be believable. Data seems extreme when it is too unusual to have occurred by random chance. For instance, suppose we observe 10 flips of an allegedly fair coin, and that coin lands on heads 8 out of 10 times. Now, we had expected the coin to hit tails half the time, not 20% of the time, so the observations seem a little strange. Is the coin actually fair? Or has it been secretly replaced with a trick coin that falls on heads a majority of the time? We’ll try to find out by asking the following question: what is the probability that 10 fair coin-flips lead to an extreme number of heads? We’ll define an extreme head-count as observing of 8 heads or more. Thus, we can describe the problem as follows: what is the probability that 10 fair coin-flips produce between 8 and 10 heads?
<!-- #endregion -->

```python id="3HuRdIIT8GPj"
coin_sides = ['Heads', 'Tails']
```

```python colab={"base_uri": "https://localhost:8080/"} id="pc2pkoil8H-6" executionInfo={"status": "ok", "timestamp": 1637254025688, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14c419fe-d4d1-4368-cade-12402d4063ca"
from pprint import pprint

def generate_coin_sample_space(num_flips):
  weighted_sample_space_coins = defaultdict(int)
  combinations = set(product(coin_sides, repeat=num_flips))
  for combination in combinations:
    num_heads = len([combo for combo in combination if combo == "Heads"])
    weighted_sample_space_coins[num_heads] += 1
  return weighted_sample_space_coins
  
weighted_sample_space_coins = generate_coin_sample_space(10)
pprint(weighted_sample_space_coins)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kQldhwmo8cwk" executionInfo={"status": "ok", "timestamp": 1637254113254, "user_tz": -330, "elapsed": 601, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f019cd79-be5b-460d-f733-b1f684d3d17e"
prob = compute_event_probability(lambda x: is_in_interval(x, 8,10), weighted_sample_space_coins)
print(f"Probability of observing more than 7 heads is {prob}")
```

<!-- #region id="PSGMmS6Y8yKH" -->
Our observed head-count does not commonly occur. Does this mean the coin is biased? Not necessarily. Observing 8 out of 10 tails is as extreme as observing 8 out of 10 heads. Had we observed 8 tails and not 8 heads, we would still be suspicious of the coin. Our computed interval did not take this tails-driven extreme into account. Instead, we treated 8 or more tails as just another normal possibility. If we truly wish to measure the fairness of our coin, we’ll need to update our interval computations. We’ll need to include the likelihood of observing 8 or more tails. This is equivalent to observing 2 heads or less. Let's formulate the problem as follows; what is the probability that 10 fair coin-flips produce either 0 to 2 heads or 8 to 10 heads? Or, stated more concisely, what is the probability that the coin-flips do NOT produce between 3 and 7 heads?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="J12L7Zkv881T" executionInfo={"status": "ok", "timestamp": 1637254163222, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f257f61-8817-491d-f94a-43b1ed36686c"
prob = compute_event_probability(lambda x: not is_in_interval(x, 3, 7), weighted_sample_space_coins)
print(f"Probability of observing more than 7 heads or 7 tails is {prob}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="IyCtMAuI8-U2" executionInfo={"status": "ok", "timestamp": 1637254175440, "user_tz": -330, "elapsed": 3704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc3ffbe1-624d-458b-876c-19c7a108bd46"
weighted_sample_space_20_flips = generate_coin_sample_space(20)
prob = compute_event_probability(lambda x: not is_in_interval(x, 5, 15),
                                 weighted_sample_space_20_flips)
print(f"Probability of observing more than 15 heads or 15 tails is {prob}")
```

<!-- #region id="RtTYxjWi9AmB" -->
The updated probability has dropped from approximately .1 to approximately .01. Thus, the added evidence has caused a 10-fold decrease in our confidence of fairness.

Despite this probability drop, the ratio of heads to tails has remained constant at 4-to-1. Both our original and updated experiments produced 80% heads and 20% tails. This leads to an interesting question: why does the probability of observing 80% or more heads decrease as the supposedly fair coin gets flipped more times?

We can find out through detailed mathematical analysis. However, a much more intuitive solution is to just visualize the distribution of head-counts across our 2 sample space dictionaries. The visualization would effectively be a plot of keys (head-counts) vs values (combination counts) present in each dictionary.
<!-- #endregion -->

<!-- #region id="gdHx9UEG9y_V" -->
## Plotting coin-flip probabilities
<!-- #endregion -->

```python id="cudqQ7Q8950d"
%matplotlib inline

import matplotlib.pyplot as plt
```

<!-- #region id="1fLRPI7t9zdL" -->
We now have tools to visualize the relationship between a coin-flip count and the probability of heads. In section 1, we examined the probability of seeing 80% or more heads across a series of coin flips. That probability decreased as the coin-flip count went up, and we wanted to know why. We’ll soon find out by plotting head counts versus their associated coin-flip combination counts. These values were already computed in our section 1 analysis. The keys in the weighted_sample_space dictionary contain all possible head counts across 10 flipped coins. These head counts map to combination counts. Meanwhile, the weighted_sample_space_20_flips dictionary contains the head-count mappings for 20 flipped coins.

Our aim is to compare the plotted data from both these dictionaries. We begin by plotting the elements of weighted_sample_space: we plot its keys on the x-axis versus the associated values on the y-axis. The x-axis corresponds to 'Head-count', and the y-axis corresponds to 'Number of coin-flip combinations with x heads'. We use a scatter plot to visualize key-to-value relationships directly without connecting any plotted points.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UbCBkyfT-vrs" executionInfo={"status": "ok", "timestamp": 1637254677489, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e6a971d-f478-4f90-bdf8-64dec982d31c"
def generate_coin_sample_space(num_flips=10):
    weighted_sample_space = defaultdict(int)
    for coin_flips in product(['Heads', 'Tails'], repeat=num_flips):
        heads_count = len([outcome for outcome in coin_flips
                          if outcome == 'Heads'])
        weighted_sample_space[heads_count] += 1

    return weighted_sample_space

weighted_sample_space = generate_coin_sample_space()
weighted_sample_space
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="tyQl5rbm94C5" executionInfo={"status": "ok", "timestamp": 1637254682099, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="502e0fd4-13d7-46b2-87c0-79cba1619883"
x_10_flips = list(weighted_sample_space.keys())
y_10_flips = [weighted_sample_space[key] for key in x_10_flips]
plt.scatter(x_10_flips, y_10_flips)
plt.xlabel('Head-count')
plt.ylabel('Number of coin-flip combinations with x heads')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="WM9mwWAG9-V9" executionInfo={"status": "ok", "timestamp": 1637254687376, "user_tz": -330, "elapsed": 785, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d591522f-e3c5-4539-dce6-7c16c8a97490"
sample_space_size = sum(weighted_sample_space.values())
prob_x_10_flips = [value / sample_space_size for value in y_10_flips]
plt.scatter(x_10_flips, prob_x_10_flips)
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

```python id="41rmTnxL-bs-"
assert sum(prob_x_10_flips) == 1.0
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="xrLfcsVA-Rij" executionInfo={"status": "ok", "timestamp": 1637254706752, "user_tz": -330, "elapsed": 547, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="88026bef-ea3b-4239-ba27-0a090ece840c"
plt.plot(x_10_flips, prob_x_10_flips)
plt.scatter(x_10_flips, prob_x_10_flips)
where = [is_in_interval(value, 8, 10) for value in x_10_flips]
plt.fill_between(x_10_flips, prob_x_10_flips, where=where)
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="AJIuRt1j-W_g" executionInfo={"status": "ok", "timestamp": 1637254722955, "user_tz": -330, "elapsed": 895, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d49ba578-2eab-4f85-b077-a46c010acd3c"
plt.plot(x_10_flips, prob_x_10_flips)
plt.scatter(x_10_flips, prob_x_10_flips)
where = [not is_in_interval(value, 3, 7) for value in x_10_flips]
plt.fill_between(x_10_flips, prob_x_10_flips, where=where)
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="M-8MvksI_UVh" executionInfo={"status": "ok", "timestamp": 1637254784981, "user_tz": -330, "elapsed": 2299, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="42d54677-afd2-45e7-d2e1-5d7d9335e2f8"
weighted_sample_space_20_flips = generate_coin_sample_space(num_flips=20)
weighted_sample_space_20_flips
```

```python id="mADs29av_KIZ"
x_20_flips = list(weighted_sample_space_20_flips.keys())
y_20_flips = [weighted_sample_space_20_flips[key] for key in x_20_flips]
sample_space_size = sum(weighted_sample_space_20_flips.values())
prob_x_20_flips = [value / sample_space_size for value in y_20_flips]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="Hss8PqrQ_MTa" executionInfo={"status": "ok", "timestamp": 1637254786674, "user_tz": -330, "elapsed": 1143, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="08a94c00-28d8-4947-a82c-cbe155996494"
plt.plot(x_10_flips, prob_x_10_flips)
plt.scatter(x_10_flips, prob_x_10_flips)
plt.plot(x_20_flips, prob_x_20_flips, color='black', linestyle='--')
plt.scatter(x_20_flips, prob_x_20_flips, color='k', marker='x')
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="gEmydVWx_O7p" executionInfo={"status": "ok", "timestamp": 1637254792136, "user_tz": -330, "elapsed": 774, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f876f57-a252-4c40-aa60-fd48cc6a2647"
plt.plot(x_10_flips, prob_x_10_flips, label='A: 10 coin-flips')
plt.plot(x_20_flips, prob_x_20_flips, color='k', linestyle='--',
         label='B: 20 coin-flips')
plt.legend()

where_10 = [not is_in_interval(value, 3, 7) for value in x_10_flips]
plt.fill_between(x_10_flips, prob_x_10_flips, where=where_10)
where_20 = [not is_in_interval(value, 5, 15) for value in x_20_flips]
plt.fill_between(x_20_flips, prob_x_20_flips, where=where_20)

plt.xlabel('Head-Count')
plt.ylabel('Probability')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="2d9JWNHn_dWd" executionInfo={"status": "ok", "timestamp": 1637254816046, "user_tz": -330, "elapsed": 927, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af3cd21d-45d9-4bd8-cab8-e3f127091a76"
x_10_frequencies = [head_count /10 for head_count in x_10_flips]
x_20_frequencies = [head_count /20 for head_count in x_20_flips]

plt.plot(x_10_frequencies, prob_x_10_flips, label='A: 10 coin-flips')
plt.plot(x_20_frequencies, prob_x_20_flips, color='k', linestyle=':', label='B: 20 coin-flips')
plt.legend()

plt.xlabel('Head-Frequency')
plt.ylabel('Probability')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="XN4brxoN_fug" executionInfo={"status": "ok", "timestamp": 1637254827793, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="03a2e123-8cc2-4534-bd4d-faae41fc66b5"
relative_likelihood_10 = [10 * prob for prob in prob_x_10_flips]
relative_likelihood_20 = [20 * prob for prob in prob_x_20_flips]

plt.plot(x_10_frequencies, relative_likelihood_10, label='A: 10 coin-flips')
plt.plot(x_20_frequencies, relative_likelihood_20, color='k',
         linestyle=':', label='B: 20 coin-flips')

plt.fill_between(x_10_frequencies, relative_likelihood_10, where=where_10)
plt.fill_between(x_20_frequencies, relative_likelihood_20, where=where_20)

plt.legend()
plt.xlabel('Head-Frequency')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="HvGTADSl_fqt" -->
## Simulating random coin flips and dice rolls using NumPy
<!-- #endregion -->

```python id="bFZpJUWPCUhV"
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="n9EmrjOMCVan" executionInfo={"status": "ok", "timestamp": 1637255582428, "user_tz": -330, "elapsed": 679, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d87beb7e-6dac-4a13-b604-5a566f242d70"
dice_roll = np.random.randint(1, 7)
assert 1 <= dice_roll <= 6
dice_roll
```

```python colab={"base_uri": "https://localhost:8080/"} id="9o9139MFCVXN" executionInfo={"status": "ok", "timestamp": 1637255613688, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b81af27b-c4a0-450e-8cee-c296df88a2b9"
np.random.seed(0)
coin_flip = np.random.randint(0, 2)
print(f"Coin landed on {'heads' if coin_flip == 1 else 'tails'}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="qL54HXrNCVUW" executionInfo={"status": "ok", "timestamp": 1637255671409, "user_tz": -330, "elapsed": 1023, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9e36ed32-1a2b-4015-ff0a-a3060fe1eff5"
np.random.seed(0)
def frequency_heads(coin_flip_sequence):
    total_heads = sum(coin_flip_sequence)
    return total_heads / len(coin_flip_sequence)

coin_flips = [np.random.randint(0, 2) for _ in range(10)]
freq_heads = frequency_heads(coin_flips)
print(f"Frequency of Heads is {freq_heads}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="mnG9ESg5CVRQ" executionInfo={"status": "ok", "timestamp": 1637255692933, "user_tz": -330, "elapsed": 628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7010678f-88b2-4d5d-d0e8-9128ec5889b5"
np.random.seed(0)
coin_flips = []
frequencies = []
for _ in range(1000):
    coin_flips.append(np.random.randint(0, 2))
    frequencies.append(frequency_heads(coin_flips))

plt.plot(list(range(1000)), frequencies)
plt.axhline(.5, color='k')
plt.xlabel('Number of Coin Flips')
plt.ylabel('Head-Frequency')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="UgttiKvMCx0g" executionInfo={"status": "ok", "timestamp": 1637255709206, "user_tz": -330, "elapsed": 720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="190cd253-ac2b-42ca-9fa2-c3ad88e22f3a"
np.random.seed(0)
print("Lets flip the biased coin once.")
coin_flip = np.random.binomial(1, .7)
print(f"Biased coin landed on {'heads' if coin_flip == 1 else 'tails'}.")

print("\nLets flip the biased coin 10 times.")
number_coin_flips = 10
head_count = np.random.binomial(number_coin_flips, .7)
print((f"{head_count} heads were observed out of "
       f"{number_coin_flips} biased coin flips"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="bsBTPiSbCxxC" executionInfo={"status": "ok", "timestamp": 1637255720050, "user_tz": -330, "elapsed": 781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="063bc4b3-f98a-4d86-988c-14d2d8303d34"
np.random.seed(0)
head_count = np.random.binomial(1000, .7)
frequency = head_count / 1000
print(f"Frequency of Heads is {frequency}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="mlSS7HSrCxtE" executionInfo={"status": "ok", "timestamp": 1637255749139, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32f2bed2-5700-48d4-ee0d-7049b0642937"
np.random.seed(0)
for i in range(1, 6):
    head_count = np.random.binomial(1000, .7)
    frequency = head_count / 1000
    print(f"Frequency at iteration {i} is {frequency}")
    if frequency == 0.7:
        print("Frequency equals the probability!\n")
```

<!-- #region id="mRmPYNaq_fnb" -->
## Computing confidence intervals using histograms and NumPy arrays
<!-- #endregion -->

<!-- #region id="vOuUIkGZDOi9" -->
Suppose we’re handed a biased coin whose bias we don’t know. We flip the coin 1,000 times and observe a frequency of 0.709. We know the frequency approximates the actual probability, but by how much? More precisely, what are the chances of the actual probability falling within an interval close to 0.709 (such as an interval between 0.7 and 0.71)? To find out, we must do additional sampling.

We’ve previously sampled our coin over five iterations of 1,000 coin flips each. The sampling produced some fluctuations in the frequency. Let’s explore these fluctuations by increasing our frequency count from 5 to 500. We can execute this supplementary sampling by running [np.random.binomial(1000, 0.7) for _ in range(500)].
<!-- #endregion -->

```python id="t3h9i5JtDPOU"
np.random.seed(0)
head_count_list = [np.random.binomial(1000, .7) for _ in range(500)]
np.random.seed(0)
head_count_array = np.random.binomial(1000, 0.7, 500)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3ObPE0fRDXmR" executionInfo={"status": "ok", "timestamp": 1637255882362, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="514705ca-8d9e-44b5-ca5c-7ebab8b8e3e6"
assert head_count_array.tolist() == head_count_list
new_array = np.array(head_count_list)
assert np.array_equal(new_array, head_count_array) == True
head_count_list[0:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Sx2aWLhNDlU_" executionInfo={"status": "ok", "timestamp": 1637255902040, "user_tz": -330, "elapsed": 815, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ffe2396c-f6fa-480e-93c0-716b83f00cc0"
frequency_array = head_count_array / 1000
assert frequency_array.tolist() == [head_count / 1000
                                    for head_count in head_count_list]
assert frequency_array.tolist() == list(map(lambda x: x / 1000,
                                        head_count_list))
print(frequency_array[:10])
```

```python colab={"base_uri": "https://localhost:8080/"} id="iQDS_1m5Dwcl" executionInfo={"status": "ok", "timestamp": 1637255943783, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99577713-8695-422c-b6ae-30f27293979a"
min_freq = frequency_array.min()
max_freq = frequency_array.max()
print(f"Minimum frequency observed: {min_freq}")
print(f"Maximum frequency observed: {max_freq}")
print(f"Difference across frequency range: {max_freq - min_freq}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="AjMenBbrDwZM" executionInfo={"status": "ok", "timestamp": 1637255958396, "user_tz": -330, "elapsed": 653, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d9d2aef-cb9b-4abc-8a58-a21a549b4450"
frequency_counts = defaultdict(int)
for frequency in frequency_array:
    frequency_counts[frequency] += 1

frequencies = list(frequency_counts.keys())
counts = [frequency_counts[freq] for freq in frequencies]
plt.scatter(frequencies, counts)
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="ceNQ6nCWDwVc" executionInfo={"status": "ok", "timestamp": 1637255977301, "user_tz": -330, "elapsed": 568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d190c4d7-9b9b-4aae-bcc5-2649fe49ad7b"
plt.hist(frequency_array, bins='auto', edgecolor='black')
plt.xlabel('Binned Frequency')
plt.ylabel('Count')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="uWEdsiw7DwRb" executionInfo={"status": "ok", "timestamp": 1637255987770, "user_tz": -330, "elapsed": 661, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c18dcf33-452a-438e-aeef-f5d03fddd61a"
counts, _, _ = plt.hist(frequency_array, bins='auto',
                        edgecolor='black')

print(f"Number of Bins: {counts.size}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="-6cdUJOUDwN1" executionInfo={"status": "ok", "timestamp": 1637255998591, "user_tz": -330, "elapsed": 655, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f31b4cde-7861-483c-a1db-a1699a76be88"
counts, bin_edges, _ = plt.hist(frequency_array, bins='auto',
                                edgecolor='black')

bin_width = bin_edges[1] - bin_edges[0]
assert bin_width == (max_freq - min_freq) / counts.size
print(f"Bin width: {bin_width}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="XxCuXDxaD-Es" executionInfo={"status": "ok", "timestamp": 1637256005290, "user_tz": -330, "elapsed": 493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3896a6b1-815e-4094-b23b-23423bfe9630"
def output_bin_coverage(i):
    count = int(counts[i])
    range_start, range_end = bin_edges[i], bin_edges[i+1]
    range_string = f"{range_start} - {range_end}"
    print((f"The bin for frequency range {range_string} contains "
           f"{count} element{'' if count == 1 else 's'}"))

output_bin_coverage(0)
output_bin_coverage(5)
```

```python id="CHnWMt-mD-Ad"
assert counts[counts.argmax()] == counts.max()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dkPPTJwvD982" executionInfo={"status": "ok", "timestamp": 1637256019891, "user_tz": -330, "elapsed": 726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77b0ce19-1032-4c55-fa9c-513f6e9e8f1d"
output_bin_coverage(counts.argmax())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="9OVfc3pzD949" executionInfo={"status": "ok", "timestamp": 1637256314418, "user_tz": -330, "elapsed": 500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6d73c151-bdd6-4637-fd58-e506ddcb8be9"
likelihoods, bin_edges, _ = plt.hist(frequency_array, bins='auto', edgecolor='black', density=True)
plt.xlabel('Binned Frequency')
plt.ylabel('Relative Likelihood')
plt.show()
```

```python id="A-2EK8XYD91D"
assert likelihoods.sum() * bin_width == 1.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="1uhpXZb1FSdp" executionInfo={"status": "ok", "timestamp": 1637256346354, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c7568ba0-fa38-44c6-b3b1-f4235929886c"
index = likelihoods.argmax()
area = likelihoods[index] * bin_width
range_start, range_end = bin_edges[index], bin_edges[index+1]
range_string = f"{range_start} - {range_end}"
print(f"Sampled frequency falls within interval {range_string} with probability {area}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="pGm_EakkFSaK" executionInfo={"status": "ok", "timestamp": 1637256352797, "user_tz": -330, "elapsed": 455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d29411d7-1322-4059-f803-9c5e68e59edc"
peak_index = likelihoods.argmax()
start_index, end_index = (peak_index - 1, peak_index + 2)
area = likelihoods[start_index: end_index + 1].sum() * bin_width
range_start, range_end = bin_edges[start_index], bin_edges[end_index]
range_string = f"{range_start} - {range_end}"
print(f"Sampled frequency falls within interval {range_string} with probability {area}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="tCeJHZtrFSWa" executionInfo={"status": "ok", "timestamp": 1637256361328, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="baa0b010-e18e-46d7-b92c-1dfb5801fa1b"
def compute_high_confidence_interval(likelihoods, bin_width):
    peak_index = likelihoods.argmax()
    area = likelihoods[peak_index] * bin_width
    start_index, end_index = peak_index, peak_index + 1
    while area < 0.95:
        if start_index > 0:
            start_index -= 1
        if end_index < likelihoods.size - 1:
            end_index += 1

        area = likelihoods[start_index: end_index + 1].sum() * bin_width

    range_start, range_end = bin_edges[start_index], bin_edges[end_index]
    range_string = f"{range_start:.6f} - {range_end:.6f}"
    print((f"The frequency range {range_string} represents a "
           f"{100 * area:.2f}% confidence interval"))
    return start_index, end_index

compute_high_confidence_interval(likelihoods, bin_width)
```

```python id="NKIp_O2QFSS4"
np.random.seed(0)
head_count_array = np.random.binomial(1000, 0.7, 100000)
frequency_array = head_count_array / 1000
assert frequency_array.size == 100000
```

```python colab={"base_uri": "https://localhost:8080/", "height": 296} id="nQZBmuELFSPB" executionInfo={"status": "ok", "timestamp": 1637256375035, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cbde15e3-f930-4872-8cd8-2312044ff8bc"
likelihoods, bin_edges, patches = plt.hist(frequency_array, bins='auto',
                                           edgecolor='black', density=True)
bin_width = bin_edges[1] - bin_edges[0]
start_index, end_index = compute_high_confidence_interval(likelihoods,
                                                          bin_width)

for i in range(start_index, end_index):
     patches[i].set_facecolor('yellow')
plt.xlabel('Binned Frequency')
plt.ylabel('Relative Likelihood')

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="_e1qxp1IFSLB" executionInfo={"status": "ok", "timestamp": 1637256382090, "user_tz": -330, "elapsed": 535, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="23c46520-e7e1-4ba4-84ac-1d31a9cf2b30"
np.random.seed(0)
head_count_array = np.random.binomial(50000, 0.7, 100000)
frequency_array = head_count_array / 50000

likelihoods, bin_edges, patches = plt.hist(frequency_array, bins=25,
                                           edgecolor='black', density=True)
bin_width = bin_edges[1] - bin_edges[0]
start_index, end_index = compute_high_confidence_interval(likelihoods,
                                                          bin_width)

for i in range(start_index, end_index):
     patches[i].set_facecolor('yellow')
plt.xlabel('Binned Frequency')
plt.ylabel('Relative Likelihood')

plt.show()
```

<!-- #region id="4sqYAYHZ_fji" -->
## Using confidence intervals to analyze a biased deck of cards
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JF6Fn9jXFgtD" executionInfo={"status": "ok", "timestamp": 1637256402391, "user_tz": -330, "elapsed": 569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="320b3da4-331a-4718-82ce-27639dcda36a"
np.random.seed(0)

likelihoods, bin_edges = np.histogram(frequency_array, bins='auto',
                                      density=True)
bin_width = bin_edges[1] - bin_edges[0]
compute_high_confidence_interval(likelihoods, bin_width)
```

```python colab={"base_uri": "https://localhost:8080/"} id="dI_U-d_tFyLg" executionInfo={"status": "ok", "timestamp": 1637256494089, "user_tz": -330, "elapsed": 474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e3fb82d3-e5e0-4627-cf46-cdd26b6516b0"
np.random.seed(0)
total_cards = 52
red_card_count = np.random.randint(0, total_cards + 1)
black_card_count = total_cards - red_card_count
assert black_card_count != red_card_count
weighted_sample_space = {'red_card': red_card_count,
                         'black_card': black_card_count}
prob_red = compute_event_probability(lambda x: x == 'red_card',
                                     weighted_sample_space)
red_card_count
```

```python colab={"base_uri": "https://localhost:8080/"} id="7iD17lC0F2zy" executionInfo={"status": "ok", "timestamp": 1637256504015, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a8a48d0-b0d2-4c0c-cfb0-e2957e748341"
assert prob_red == red_card_count / total_cards
np.random.seed(0)
color = 'red' if np.random.binomial(1, prob_red) else 'black'
print(f"The first card in the shuffled deck is {color}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="tL0WT9wXF49d" executionInfo={"status": "ok", "timestamp": 1637256524337, "user_tz": -330, "elapsed": 529, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9ad394d-8ccf-40ec-c921-2191273dd2a8"
np.random.seed(0)
red_count = np.random.binomial(10, prob_red)
print(f"In {red_count} of out 10 shuffles, a red card came up first.")
```

```python colab={"base_uri": "https://localhost:8080/"} id="FtshU09NFqQ8" executionInfo={"status": "ok", "timestamp": 1637256532725, "user_tz": -330, "elapsed": 517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b76fa15d-878d-408c-cdb8-07ecb502060d"
np.random.seed(0)
red_card_count_array = np.random.binomial(50000, prob_red, 100000)
frequency_array = red_card_count_array / 50000

likelihoods, bin_edges = np.histogram(frequency_array, bins='auto',
                                      density=True)
bin_width = bin_edges[1] - bin_edges[0]
start_index, end_index = compute_high_confidence_interval(likelihoods,
                                                          bin_width)
```

```python colab={"base_uri": "https://localhost:8080/"} id="uvFl_pT0FsIO" executionInfo={"status": "ok", "timestamp": 1637256552899, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f74a816d-9442-46d9-91f9-94c5f596ab81"
range_start = round(0.842771 * total_cards)
range_end = round(0.849139 * total_cards)
print(f"The number of red cards in the deck is between {range_start} and {range_end}")

if red_card_count == 44:
    print('We are correct! There are 44 red cards in the deck')
else:
    print('Oops! Our sampling estimation was wrong.')
```

<!-- #region id="uWjbBQNxCKs-" -->
## Using permutations to shuffle cards
<!-- #endregion -->

<!-- #region id="EBJBkEL2GMjc" -->
Card shuffling requires us to randomly reorder the elements of a card deck. That random reordering can be carried out using the np.random.shuffle method. The function takes as input an ordered array or list and shuffles its elements in place. The following code randomly shuffles a deck of cards containing two red cards (represented by 1s) and two black cards (represented by 0s).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xOH0zVyYGM0a" executionInfo={"status": "ok", "timestamp": 1637256591877, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8298c5d6-d1ef-4ad3-9e03-8dd7a2eaa4ee"
np.random.seed(0)
card_deck = [1, 1, 0, 0]
np.random.shuffle(card_deck)
print(card_deck)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ctflfV-6GPTX" executionInfo={"status": "ok", "timestamp": 1637256598220, "user_tz": -330, "elapsed": 695, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2bb0a8a5-0229-4667-9bbd-2e0082b330b4"
np.random.seed(0)
unshuffled_deck = [1, 1, 0, 0]
shuffled_deck = np.random.permutation(unshuffled_deck)
assert unshuffled_deck == [1, 1, 0, 0]
print(shuffled_deck)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kIJGyKeoGQwh" executionInfo={"status": "ok", "timestamp": 1637256603896, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e0247d11-0a07-4d4e-99fb-95c308562b26"
import itertools
for permutation in list(itertools.permutations(unshuffled_deck))[:3]:
    print(permutation)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6aXZ8U2DGSO-" executionInfo={"status": "ok", "timestamp": 1637256609297, "user_tz": -330, "elapsed": 670, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b9ff053-9bad-4814-a0cf-7078185119ec"
for permutation in list(itertools.permutations([0, 1, 2, 3]))[:3]:
    print(permutation)
```

```python colab={"base_uri": "https://localhost:8080/"} id="96wGVBLNGTeC" executionInfo={"status": "ok", "timestamp": 1637256614953, "user_tz": -330, "elapsed": 555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="74cb1964-9717-4710-8e38-57579f49f40a"
weighted_sample_space = defaultdict(int)
for permutation in itertools.permutations(unshuffled_deck):
    weighted_sample_space[permutation] += 1

for permutation, count in weighted_sample_space.items():
    print(f"Permutation {permutation} occurs {count} times")
```

```python colab={"base_uri": "https://localhost:8080/"} id="O2ZejIlpGU1Q" executionInfo={"status": "ok", "timestamp": 1637256675090, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2ec8222d-1b69-498c-c758-324ed17db6bc"
sample_space = set(itertools.permutations(unshuffled_deck))
event_condition = lambda x: list(x) == unshuffled_deck
prob = compute_event_probability(event_condition, sample_space)
assert prob == 1 / len(sample_space)
print(f"Probability that a shuffle does not alter the deck is {prob}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="T0HsSwfQGjpN" executionInfo={"status": "ok", "timestamp": 1637256678760, "user_tz": -330, "elapsed": 893, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="85fe2db8-df50-477a-b44c-7d2be619b7a0"
red_cards = 5 * [1]
black_cards = 5 * [0]
unshuffled_deck = red_cards + black_cards
sample_space = set(itertools.permutations(unshuffled_deck))
print(f"Sample space for a 10-card deck contains {len(sample_space)} elements")
```

<!-- #region id="JPh4pgW6GkSd" -->
**END**
<!-- #endregion -->
