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

```python id="6quAOvYG3n2m"
import random
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

```python id="v349jzvM4xFP"
class Bandit:
    """A useful class containing the multi-armed bandit and all its actions.
    
    Attributes:
        actions The actions that can be performed, numbered automatically 0, 1, 2...
        payoff_probs    The underlying pay-off probabilities for each action.
    """

    def __init__(self, payoff_probs):
        self.actions = range(len(payoff_probs))
        self.pay_offs = payoff_probs

    def sample(self, action):
        """Sample from the multi-armed by performing an action.
        
        Args:
            action (int): The action performed on the multi-armed bandit.

        Returns:
            int: It returns a reward based on that arm's pay-off probability.
        """
        selector = random.random()
        return 1 if selector <= self.pay_offs[action] else 0
```

```python id="OZpjiBUh4w_e"
def random_agent(bandit, iterations):
    """Randomly select an action and reward."""

    for i in range(iterations):
        a = random.choice(bandit.actions)
        r = bandit.sample(a)
        yield a, r

def optimal_agent(bandit, iterations):
    """Select the best action by taking a sneak-peek at the bandit's probabilities."""

    for i in range(iterations):
        a = bandit.pay_offs.index(max(bandit.pay_offs))
        r = bandit.sample(a)
        yield a, r

def initial_explore_agent(bandit, iterations, initial_rounds = 10):
    """Initially explore initial_rounds times and then stick to the best action."""
    pay_offs = dict()
    best_action = -1

    for i in range(iterations):
        # for the initial rounds pick a random action
        if i < initial_rounds:
            a = random.choice(bandit.actions)
            r = bandit.sample(a)

            #update rewards
            if a in pay_offs:
                pay_offs[a].append(r)
            else:
                pay_offs[a] = [r]
        # otherwise pick the best one thus far
        else:
            if (best_action == -1):
                # check for the lever with the best average payoff
                mean_dict = {}
                for key,val in pay_offs.items():
                    mean_dict[key] = np.mean(val) 
                best_action = max(mean_dict, key=mean_dict.get)
            a = best_action

            r = bandit.sample(a)
        
        yield a, r    

def epsilon_greedy_agent(bandit, iterations, epsilon = 0.2, initial_rounds = 1):
    """Use the epsilon-greedy algorithm by performing the action with the best average
    pay-off with the probability (1-epsilon), otherwise pick a random action to keep exploring."""

    pay_offs = dict()

    for i in range(iterations):
        # sometimes randomly pick an action to explore
        if random.random() < epsilon or i < initial_rounds:
            a = random.choice(bandit.actions)
        # otherwise pick the best one thus far
        else:
            # check for the lever with the best average payoff
            new_dict = {}
            for key,val in pay_offs.items():
                new_dict[key] = np.mean(val) 
            a = max(new_dict, key=new_dict.get)

        r = bandit.sample(a)

        #update rewards
        if a in pay_offs:
            pay_offs[a].append(r)
        else:
            pay_offs[a] = [r]
        
        yield a, r

def decaying_epsilon_greedy_agent(bandit, iterations, epsilon = 0.2, initial_rounds = 1, decay = 0.999):

    pay_offs = dict()

    for i in range(iterations):
        # sometimes randomly pick an action
        if random.random() < epsilon or i < initial_rounds:
            a = random.choice(bandit.actions)
        # otherwise pick the best one thus far
        else:
            # check for the lever with the best average payoff
            new_dict = {}
            for key,val in pay_offs.items():
                new_dict[key] = np.mean(val) 
            a = max(new_dict, key=new_dict.get)

        r = bandit.sample(a)

        #update rewards
        if a in pay_offs:
            pay_offs[a].append(r)
        else:
            pay_offs[a] = [r]
        
        epsilon *= decay

        yield a, r 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 380} id="tbwiPI4M3XgE" executionInfo={"status": "ok", "timestamp": 1617288720957, "user_tz": -330, "elapsed": 78228, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="35afcfbe-74ad-4a1b-ba9f-a23c66d5936b"
random.seed(200) #used for reproducibility

pay_offs = [0.25, 0.3, 0.5, 0.1, 0.3, 0.25, 0]
bandit = Bandit(pay_offs)

f = plt.figure()

methods = [random_agent, initial_explore_agent, epsilon_greedy_agent, decaying_epsilon_greedy_agent, optimal_agent]

number_of_iterations = 200
number_of_trials = 1000

for m in range(len(methods)):
    method = methods[m]
    total_rewards = []

    list_of_cumulative_rewards = []
    fan = []

    for trial in range(number_of_trials):
        total_reward = 0
        cumulative_reward = []

        for a, r in method(bandit, number_of_iterations):
            total_reward += r
            cumulative_reward.append(total_reward)


        #plt.plot(cumulative_reward, alpha=.02, color=colors[m])
        total_rewards.append(total_reward)

        if trial == 0:
            fan = pd.DataFrame(cumulative_reward, columns=['y'])
            fan['x'] = fan.index+1
        else:
            fan2 = pd.DataFrame(cumulative_reward, columns=['y'])
            fan2['x'] = fan2.index+1

            fan = fan.append(fan2, ignore_index=True)

        list_of_cumulative_rewards.append(cumulative_reward)

    sns.lineplot(x='x', y='y', data=fan)  #default is to use bootstrap to calculate confidence interval     
    
    print(method.__name__, ":", np.mean(total_rewards))

plt.title("Cumulative reward for each algorithm over {} iterations with {} trials.".format(number_of_iterations, number_of_trials))
plt.ylabel("Cumulative reward")
plt.xlabel("Iterations")
plt.legend([method.__name__ for method in methods])

f.savefig("Iterations.pdf", bbox_inches='tight')
f.savefig("Iterations.svg", bbox_inches='tight')

plt.show()
```
