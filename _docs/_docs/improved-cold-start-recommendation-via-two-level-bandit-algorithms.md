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

<!-- #region id="ygdTFp2lO9pN" -->
# Improved Cold-Start Recommendation via Two-Level Bandit Algorithms
<!-- #endregion -->

```python id="QB2fReibVwhA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635755416264, "user_tz": -330, "elapsed": 4046, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="97ece3aa-1702-49b2-ed5a-821bcdd61843"
!pip install xlutils
```

<!-- #region id="ZtGT77eRP0Rf" -->
## Imports
<!-- #endregion -->

```python id="bbf2p5AhO9t7" executionInfo={"status": "ok", "timestamp": 1635755416265, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import math
from collections import defaultdict
from numpy.random import beta
from operator import itemgetter
import numpy as np
import random
from random import betavariate
from scipy.special import btdtri
import sys
```

<!-- #region id="Yd_XDHMfRtm2" -->
## Utils
<!-- #endregion -->

<!-- #region id="mLwxEjFSRumD" -->
### Posterior
<!-- #endregion -->

```python id="-OR7n3LaRukA" executionInfo={"status": "ok", "timestamp": 1635755423767, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Posterior:
    ''' Generic class for posteriors, empty for the time being''' 
    def __init__(self): pass
```

<!-- #region id="8Nu0K9rCRuhE" -->
### Beta
<!-- #endregion -->

```python id="XEnEg-49R5Qe" executionInfo={"status": "ok", "timestamp": 1635755424373, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Beta(Posterior):
    """Manipulate posteriors of Bernoulli/Beta experiments.
    """
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        
    def reset(self, a=0, b=0):
        if a==0:
            a = self.a
        if b==0:
            b = self.b
        self.N = [a, b]

    def update(self, obs):
        self.N[int(obs)] += 1
        
    def sample(self):
        return betavariate(self.N[1], self.N[0])

    def quantile(self, p):
        return btdtri(self.N[1], self.N[0], p) # Bug: do not call btdtri with (0.5,0.5,0.5) in scipy < 0.9
```

<!-- #region id="dHRZgyQkO9wD" -->
## Random Choice
<!-- #endregion -->

```python id="rYwgueJtP7Az" executionInfo={"status": "ok", "timestamp": 1635755425350, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class RandomChoice(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms
        self.values = [0.] * n_arms
        self.flag_set = True

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Random Choice not set. Aborting')
            import sys
            sys.exit(1)

        """Choose an random arm """
        # Explore (test all arms)
        return np.random.randint(self.n)

    def update(self, choosen_arm, reward):
        if self.flag_set == False:
            print('Error: Random Choice not set. Aborting')
            import sys
            sys.exit(1)

        """Update an arm with some reward value"""
        self.counts[choosen_arm] = self.counts[choosen_arm] + 0.1
        n = self.counts[choosen_arm]
        value = np.random.randint(self.n) #self.values[choosen_arm]
        
        # Running product
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[choosen_arm] = new_value
        return
```

```python id="sKRAJkoRP6-Q" executionInfo={"status": "ok", "timestamp": 1635755425351, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class RandomChoiceLevel2(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms
        self.values = [0.] * n_arms
        self.flag_set = True

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Random Choice not set. Aborting')
            import sys
            sys.exit(1)

        """Choose an random arm """
        # Explore (test all arms)
        return np.random.randint(self.n)

    def update(self, choosen_arm, reward):
        if self.flag_set == False:
            print('Error: Random Choice not set. Aborting')
            import sys
            sys.exit(1)

        """Update an arm with some reward value"""
        self.counts[choosen_arm] = self.counts[choosen_arm] + 0.1
        n = self.counts[choosen_arm]
        value = np.random.randint(self.n) #self.values[choosen_arm]
        
        # Running product
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[choosen_arm] = new_value
        return
```

<!-- #region id="dDUi0DsAROeS" -->
## Epsilon greedy
<!-- #endregion -->

```python id="tBzeuzBeROcg" executionInfo={"status": "ok", "timestamp": 1635755426112, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class EpsilonGreedy(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms   # number of likes
        self.values = [0.] * n_arms  # number of likes
        self.flag_set = True

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Epsilon-greedy not set. Aborting')
            import sys
            sys.exit(1)

        """Choose an arm for testing"""
        epsilon = 0.2
        if np.random.random() > epsilon:
            # Exploit (use best arm)
            return np.argmax(self.values)
        else:
            # Explore (test all arms)
            return np.random.randint(self.n)

    def update(self, arm, reward):
        if self.flag_set == False:
            print('Error: Epsilon-greedy not set. Aborting')
            import sys
            sys.exit(1)

        """Update an arm with some reward value"""  # Example: like = 1; no like = 0
        #print(arm, type(arm))
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        # Running product
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        return
```

```python id="2qBMmX3vROZJ" executionInfo={"status": "ok", "timestamp": 1635755426113, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class EpsilonGreedyLevel2(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms   # number of likes
        self.values = [0.] * n_arms  # number of likes
        self.flag_set = True

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Epsilon-greedy not set. Aborting')
            import sys
            sys.exit(1)

        """Choose an arm for testing"""
        epsilon = 1.0
        if np.random.random() > epsilon:
            # Exploit (use best arm)
            return np.argmax(self.values)
        else:
            # Explore (test all arms)
            return np.random.randint(self.n)

    def update(self, arm, reward):
        if self.flag_set == False:
            print('Error: Epsilon-greedy not set. Aborting')
            import sys
            sys.exit(1)

        """Update an arm with some reward value"""  # Example: like = 1; no like = 0
        #print(arm, type(arm))
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        # Running product
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        return
```

<!-- #region id="qo4if-i3O906" -->
## UCB
<!-- #endregion -->

```python id="T5rhtif7O93O" executionInfo={"status": "ok", "timestamp": 1635755426114, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def ind_max(x):
    m = max(x)
    return x.index(m)

class UCB1(object):
    def __init__(self): #, counts, values):
        #self.counts = counts
        #self.values = values
        #return
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.flag_set = True
        return

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
```

```python id="kCo5AwKXO947" executionInfo={"status": "ok", "timestamp": 1635755426114, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def ind_max(x):
    m = max(x)
    return x.index(m)

class UCB1level2(object):
    def __init__(self): #, counts, values):
        #self.counts = counts
        #self.values = values
        #return
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.flag_set = True
        return

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
```

```python id="B2wNNwqgO97l" executionInfo={"status": "ok", "timestamp": 1635755426115, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def ind_max(x):
    m = max(x)
    return x.index(m)

class UCB2(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.alpha = 0.5
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.r = [0 for col in range(n_arms)]
        self.__current_arm = 0
        self.__next_update = 0
        self.flag_set = True
        return

    def __bonus(self, n, r):
        tau = self.__tau(r)
        bonus = math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / tau) / (2 * tau))
        return bonus

    def __tau(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def __set_arm(self, arm):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.__current_arm = arm
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        n_arms = len(self.counts)
        # play each arm once
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                self.__set_arm(arm)
                return arm

        # make sure we aren't still playing the previous arm.
        if self.__next_update > sum(self.counts):
            return self.__current_arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in xrange(n_arms):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.values[arm] + bonus

        chosen_arm = ind_max(ucb_values)
        self.__set_arm(chosen_arm)
        return chosen_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
```

```python id="y-C17esRPS5v" executionInfo={"status": "ok", "timestamp": 1635755426116, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def ind_max(x):
    m = max(x)
    return x.index(m)

class UCB2Level2(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.n = n_arms
        self.alpha = 0.5
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.r = [0 for col in range(n_arms)]
        self.__current_arm = 0
        self.__next_update = 0
        self.flag_set = True
        return

    def __bonus(self, n, r):
        tau = self.__tau(r)
        bonus = math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / tau) / (2 * tau))
        return bonus

    def __tau(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def __set_arm(self, arm):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.__current_arm = arm
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        n_arms = len(self.counts)
        # play each arm once
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                self.__set_arm(arm)
                return arm

        # make sure we aren't still playing the previous arm.
        if self.__next_update > sum(self.counts):
            return self.__current_arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in xrange(n_arms):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.values[arm] + bonus

        chosen_arm = ind_max(ucb_values)
        self.__set_arm(chosen_arm)
        return chosen_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: UCB not set. Aborting')
            import sys
            sys.exit(1)

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
```

<!-- #region id="OMq33iqSRoJ8" -->
## Bayes UCB
<!-- #endregion -->

```python id="o1PhQQgcRoHc" executionInfo={"status": "ok", "timestamp": 1635755427475, "user_tz": -330, "elapsed": 777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class BayesUCB():
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.posterior_dist = Beta
        self.t         = 1.0
        self.arms      = n_arms
        self.posterior = defaultdict(lambda: None)
        for arm_id in range(self.arms):
            self.posterior[arm_id] = self.posterior_dist()
        for arm_id in range(self.arms):
            self.posterior[arm_id].reset()
        self.flag_set = True
        return

    def compute_index(self, arm_id):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)

        return self.posterior[arm_id].quantile(1 - (1. / self.t))

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)
        
        index = dict()
        for arm_id in range(self.arms):
            index[arm_id] = self.compute_index(arm_id)
        best_arm = np.argmax(index.values())
        #best_arm_id = [arm_id for arm_id in range(len(index.keys())) if index[arm_id] == best_arm][0]
        return best_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)

        if chosen_arm not in range(self.arms):
            print('Error in BayesUCB. Invalid chosen arm. Aborting.')
            sys.exit(1)

        self.posterior[chosen_arm].update(reward)
        self.t += 1
        return
```

```python id="BggBkJXnRoEu" executionInfo={"status": "ok", "timestamp": 1635755427476, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class BayesUCBlevel2():
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.posterior_dist = Beta
        self.t         = 1.0
        self.arms      = n_arms
        self.posterior = defaultdict(lambda: None)
        for arm_id in range(self.arms):
            self.posterior[arm_id] = self.posterior_dist()
        for arm_id in range(self.arms):
            self.posterior[arm_id].reset()
        self.flag_set = True
        return

    def compute_index(self, arm_id):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)

        return self.posterior[arm_id].quantile(1 - (1. / self.t))

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)
        
        index = dict()
        for arm_id in range(self.arms):
            index[arm_id] = self.compute_index(arm_id)
        best_arm = np.argmax(index.values())
        #best_arm_id = [arm_id for arm_id in range(len(index.keys())) if index[arm_id] == best_arm][0]
        return best_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: BayesUCB not set. Aborting')
            import sys
            sys.exit(1)

        if chosen_arm not in range(self.arms):
            print('Error in BayesUCB. Invalid chosen arm. Aborting.')
            sys.exit(1)

        self.posterior[chosen_arm].update(reward)
        self.t += 1
        return
```

<!-- #region id="wnvexRGCO9yT" -->
## Thompson Sampling
<!-- #endregion -->

```python id="4Du_KbUwPoxh" executionInfo={"status": "ok", "timestamp": 1635755427477, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ThompsonSampling(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.param_alpha   = defaultdict(lambda: None)
        self.param_beta    = defaultdict(lambda: None)
        self.num_successes = defaultdict(lambda: None)
        self.num_fails     = defaultdict(lambda: None)
        #self.counts = [0 for col in range(n_arms)]
        for arm in range(n_arms):
            self.param_alpha[arm]   = 1.0
            self.param_beta[arm]    = 1.0
            self.num_successes[arm] = 0.0
            self.num_fails[arm]     = 0.0
        self.n_arms = n_arms

        self.counts = defaultdict(lambda: None)
        for arm in range(self.n_arms):
            self.counts[arm] = 0
        self.flag_set = True
        return

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Thompson Sampling not set. Aborting')
            import sys
            sys.exit(1)

        scores = [(arm_id, beta(self.num_successes[arm_id] + self.param_alpha[arm_id],
                        self.num_fails[arm_id] + self.param_beta[arm_id]))
                        for arm_id in range(self.n_arms)]
        
        scores = sorted(scores, key=itemgetter(0))
        ranking = sorted(scores, key=itemgetter(1), reverse=True)
        selected_arm = ranking[0][0]
        return selected_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: Thompson Sampling not set. Aborting')
            import sys
            sys.exit(1)		

        if chosen_arm not in range(self.n_arms):
            print('--- self.n_arms')
            print(self.n_arms)
            print('--- chosen arm')
            print(chosen_arm)
            print('Error in thompson sampling. Invalid chosen arm. Aborting')
            sys.exit(1)

        if reward == 1.0:
            self.num_successes[chosen_arm] += 1.0
        elif reward == 0:
            self.num_fails[chosen_arm] += 1.0
        else:
            print('Error in thompson sampling. Invalid reward. Aborting')
            sys.exit(1)
        return
```

```python id="54InJVriPovn" executionInfo={"status": "ok", "timestamp": 1635755427478, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ThompsonSamplingLevel2(object):
    def __init__(self):
        self.flag_set = False

    def set_arms(self, n_arms):
        self.param_alpha   = defaultdict(lambda: None)
        self.param_beta    = defaultdict(lambda: None)
        self.num_successes = defaultdict(lambda: None)
        self.num_fails     = defaultdict(lambda: None)
        #self.counts = [0 for col in range(n_arms)]
        for arm in range(n_arms):
            self.param_alpha[arm]   = 1.0
            self.param_beta[arm]    = 1.0
            self.num_successes[arm] = 0.0
            self.num_fails[arm]     = 0.0
        self.n_arms = n_arms

        self.counts = defaultdict(lambda: None)
        for arm in range(self.n_arms):
            self.counts[arm] = 0
        self.flag_set = True
        return

    def choose_arm(self):
        if self.flag_set == False:
            print('Error: Thompson Sampling not set. Aborting')
            import sys
            sys.exit(1)

        scores = [(arm_id, beta(self.num_successes[arm_id] + self.param_alpha[arm_id],
                        self.num_fails[arm_id] + self.param_beta[arm_id]))
                        for arm_id in range(self.n_arms)]
        
        scores = sorted(scores, key=itemgetter(0))
        ranking = sorted(scores, key=itemgetter(1), reverse=True)
        selected_arm = ranking[0][0]
        return selected_arm

    def update(self, chosen_arm, reward):
        if self.flag_set == False:
            print('Error: Thompson Sampling not set. Aborting')
            import sys
            sys.exit(1)		

        if chosen_arm not in range(self.n_arms):
            print('--- self.n_arms')
            print(self.n_arms)
            print('--- chosen arm')
            print(chosen_arm)
            print('Error in thompson sampling. Invalid chosen arm. Aborting')
            sys.exit(1)

        if reward == 1.0:
            self.num_successes[chosen_arm] += 1.0
        elif reward == 0:
            self.num_fails[chosen_arm] += 1.0
        else:
            print('Error in thompson sampling. Invalid reward. Aborting')
            sys.exit(1)
        return
```

<!-- #region id="CWqhk35JPS3n" -->
## Policy Evaluator
<!-- #endregion -->

```python id="frY9NjPTPS0j" executionInfo={"status": "ok", "timestamp": 1635755427478, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import copy
from collections import Counter
from collections import defaultdict
import sys
import numpy as np
import os
```

```python id="4EWl5tvkTzyz" executionInfo={"status": "ok", "timestamp": 1635755427479, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class select_arm(object):
    def __init__(self, cluster, ncluster):
        self.cluster   = cluster
        self.ncluster  = ncluster
        self.mab_alg   = copy.copy(BayesUCBlevel2())
        self.log_file  = 'Movielens LDA/clusters/%s/movie_cluster_%s.txt' %(self.ncluster, self.cluster)
        self.idvideo2idarm = defaultdict(lambda: None)
        self.idarm2idvideo = defaultdict(lambda: None)
        self.contidarm = 0
        self.before    = defaultdict(lambda: None)

        if os.stat(self.log_file).st_size == 0:
            self.contidarm = 1
        
        #print('Policy evaluator started with log file %s' % (self.log_file))

        self.count_itens = defaultdict(lambda: 0)
        intro_file = open(self.log_file, 'r')
        logs = intro_file.readlines()
        logs2 = logs
        intro_file.close()
        for line in logs:
            arm_id, inf_value = line.strip().split(',')
            self.count_itens[arm_id] += 1
        
        for line in logs2:
            arm_id, inf_value = line.strip().split(',')
            if arm_id not in self.idvideo2idarm:
                self.idvideo2idarm[arm_id] = self.contidarm
                self.idarm2idvideo[self.contidarm] = arm_id
                self.contidarm += 1

        #print('Selected videos %s' % self.contidarm)
        self.mab_alg.set_arms(self.contidarm)
        self.before = self.mab_alg.choose_arm()
        self.count_n = 0		

    def get_video(self):
        self.mab_alg.set_arms(self.contidarm)
        recommendation = self.mab_alg.choose_arm()
        tmp_id_arm     = self.idarm2idvideo[recommendation]
        #print('Selected arm:', tmp_id_arm)
        if (self.before == recommendation):
            reward = 1.0
            self.count_n += 1
        else:
            reward = 0.0
        self.mab_alg.update(recommendation, reward)
        self.before = recommendation
        return tmp_id_arm
```

```python id="X4LJxIj8TzwX"
'''
#TEST
x=2
num_model=100
#mab_alg = copy.copy(RandomChoiceLevel2())
pegararm = select_arm(x,num_model)
video = pegararm.get_video()
print(video)'''
```

```python id="6VPYGBLNTzty" executionInfo={"status": "ok", "timestamp": 1635755428053, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from collections import Counter
from collections import defaultdict
from xlrd import open_workbook
from xlutils.copy import copy
import matplotlib.pyplot as plt
import sys
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="0-IHRrJdp22Y" executionInfo={"status": "ok", "timestamp": 1635756060881, "user_tz": -330, "elapsed": 58527, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="76da1380-146b-4ea5-a1d7-5af68be73b6a"
!git clone https://github.com/OtavioAugusto/RecSys.git
%cd RecSys/src
```

```python colab={"base_uri": "https://localhost:8080/"} id="unaoEUJzq7W2" executionInfo={"status": "ok", "timestamp": 1635756304520, "user_tz": -330, "elapsed": 1293, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a256ab68-fdcb-4c99-bcbd-dd1e424d9c8d"
!wget -q --show-progress -O vids_very_ratings.txt https://raw.githubusercontent.com/OtavioAugusto/RecSys/master/src/Very%20reward.txt
!wget -q --show-progress -O vids_less_ratings.txt https://raw.githubusercontent.com/OtavioAugusto/RecSys/master/src/Low%20reward.txt
```

```python id="Vp6Ma2PzV-mk" executionInfo={"status": "ok", "timestamp": 1635756318736, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class PolicyEvaluator(object):
    def __init__(self, context):#, num_runs):
        self.context           = context
        self.mab_alg           = context['mab']
        self.log_file  	       = context['log_file']
        self.column	       = context['column']
        self.ncluster           = context['cluster']
        #self.num_runs          = num_runs

        self.cumulative_reward = [ ]
        '''self.idcluster2idarm     = defaultdict(lambda: None)
        self.idarm2idcluster     = defaultdict(lambda: None)'''
        self.idvideo2idarm = defaultdict(lambda: None)
        self.idarm2idvideo = defaultdict(lambda: None)
        self.contidarm         = 0
        self.contidarm = self.ncluster
        print('Selected clusters %s' % self.contidarm)
        self.mab_alg.set_arms(self.contidarm)

    def run(self):
        tmp_result = []
        rme_result = []
        rle_result = []
        self.mab_alg.set_arms(self.contidarm)
        self.cumulative_reward = [ ]
        
        #Verifying most and less ratings
        me = []
        le = []
        rat = open('vids_very_ratings.txt', 'r')
        lrat = open('vids_less_ratings.txt', 'r')
        acum = rat.readlines()
        acum1 = lrat.readlines()
        rat.close()
        lrat.close()
        #
            
        count_n = 0.0
        intro_file = open(self.log_file, 'r')
        logs = intro_file.readlines()
        intro_file.close()
        for line in logs:
            user_id, item_id = line.strip().split(',')
            cluster = self.mab_alg.choose_arm()
            get_arm = select_arm(cluster, self.ncluster)
            recommendation = get_arm.get_video()
            '''recommendation = self.mab_alg.choose_arm()
                        tmp_id_arm     = self.idarm2idvideo[recommendation]
            if (item_id == tmp_id_arm):'''
            if (item_id == recommendation):
                reward = 1.0
                count_n += 1
                for line in acum: #verifying if video is in most or less ratings
                    item_r = line.strip()								
                    if recommendation == item_r: 
                        rme = 1.0
                        me.append(rme)
                for line in acum1:
                    item_l = line.strip()
                    if recommendation == item_l:
                        rle = 1.0
                        le.append(rle)
            else:
                reward = 0.0
            self.mab_alg.update(cluster, reward)
            #self.mab_alg.update(recommendation, reward)
            self.cumulative_reward.append(reward)
        rme_result.append([sum(me[0:i]) for i, value in enumerate(me)])
        rle_result.append([sum(le[0:i]) for i, value in enumerate(le)])
        tmp_result.append([sum(self.cumulative_reward[0:i]) for i, value in enumerate(self.cumulative_reward)])
        mean_tmp_result = np.mean(tmp_result, axis=0)
        std_tmp_result = np.std(tmp_result, axis=0)
        final = open('Movielens LDA/clusters/Arrays/BayesUCB/reward_%s_clusters_test'%(self.ncluster),'w')
        final.write(str(tmp_result))
        final.close()
        veryrew = open('Very reward.txt','w')
        veryrew.write(str(rme_result))
        veryrew.close()
        lowrew = open('Low reward.txt','w')
        lowrew.write(str(rle_result))
        lowrew.close()

        # Open an Excel file and add a worksheet.
        rb = open_workbook("Results.xls")
        wb = copy(rb)
        # Write text in cells.
        worksheet = wb.get_sheet(0) #Pay attention for this
        worksheet.write(21, self.column, str(round(np.mean(mean_tmp_result))) + ' - ' + str(np.std(std_tmp_result)))
        wb.save('Results.xls')

        rc = open_workbook("Ratings_frequency.xls")
        wc = copy(rc)
        # Write text in cells.
        wks = wc.get_sheet(0) #Pay attention for this
        wks.write(2, 1, 'Muitas avaliacoes: ' + str(round(np.mean(rme_result))) +
                'Poucas avaliacoes: ' + str(round(np.mean(rle_result))))
        wc.save('Ratings_frequency.xls')
```

```python id="MRBV3H4DWz_i" executionInfo={"status": "ok", "timestamp": 1635755954650, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import copy
```

```python colab={"base_uri": "https://localhost:8080/", "height": 391} id="PVhA6mVAO99X" executionInfo={"status": "error", "timestamp": 1635763442831, "user_tz": -330, "elapsed": 7122365, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a0e4a21b-33df-41ef-c377-773234daf058"
#num_runs = 1
cell = 1
num_cluster = [5]#[5,10,17,25,50,100]

for x in num_cluster:
	context = {
	'mab'          : copy.copy(BayesUCB()),
	'log_file'     : 'log.txt',
	'column'       : cell,
	'cluster'      : x
	}
	#print x, j
	if __name__ == '__main__':
		evaluator = PolicyEvaluator(context)#, num_runs)
		evaluator.run()
	cell += 1
```

```python id="KeItmV3TWxVO"

```
