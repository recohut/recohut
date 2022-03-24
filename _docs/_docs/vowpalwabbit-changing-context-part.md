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

<!-- #region id="RnXUndQf1Zqt" -->
# Contextual bandit with changing context part 2
> Customizing the context and changing it midway to see how fast the agent can adapt to the new context and start recommending better products as per the context

- toc: true
- badges: true
- comments: true
- categories: [contextual bandit]
- image: 
<!-- #endregion -->

```python id="NDf-kJiZ1WfJ"
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

# {v:k for k,v in mappings.items()}
```

```python id="ywUBRGWp1WfO"
from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
```

```python id="bLp9p35a1WfT"
users = list(mapping_users.values())
items = list(mapping_items.values())
context1 = list(mapping_context1.values())
context2 = list(mapping_context2.values())
```

```python id="bJPAR-pJ1WfW"
context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context))
context['cost'] = context['reward']*-1
contextdf = context.copy()
```

```python id="EklmWXvw1WfY" outputId="13a9bf54-8780-40ac-b8ee-6a1f7a7b8d63"
contextdf
```

```python id="FAOCxBU31Wfb"
import numpy as np
import scipy
import scipy.stats as stats
from vowpalwabbit import pyvw
import random
import pandas as pd
from itertools import product
```

```python id="ojr3s0vC1Wfc"
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

```python id="S2WXd8Xd1Wff"
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

```python id="z_33SgaP1Wfi" outputId="bc8f39ed-e666-40d8-f69d-58960bbba94c"
context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = np.random.choice([0,1],len(context),p=[0.8,0.2])
contextdf = context.copy()
contextdf.reward.value_counts()
```

```python id="hLYqC-B21Wfj"
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
vws = VWCSimulation(vw, contextdf)
```

```python id="iN7_K3-l1Wfl" outputId="382ec01f-22ba-45c2-ba2a-d181ac41db11"
vws.step()
```

```python id="GnEvCkub1Wfm"
_temp = []
for i in range(5000):
    _temp.append(vws.step())
```

```python id="IykFBA931Wfn"
x = pd.DataFrame.from_records(_temp, columns=['user','context1','context2','item','cost','prob'])
```

```python id="-36YdGnH1Wfn" outputId="3adfdb67-ec48-4a0e-bfd9-861dd5be98d2"
xx = x.copy()
xx['ccost'] = xx['cost'].cumsum()
xx = xx.fillna(0)
xx = xx.rename_axis('iter').reset_index()
xx['ctr'] = -1*xx['ccost']/xx['iter']
xx.sample(10)
```

```python id="X1Aa6gjX1Wfo" outputId="5b62b160-8a68-4b61-f1f2-28f524737614"
xx['ccost'].plot()
```

```python id="L4MXTt071Wfo" outputId="4d5ae04a-a638-4b3a-8f98-eef32ede93f4"
xx['ctr'].plot()
```

```python id="M6WRP_0P1Wfp"
tempdf1 = xx.copy()
```

```python id="ukACuqoc1Wfq" outputId="f6941465-fc4c-40ff-8a7c-84f8e104366e"
context = pd.DataFrame(list(product(users, context1, context2, items)), columns=['users', 'context1', 'context2', 'items'])
context['reward'] = 0
X = context.copy()
X.loc[(X['users']=='usera')&(X['items']=='item1'),'reward']=1
X.loc[(X['users']=='userb')&(X['items']=='item2'),'reward']=1
X.loc[(X['users']=='userc')&(X['items']=='item3'),'reward']=1
X.reward.value_counts()
```

```python id="NwVd7J4C1Wfq"
vws.update_context(X)
```

```python id="nqs6J8sy1Wfr"
_temp = []
for i in range(5000):
    _temp.append(vws.step())
```

```python id="JV3H9J4N1Wfr" outputId="5f675cad-76e7-42eb-b769-bd2b896cca7f"
x = pd.DataFrame.from_records(_temp, columns=['user','context1','context2','item','cost','prob'])
xx = x.copy()
xx['ccost'] = xx['cost'].cumsum()
xx = xx.fillna(0)
xx = xx.rename_axis('iter').reset_index()
xx['ctr'] = -1*xx['ccost']/xx['iter']
xx.sample(10)
```

```python id="6-2NfwBF1Wfs" outputId="5b89213a-e4f0-48a2-af31-cf79277a650a"
tempdf2 = tempdf1.append(xx, ignore_index=True)
tempdf2.sample(10)
```

```python id="28SHrzVq1Wfs" outputId="16959443-416f-4fbe-f0e7-e331e324fc93"
tempdf2['ccost'].plot()
```

```python id="2Nq8ZhSr1Wft" outputId="a9f12adf-13da-416b-e51c-dc66bdac8c59"
tempdf2['ctr'].plot()
```
