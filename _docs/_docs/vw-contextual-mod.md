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

```python colab={"base_uri": "https://localhost:8080/"} id="7SCDe05G1QSx" executionInfo={"status": "ok", "timestamp": 1620185578973, "user_tz": -330, "elapsed": 9152, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79b0af8b-de7d-4b35-909c-66990c44fad4"
!pip install vowpalwabbit
```

```python colab={"base_uri": "https://localhost:8080/"} id="njhsjoqcSq3_" executionInfo={"status": "ok", "timestamp": 1620231073879, "user_tz": -330, "elapsed": 2707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afcf7f99-6177-47d8-b259-e1a0cdf9675b"
!zip -r app.zip /content/dash-sample-apps/apps/dash-clinical-analytics
```

```python id="U17IKmm51Sa_" executionInfo={"status": "ok", "timestamp": 1620186109593, "user_tz": -330, "elapsed": 1945, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
```

```python id="AFdEwthO1U6N" executionInfo={"status": "ok", "timestamp": 1620185581218, "user_tz": -330, "elapsed": 2191, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0
```

```python id="Joz5em0y7kfi"
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

```python id="_S9shNn-liOP" executionInfo={"status": "ok", "timestamp": 1620191384481, "user_tz": -330, "elapsed": 1733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="rpgSzAFjv_Rh" executionInfo={"status": "ok", "timestamp": 1620191384483, "user_tz": -330, "elapsed": 938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29a5d4c3-92f6-4ee4-f56f-31752de3c4b0"
contextdf.cost.value_counts()
```

```python id="Z8Uuu0axyu-C"
# def get_cost(context,action):
#     if context['user'] == "Tom":
#         if context['time_of_day'] == "morning" and action == 'politics':
#             return USER_LIKED_ARTICLE
#         elif context['time_of_day'] == "afternoon" and action == 'music':
#             return USER_LIKED_ARTICLE
#         else:
#             return USER_DISLIKED_ARTICLE
#     elif context['user'] == "Anna":
#         if context['time_of_day'] == "morning" and action == 'sports':
#             return USER_LIKED_ARTICLE
#         elif context['time_of_day'] == "afternoon" and action == 'politics':
#             return USER_LIKED_ARTICLE
#         else:
#             return USER_DISLIKED_ARTICLE
```

```python id="QlOd83hd1Z3F" executionInfo={"status": "ok", "timestamp": 1620191846746, "user_tz": -330, "elapsed": 671, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_cost(context,action):
    return contextdf.loc[(contextdf['users']==context['user']) & \
            (contextdf.context1==context['context1']) & \
            (contextdf.context2==context['context2']) & \
            (contextdf['items']==action), \
            'cost'].values[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="6P_HYhhW8UWO" executionInfo={"status": "ok", "timestamp": 1620191848461, "user_tz": -330, "elapsed": 1576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="74299b34-3a40-4141-df03-76edf4b56edf"
get_cost({'user':'A','context1':'morning','context2':'summer'},'Item2')
```

```python id="ub-6-Y5P1bhd" executionInfo={"status": "ok", "timestamp": 1620156279124, "user_tz": -330, "elapsed": 1383, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# # This function modifies (context, action, cost, probability) to VW friendly format
# def to_vw_example_format(context, actions, cb_label = None):
#     if cb_label is not None:
#         chosen_action, cost, prob = cb_label
#     example_string = ""
#     example_string += "shared |User user={} time_of_day={}\n".format(context["user"], context["time_of_day"])
#     for action in actions:
#         if cb_label is not None and action == chosen_action:
#             example_string += "0:{}:{} ".format(cost, prob)
#         example_string += "|Action article={} \n".format(action)
#     #Strip the last newline
#     return example_string[:-1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="onZzvDqJ1uNW" executionInfo={"status": "ok", "timestamp": 1620156324184, "user_tz": -330, "elapsed": 863, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef1a60f4-5acc-4ef9-dfde-a1e9f1b64fde"
# context = {"user":"Tom","time_of_day":"morning"}
# actions = ["politics", "sports", "music", "food"]

# print(to_vw_example_format(context,actions))
```

```python id="pTCHTNFf3jRe" executionInfo={"status": "ok", "timestamp": 1620191461983, "user_tz": -330, "elapsed": 1313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/"} id="k8dLifwF3eDJ" executionInfo={"status": "ok", "timestamp": 1620191461985, "user_tz": -330, "elapsed": 802, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="54172834-13fb-4e48-a228-daf9a5ec0680"
context = {"user":"A","context1":"morning","context2":"summer"}

print(to_vw_example_format(context,item))
```

```python id="dM_aaYyb5xe9" executionInfo={"status": "ok", "timestamp": 1620191478597, "user_tz": -330, "elapsed": 1590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="1xeAG23o2EGj" executionInfo={"status": "ok", "timestamp": 1620191478599, "user_tz": -330, "elapsed": 1413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob
```

```python id="VWsWZZWgjVZR"

```

```python id="az_BHiJ32EwX" executionInfo={"status": "ok", "timestamp": 1620191478600, "user_tz": -330, "elapsed": 1293, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def choose_user(users):
    return random.choice(users)

def choose_context1(context1):
    return random.choice(context1)

def choose_context2(context2):
    return random.choice(context2)
```

```python id="XqWD3pqt2GTr" executionInfo={"status": "ok", "timestamp": 1620192002733, "user_tz": -330, "elapsed": 1076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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
```

```python id="fPDQ5H512KEj" executionInfo={"status": "ok", "timestamp": 1620192004222, "user_tz": -330, "elapsed": 813, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def plot_ctr(num_iterations, ctr):
    plt.plot(range(1,num_iterations+1), ctr)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('ctr', fontsize=14)
    plt.ylim([0,1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="wPhRpLyo2MmF" executionInfo={"status": "ok", "timestamp": 1620192011595, "user_tz": -330, "elapsed": 8030, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5b04514-516a-4c4f-f05c-f6437d3b23de"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost)

plot_ctr(num_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="TL81VOqK2Tbq" executionInfo={"status": "ok", "timestamp": 1620192585543, "user_tz": -330, "elapsed": 7245, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1171c8f1-11f3-41bc-ad97-a3e518693896"
# Instantiate learner in VW but without -q
vw = pyvw.vw("--cb_explore_adf --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost)

plot_ctr(num_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="2vU5Kozv2WDg" executionInfo={"status": "ok", "timestamp": 1620192610855, "user_tz": -330, "elapsed": 6551, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3dae15ae-38e8-418b-dd1b-37e78ce6f828"
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
ctr = run_simulation(vw, num_iterations, users, context1, context2, items, get_cost, do_learn=False)

plot_ctr(num_iterations, ctr)
```

```python id="UzavSLvC2ZHm" executionInfo={"status": "ok", "timestamp": 1620192940933, "user_tz": -330, "elapsed": 1344, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python id="Mqvn5gLc2ik2" executionInfo={"status": "ok", "timestamp": 1620193132066, "user_tz": -330, "elapsed": 1346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
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

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="ExchBfTr2lBS" executionInfo={"status": "ok", "timestamp": 1620193222170, "user_tz": -330, "elapsed": 13076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c23545d-652c-4deb-971c-8da1b409fbd9"
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

ctr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, context1, context2, items, cost_functions)

plot_ctr(total_iterations, ctr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="mgaEviR42mlI" executionInfo={"status": "ok", "timestamp": 1620193274550, "user_tz": -330, "elapsed": 12387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="00ccc48f-f222-4a58-fe14-d8675d4b13cc"
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

```python id="4EEJ473k2xwx"

```
