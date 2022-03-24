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

<!-- #region id="prRPDqOQVl5L" -->
# Bandit-based online learning using Thompson sampling App
> Plotly dash app for thompson sampling simulation for product recommendations

- toc: true
- badges: true
- comments: true
- categories: [reinforcement learning, bandit, dash app]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6zt4vI__WSCR" outputId="73b6995f-5c15-45f4-b7c1-c02661085d04"
!pip install -q dash dash-html-components dash-core-components dash_bootstrap_components jupyter-dash
```

```python id="vHkx-wB7WBhe"
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import itertools
from copy import deepcopy

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import plotly.express as px
```

```python id="DsPNxaNdVeKW"
class THSimulationAdv():
    def __init__(self, nb_bandits):
        self.nb_bandits = nb_bandits
        self.trials = [0] * self.nb_bandits
        self.wins = [0] * self.nb_bandits
    def pull(self, i, p_bandits):
        if np.random.rand() < p_bandits[i]:
            return 1
        else:
            return 0
    def step(self, p_bandits):
        # Define the prior based on current observations
        bandit_priors = [stats.beta(a=1+w, b=1+t-w) for t, w in zip(self.trials, self.wins)]
        # Sample a probability theta for each bandit
        theta_samples = [d.rvs(1) for d in bandit_priors]
        # choose a bandit
        chosen_bandit = np.argmax(theta_samples)
        # Pull the bandit
        x = self.pull(chosen_bandit, p_bandits)
        # Update trials and wins (defines the posterior)
        self.trials[chosen_bandit] += 1
        self.wins[chosen_bandit] += x
        return self.trials, self.wins
```

```python id="fB3kVN_AVjpo"
n_bandits = 10
thsim = THSimulationAdv(nb_bandits=n_bandits)

app = JupyterDash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

white_button_style = {'background-color': 'white', 'color': 'black'}
green_button_style = {'background-color': 'green', 'color': 'white'}
red_button_style = {'background-color': 'red', 'color': 'white'}

def create_row(nb=1, wd=1, pb='0.5'):
    return dbc.Row(children=[
        dbc.Col(dbc.Input(id='bandit{}_prob'.format(str(nb)), type="number", min=0, max=1, 
                          step=0.01, value=pb), width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_hits'.format(str(nb))), color="success"),width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_miss'.format(str(nb))), color="danger"),width=wd),
        dbc.Col(dbc.Card(html.Div(id='bandit{}_total'.format(str(nb))), color="light"),width=wd),
    ], align="center", justify="start")

def create_table():
    row_list = [create_row(nb=i) for i in range(1,n_bandits+1)]
    return html.Div(row_list)

app.layout = html.Div(children=[
    dbc.Button("Start Simulation", color="primary"),
    create_table(),
    dcc.Interval(
            id='interval-component',
            interval=1000, # in milliseconds
            n_intervals=0
        ),
    html.Div(id='p_bandits'),
])

p_bandits = [np.random.rand() for i in range(n_bandits)]
last_update = thsim.step(p_bandits)

input_list = [eval(f"Input('bandit{i}_prob', 'value')") for i in range(1,n_bandits+1)]

@app.callback(
    Output('p_bandits', 'children'),
    input_list)
def update_probs(*args):
    global p_bandits
    p_bandits = [float(prob) for prob in args] 
    return ""

output_list_hits = [eval(f"Output('bandit{i}_hits', 'children')") for i in range(1,n_bandits+1)]
output_list_miss = [eval(f"Output('bandit{i}_miss', 'children')") for i in range(1,n_bandits+1)]
output_list_total = [eval(f"Output('bandit{i}_total', 'children')") for i in range(1,n_bandits+1)]
output_list = list(itertools.chain(output_list_hits,
                                   output_list_miss,
                                   output_list_total)
                  )

@app.callback(
    output_list,
    Input('interval-component', 'n_intervals'))
def update_metrics(n):
    x = thsim.step(p_bandits)
    totals = x[0]
    hits = x[1]
    global last_update
    hitlist=[]; misslist=[]; totallist=[]
    for i in range(n_bandits):
        hit_style = green_button_style if hits[i]!=last_update[1][i] else white_button_style
        miss_style = red_button_style if (totals[i]-hits[i])!=(last_update[0][i]-last_update[1][i]) else white_button_style
        hitlist.append(html.Div(hits[i], style=hit_style))
        misslist.append(html.Div(totals[i]-hits[i], style=miss_style))
        totallist.append(totals[i])
    last_update = deepcopy(x)
    return list(itertools.chain(hitlist,misslist,totallist))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 671} id="pBAbu51BWtCb" outputId="bc1c6689-5f05-4332-fead-4f51c5fe4f16"
app.run_server(mode='inline', port=8081)
```

```python id="vdnJaqmZXpom"
!kill -9 $(lsof -t -i:8081) # command to kill the dash once done
```
