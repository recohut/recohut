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

<!-- #region id="OG8zBtp4CCs2" -->
# Efficient Frontier
<!-- #endregion -->

<!-- #region id="RDiDCFPNBkrF" -->
**Scenario: Portfolio optimization**

> Efficient frontier is the Nobel Prize Winner Theory To Gain Higher Returns In Your Investment.

Let’s consider you have \$10'000 of cash available and you are interested in investing it. Your aim is to invest the money for a year. Like any rational investor, you expect the final amount in a years time to be higher than the $10'000 amount you want to invest.

There are many investment options available, such as buying a T-bill or company shares, etc. Some of the investment options are riskier than the others because they attract us to gain higher returns. Hence, the point to note is that there exists a risk-return trade-off.

If we buy a number of assets such as shares of different companies then the total risk of the portfolio can be reduced due to diversification. This means that an investor can reduce the total risk and increase the return by choosing different assets with different proportions in a portfolio. This is due to the fact that the assets can be correlated with each other.

We understood that the allocations (weights) of the assets can change the risk of the portfolio. Hence, we can generate 1000s of portfolios randomly where each portfolio will contain a different set of weights for the assets.

We know that as we increase the number of portfolios, we will get closer to the real optimum portfolio. This is the brute force approach and it can turn out to be a time-consuming task. Furthermore, there is no guarantee that we will find the right allocations.
<!-- #endregion -->

<!-- #region id="2Zu75MCKB241" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8C3mduG_bweX" executionInfo={"status": "ok", "timestamp": 1634669194904, "user_tz": -330, "elapsed": 1133, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="093ee617-ce81-4ffd-ccbc-ec76f0495e1d"
!wget -q --show-progress https://github.com/rian-dolphin/Efficient-Frontier-Python/raw/main/daily_returns.csv
```

```python id="5mcYJszs4GTw"
import pandas as pd
import numpy as np

from tqdm import tqdm

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="qnbIYF5m4NOx" executionInfo={"status": "ok", "timestamp": 1634669223469, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="611d033c-0d3c-4b73-af77-c7054cab1045"
daily_returns = pd.read_csv('daily_returns.csv', index_col=0)
daily_returns.head()
```

```python id="awEbz0Ug4NaH"
#-- Get annualised mean returns
mus = (1+daily_returns.mean())**252 - 1

#-- Get covariances
#- Multiply by 252 to annualise it (square root time for volatility but no square root for variance)
#- Note: 252 trading days in a year
#- https://quant.stackexchange.com/questions/4753/annualized-covariance
cov = daily_returns.cov()*252
```

<!-- #region id="NEX2VuJ74R2U" -->
## Create Random Portfolios
<!-- #endregion -->

```python id="HIqTqK8P4ZfP"
#- How many assests to include in each portfolio
n_assets = 5
#-- How many portfolios to generate
n_portfolios = 1000

#-- Initialize empty list to store mean-variance pairs for plotting
mean_variance_pairs = []

np.random.seed(75)
#-- Loop through and generate lots of random portfolios
for i in range(n_portfolios):
    #- Choose assets randomly without replacement
    assets = np.random.choice(list(daily_returns.columns), n_assets, replace=False)
    #- Choose weights randomly
    weights = np.random.rand(n_assets)
    #- Ensure weights sum to 1
    weights = weights/sum(weights)

    #-- Loop over asset pairs and compute portfolio return and variance
    #- https://quant.stackexchange.com/questions/43442/portfolio-variance-explanation-for-equation-investments-by-zvi-bodie
    portfolio_E_Variance = 0
    portfolio_E_Return = 0
    for i in range(len(assets)):
        portfolio_E_Return += weights[i] * mus.loc[assets[i]]
        for j in range(len(assets)):
            #-- Add variance/covariance for each asset pair
            #- Note that when i==j this adds the variance
            portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]
            
    #-- Add the mean/variance pairs to a list for plotting
    mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="jJmqt1lV4W7X" executionInfo={"status": "ok", "timestamp": 1634669287162, "user_tz": -330, "elapsed": 2311, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df4cb187-2b41-4f3b-e320-68f07e37a7aa"
#-- Plot the risk vs. return of randomly generated portfolios
#-- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate=0 #-- Include risk free rate here

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                  showscale=True, 
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ), 
                      mode='markers'))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of Random Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0.18, 0.32])
fig.update_yaxes(range=[0.02,0.27])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
```

<!-- #region id="z0KG5rUt4cs9" -->
## Sample only from efficient frontier
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5KHvQdol4iMD" executionInfo={"status": "ok", "timestamp": 1634669319847, "user_tz": -330, "elapsed": 5346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f20e2f5f-050b-4077-811e-535e7b0c6f68"
#-- Create random portfolio weights and indexes
#- How many assests in the portfolio
n_assets = 5

mean_variance_pairs = []
weights_list=[]
tickers_list=[]

for i in tqdm(range(10000)):
    next_i = False
    while True:
        #- Choose assets randomly without replacement
        assets = np.random.choice(list(daily_returns.columns), n_assets, replace=False)
        #- Choose weights randomly ensuring they sum to one
        weights = np.random.rand(n_assets)
        weights = weights/sum(weights)

        #-- Loop over asset pairs and compute portfolio return and variance
        portfolio_E_Variance = 0
        portfolio_E_Return = 0
        for i in range(len(assets)):
            portfolio_E_Return += weights[i] * mus.loc[assets[i]]
            for j in range(len(assets)):
                portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

        #-- Skip over dominated portfolios
        for R,V in mean_variance_pairs:
            if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                next_i = True
                break
        if next_i:
            break

        #-- Add the mean/variance pairs to a list for plotting
        mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
        weights_list.append(weights)
        tickers_list.append(assets)
        break
```

```python colab={"base_uri": "https://localhost:8080/"} id="GnS0ighc4j8N" executionInfo={"status": "ok", "timestamp": 1634669321414, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="560df408-b6f9-422f-aa09-142f541d68d7"
len(mean_variance_pairs)
```

<!-- #region id="98aAM_4nB8c6" -->
If we plot the risk and return for each of the portfolios on a chart then we will see an arch line at the top of the portfolios.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 517} id="tKciavCy4lf3" executionInfo={"status": "ok", "timestamp": 1634669329630, "user_tz": -330, "elapsed": 464, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1689ef1-9b49-4558-f461-d10502558972"
#-- Plot the risk vs. return of randomly generated portfolios
#-- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate=0 #-- Include risk free rate here

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                  showscale=True, 
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ), 
                      mode='markers',
                      text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))]))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of Random Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0.18, 0.35])
fig.update_yaxes(range=[0.05,0.29])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
```

<!-- #region id="i-Iha0iHB50X" -->
We can see that the chart formed an arch at the top of the portfolios which I marked with the black pen. This line is essentially pointing at the portfolios that are the most efficient. This line is known as the efficient frontier. Every other portfolio that does not reside on the efficient frontier is not as efficient because it offers the same return as a portfolio on the efficient frontier but by taking a higher risk.
<!-- #endregion -->
