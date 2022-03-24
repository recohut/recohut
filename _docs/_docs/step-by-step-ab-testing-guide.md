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

<!-- #region id="-44lu3y8xbZz" -->
# A/B testing step-by-step guide in Python
> In this notebook we'll go over the process of analysing an A/B test, from formulating a hypothesis, testing it, and finally interpreting results.

- toc: true
- badges: true
- comments: true
- categories: [ABTest]
- author: "<a href='https://medium.com/@RenatoFillinich/ab-testing-with-python-e5964dd66143'>Renato Fillinich</a>"
- image:
<!-- #endregion -->

<!-- #region id="D3XBuKL-xbZ3" -->
In this notebook we'll go over the process of analysing an A/B test, from formulating a hypothesis, testing it, and finally interpreting results. For our data, we'll use a <a href='https://www.kaggle.com/zhangluyuan/ab-testing?select=ab_data.csv'>dataset from Kaggle</a> which contains the results of an A/B test on what seems to be 2 different designs of a website page (old_page vs. new_page). Here's what we'll do:

1. Designing our experiment
2. Collecting and preparing the data
3. Visualising the results
4. Testing the hypothesis
5. Drawing conclusions

To make it a bit more realistic, here's a potential **scenario** for our study:

> Let's imagine you work on the product team at a medium-sized **online e-commerce business**. The UX designer worked really hard on a new version of the product page, with the hope that it will lead to a higher conversion rate. The product manager (PM) told you that the **current conversion rate** is about **13%** on average throughout the year, and that the team would be happy with an **increase of 2%**, meaning that the new design will be considered a success if it raises the conversion rate to 15%.

Before rolling out the change, the team would be more comfortable testing it on a small number of users to see how it performs, so you suggest running an **A/B test** on a subset of your user base users.
<!-- #endregion -->

<!-- #region id="mSR2XKnOxbZ5" -->
***
## 1. Designing our experiment
<!-- #endregion -->

<!-- #region id="UddxvZVjxbZ6" -->
### Formulating a hypothesis

First things first, we want to make sure we formulate a hypothesis at the start of our project. This will make sure our interpretation of the results is correct as well as rigorous.

Given we don't know if the new design will perform better or worse (or the same?) as our current design, we'll choose a <a href="https://en.wikipedia.org/wiki/One-_and_two-tailed_tests">**two-tailed test**</a>:

$$H_0: p = p_0$$
$$H_a: p \ne p_0$$

where $p$ and $p_0$ stand for the conversion rate of the new and old design, respectively. We'll also set a **confidence level of 95%**:

$$\alpha = 0.05$$

The $\alpha$ value is a threshold we set, by which we say "if the probability of observing a result as extreme or more ($p$-value) is lower than $\alpha$, then we reject the null hypothesis". Since our $\alpha=0.05$ (indicating 5% probability), our confidence (1 - $\alpha$) is 95%.

Don't worry if you are not familiar with the above, all this really means is that whatever conversion rate we observe for our new design in our test, we want to be 95% confident it is statistically different from the conversion rate of our old design, before we decide to reject the Null hypothesis $H_0$. 
<!-- #endregion -->

<!-- #region id="23b3NwD9xbZ7" -->
### Choosing the variables

For our test we'll need **two groups**:
* A `control` group - They'll be shown the old design
* A `treatment` (or experimental) group - They'll be shown the new design

This will be our *Independent Variable*. The reason we have two groups even though we know the baseline conversion rate is that we want to control for other variables that could have an effect on our results, such as seasonality: by having a `control` group we can directly compare their results to the `treatment` group, because the only systematic difference between the groups is the design of the product page, and we can therefore attribute any differences in results to the designs.

For our *Dependent Variable* (i.e. what we are trying to measure), we are interested in capturing the `conversion rate`. A way we can code this is by  each user session with a binary variable:
* `0` - The user did not buy the product during this user session
* `1` - The user bought the product during this user session

This way, we can easily calculate the mean for each group to get the conversion rate of each design.
<!-- #endregion -->

<!-- #region id="yKqt1-cOxbZ-" -->
### Choosing a sample size

It is important to note that since we won't test the whole user base (our <a href="https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/3-populations-and-samples">population</a>), the conversion rates that we'll get will inevitably be only *estimates* of the true rates.

The number of people (or user sessions) we decide to capture in each group will have an effect on the precision of our estimated conversion rates: **the larger the sample size**, the more precise our estimates (i.e. the smaller our confidence intervals), **the higher the chance to detect a difference** in the two groups, if present.

On the other hand, the larger our sample gets, the more expensive (and impractical) our study becomes.

*So how many people should we have in each group?*

The sample size we need is estimated through something called <a href="https://research.usu.edu//irb/wp-content/uploads/sites/12/2015/08/A_Researchers_Guide_to_Power_Analysis_USU.pdf">*Power analysis*</a>, and it depends on a few factors:
* **Power of the test** ($1 - \beta$) - This represents the probability of finding a statistical difference between the groups in our test when a difference is actually present. This is usually set at 0.8 as a convention (here's more info on <a href="https://en.wikipedia.org/wiki/Power_of_a_test">statistical power</a>, if you are curious)
* **Alpha value** ($\alpha$) - The critical value we set earlier to 0.05
* **Effect size** - How big of a difference we expect there to be between the conversion rates

Since our team would be happy with a difference of 2%, we can use 13% and 15% to calculate the effect size we expect. 

Luckily, **Python takes care of all these calculations for us**:
<!-- #endregion -->

```python id="J0Bq-G8DxbaA"
# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# Some plot styling preferences
plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 14}

mpl.rc('font', **font)
```

```python id="eiaZhk-vxbaC" colab={"base_uri": "https://localhost:8080/"} outputId="8c4fc745-a881-4d6d-f6f3-6e25308ff967"
effect_size = sms.proportion_effectsize(0.13, 0.15)    # Calculating effect size based on our expected rates

required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  # Calculating sample size needed

required_n = ceil(required_n)                          # Rounding up to next whole number                          

print(required_n)
```

<!-- #region id="eZl2r4v7xbaF" -->
We'd need **at least 4720 observations for each group**. 

Having set the `power` parameter to 0.8 in practice means that if there exists an actual difference in conversion rate between our designs, assuming the difference is the one we estimated (13% vs. 15%), we have about 80% chance to detect it as statistically significant in our test with the sample size we calculated.
<!-- #endregion -->

<!-- #region id="CURpYxEZxbaG" -->
***
## 2. Collecting and preparing the data
<!-- #endregion -->

<!-- #region id="QnEwYEjhxbaH" -->
Great stuff! So now that we have our required sample size, we need to collect the data. Usually at this point you would work with your team to set up the experiment, likely with the help of the Engineering team, and make sure that you collect enough data based on the sample size needed.

However, since we'll use a dataset that we found online, in order to simulate this situation we'll:
1. Download the <a href='https://www.kaggle.com/zhangluyuan/ab-testing?select=ab_data.csv'>dataset from Kaggle</a>
2. Read the data into a pandas DataFrame
3. Check and clean the data as needed
4. Randomly sample `n=4720` rows from the DataFrame for each group *****

***Note**: Normally, we would not need to perform step 4, this is just for the sake of the exercise

Since I already downloaded the dataset, I'll go straight to number 2.
<!-- #endregion -->

```python id="Z4s5ICJQxbaH" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="36350c16-000d-4124-f4be-38115ca5b396"
df = pd.read_csv('https://github.com/sparsh-ai/reco-data/raw/master/ab-testing.zip')

df.head()
```

```python id="Z3RiYHMBxbaI" colab={"base_uri": "https://localhost:8080/"} outputId="d912411a-3e0c-4bbe-dba2-d67f074a1238"
df.info()
```

```python id="3jGYR9EJxbaI" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="dd8c7044-844f-4f04-a0bd-8a6b535e2117"
# To make sure all the control group are seeing the old page and viceversa

pd.crosstab(df['group'], df['landing_page'])
```

<!-- #region id="v4EyzPcPxbaJ" -->
There are **294478 rows** in the DataFrame, each representing a user session, as well as **5 columns** :
* `user_id` - The user ID of each session
* `timestamp` - Timestamp for the session
* `group` - Which group the user was assigned to for that session {`control`, `treatment`}
* `landing_page` - Which design each user saw on that session {`old_page`, `new_page`}
* `converted` - Whether the session ended in a conversion or not (binary, `0`=not converted, `1`=converted)

We'll actually only use the `group` and `converted` columns for the analysis.

Before we go ahead and sample the data to get our subset, let's make sure there are no users that have been sampled multiple times.
<!-- #endregion -->

```python id="32x1ywhYxbaJ" colab={"base_uri": "https://localhost:8080/"} outputId="cb57b90a-00eb-4144-ec79-c88774bd58c8"
session_counts = df['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()

print(f'There are {multi_users} users that appear multiple times in the dataset')
```

<!-- #region id="yI1v4QCrxbaK" -->
There are, in fact, users that appear more than once. Since the number is pretty low, we'll go ahead and remove them from the DataFrame to avoid sampling the same users twice.
<!-- #endregion -->

```python id="1VTCNR5SxbaK" colab={"base_uri": "https://localhost:8080/"} outputId="298a791b-4df8-4ea2-d2f1-293cae82b43b"
users_to_drop = session_counts[session_counts > 1].index

df = df[~df['user_id'].isin(users_to_drop)]
print(f'The updated dataset now has {df.shape[0]} entries')
```

<!-- #region id="VYalyUeJxbaK" -->
### Sampling

Now that our DataFrame is nice and clean, we can proceed and sample `n=4720` entries for each of the groups. We can use pandas' `DataFrame.sample()` method to do this, which will perform Simple Random Sampling for us. 

**Note**: I've set `random_state=22` so that the results are reproducible if you feel like following on your own Notebook: just use `random_state=22` in your function and you should get the same sample as I did.
<!-- #endregion -->

```python id="6EXbtFaexbaL"
control_sample = df[df['group'] == 'control'].sample(n=required_n, random_state=22)
treatment_sample = df[df['group'] == 'treatment'].sample(n=required_n, random_state=22)

ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
```

```python id="ORf0Hlv2xbaL" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="0edd346b-7612-4ef1-bb54-7578893a3326"
ab_test
```

```python id="fIFKXUmBxbaN" colab={"base_uri": "https://localhost:8080/"} outputId="cdecfb42-bbf6-4e21-ab13-778bb1b5af33"
ab_test.info()
```

```python id="5mGQRVnSxbaO" colab={"base_uri": "https://localhost:8080/"} outputId="9c440cdc-37ac-4233-eb7f-b824768b18b1"
ab_test['group'].value_counts()
```

<!-- #region id="HPUc1bznxbaO" -->
Great, looks like everything went as planned, and we are now ready to analyse our results.
<!-- #endregion -->

<!-- #region id="g1PoRb9KxbaP" -->
***
## 3. Visualising the results
<!-- #endregion -->

<!-- #region id="fmwnE9rXxbaP" -->
The first thing we can do is to calculate some **basic statistics** to get an idea of what our samples look like.
<!-- #endregion -->

```python id="OrfD9yUyxbaQ" colab={"base_uri": "https://localhost:8080/", "height": 103} outputId="2e3fc958-aaf7-49c1-aed5-a8bd53ac2b68"
conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')
```

<!-- #region id="MPADWAVexbaR" -->
Judging by the stats above, it does look like **our two designs performed very similarly**, with our new design performing slightly better, approx. **12.3% vs. 12.6% conversion rate**.

Plotting the data will make these results easier to grasp:
<!-- #endregion -->

```python id="gcasaVmXxbaR" colab={"base_uri": "https://localhost:8080/", "height": 489} outputId="11dcf7af-5468-43f0-959e-a7e4577fb1bd"
plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);
```

<!-- #region id="9s8WH7MoxbaS" -->
The conversion rates for our groups are indeed very close. Also note that the conversion rate of the `control` group is lower than what we would have expected given what we knew about our avg. conversion rate (12.3% vs. 13%). This goes to show that there is some variation in results when sampling from a population.

So... the `treatment` group's value is higher. **Is this difference *statistically significant***?
<!-- #endregion -->

<!-- #region id="Cz0hF-gXxbaS" -->
***
## 4. Testing the hypothesis
<!-- #endregion -->

<!-- #region id="np2wDTn4xbaS" -->
The last step of our analysis is testing our hypothesis. Since we have a very large sample, we can use the <a href="https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval">normal approximation</a> for calculating our $p$-value (i.e. z-test). 

Again, Python makes all the calculations very easy. We can use the `statsmodels.stats.proportion` module to get the $p$-value and confidence intervals:
<!-- #endregion -->

```python id="WRFRW8NpxbaT"
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
```

```python id="dbjM4L56xbaU"
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']
```

```python id="HUm9Alv-xbaV" colab={"base_uri": "https://localhost:8080/"} outputId="bd092939-cc8c-4401-97f0-73e4ac35d3c4"
n_con = control_results.count()
n_treat = treatment_results.count()
successes = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
```

<!-- #region id="wY-zEPrLxbaV" -->
***
## 5. Drawing conclusions
<!-- #endregion -->

<!-- #region id="AqesRmhrxbaW" -->
Since our $p$-value=0.732 is way above our $\alpha$=0.05, we cannot reject the null hypothesis $H_0$, which means that our new design did not perform significantly different (let alone better) than our old one :(

Additionally, if we look at the confidence interval for the `treatment` group ([0.116, 0.135], i.e. 11.6-13.5%) we notice that:
1. It includes our baseline value of 13% conversion rate
2. It does not include our target value of 15% (the 2% uplift we were aiming for)

What this means is that it is more likely that the true conversion rate of the new design is similar to our baseline, rather than the 15% target we had hoped for. This is further proof that our new design is not likely to be an improvement on our old design, and that unfortunately we are back to the drawing board! 
<!-- #endregion -->
