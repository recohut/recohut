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

<!-- #region id="2b4F9Warxzut" -->
# Statictics Fundamentals
<!-- #endregion -->

<!-- #region id="jREBos-5PDWv" -->
## Fitting distributions to get parameters
<!-- #endregion -->

<!-- #region id="OxUQZUGHSriF" -->
### Fitting normal distribution on solar cell efficiency data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uYEHjHNQR3eL" executionInfo={"status": "ok", "timestamp": 1633434445600, "user_tz": -330, "elapsed": 1870, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1c4569c3-5359-4658-e862-ec4c7a517a47"
!wget -q --show-progress https://github.com/PacktPublishing/Practical-Data-Science-with-Python/raw/main/Chapter8/data/solar_cell_efficiencies.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="YfhLmUlPR50z" executionInfo={"status": "ok", "timestamp": 1633434535139, "user_tz": -330, "elapsed": 672, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d65ca160-c8d5-4a9a-b178-6edfde95e853"
import pandas as pd

df = pd.read_csv('solar_cell_efficiencies.csv')
df.describe()
```

```python id="kqjjzD47O_e-" colab={"base_uri": "https://localhost:8080/", "height": 281} executionInfo={"status": "ok", "timestamp": 1633434535913, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="92939753-b929-4f22-cabc-6051e6082788"
df.hist(bins=40);
```

```python id="IV8lmprlO_e-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434535915, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a9db0dd6-1bc9-44e6-cdd2-0998ffa9cea0"
df['efficiency'].skew()
```

```python id="WgqXLFhPO_e_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434535917, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8d4b43cd-46d7-4e3a-f5ce-d682a7bae98f"
df['efficiency'].kurt()
```

```python id="03LqEPFTO_fA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434581335, "user_tz": -330, "elapsed": 625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6f67252-9f02-44c9-83c0-2688edc0cf48"
import scipy.stats

scipy.stats.norm.fit(df['efficiency'])
```

<!-- #region id="fq4wOhSsSEJP" -->
### Fitting weibull on MISO wind data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JlnfNiPmS1C9" executionInfo={"status": "ok", "timestamp": 1633434735289, "user_tz": -330, "elapsed": 1283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="88dd7528-b98c-493a-e5b2-13c927b6032e"
!wget -q --show-progress https://github.com/PacktPublishing/Practical-Data-Science-with-Python/raw/main/Chapter8/test_your_knowledge/data/miso_wind_data.csv
```

```python id="3vmJ9IXGSpju" executionInfo={"status": "ok", "timestamp": 1633434801783, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
```

```python id="dyqIxTgySpjx" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1633434805234, "user_tz": -330, "elapsed": 570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a249c9d-eaba-46a4-95b9-dd4b00d4f7cc"
df = pd.read_csv('miso_wind_data.csv')
df.head()
```

```python id="WmIMgVelSpj1" colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"status": "ok", "timestamp": 1633434813957, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e9de949-1059-4114-e1ff-dfef0ae78cfe"
df.describe()
```

```python id="Vo1TtJCLSpj3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434816011, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2281514b-7b9c-4899-ae38-3241607ef19d"
df.info()
```

```python id="CXhvu2YBSpj6" executionInfo={"status": "ok", "timestamp": 1633434818003, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
df['MWh'] = df['MWh'].astype('float')
```

```python id="GufpqFf8Spj7" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1633434833202, "user_tz": -330, "elapsed": 1562, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2358b102-b452-4378-ee86-1e0dc161fee7"
sns.histplot(df['MWh'], kde=True);
```

<!-- #region id="H0wW6kxISpj8" -->
Recall from the chapter that Weibull can often be used to model windspeed-related data. The distribution doesn't look perfect here, we should probably break it up by season. But we will still try fitting with a Weibull and see how it compares.
<!-- #endregion -->

```python id="GhWvy1ZzSpj9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434834490, "user_tz": -330, "elapsed": 744, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2e847521-9e4c-4047-aeec-ae98690e05a2"
# this gives us c, loc, and scale
wb_fit = weibull_min.fit(df['MWh'])
wb_fit
```

```python id="uxl4D4QXSpj-" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1633434844575, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="939cd264-ec5e-44d2-f655-347bb4829a37"
wb = weibull_min(c=wb_fit[0], loc=wb_fit[1], scale=wb_fit[2])
x = np.linspace(0, 20000, 1000)
plt.plot(x, wb.pdf(x))
plt.show()
```

<!-- #region id="0p6W7ayfSpj_" -->
That doesn't look right at all. We need to give starting values for our parameters so it has a better chance of fitting. Let's play around with the parameters to figure out what it should be closer to. The key here was the scale parameter - it needs to be on the order of the spread of the data.
<!-- #endregion -->

```python id="5IwrJB_FSpj_" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1633434855791, "user_tz": -330, "elapsed": 820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6d8b18a5-c95c-4087-d173-7c59a9b71e00"
wb = weibull_min(c=5, loc=0, scale=10000)
plt.plot(x, wb.pdf(x))
plt.show()
```

```python id="AehD-43GSpkB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633434861614, "user_tz": -330, "elapsed": 856, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7bfa333e-6a27-4f33-af5d-7cd2c623866f"
params = weibull_min.fit(df['MWh'].values, scale=20000)
params
```

```python id="babv89EESpkC" colab={"base_uri": "https://localhost:8080/", "height": 276} executionInfo={"status": "ok", "timestamp": 1633434868296, "user_tz": -330, "elapsed": 501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a4d6bfe-1620-4daf-bdb3-07b89c4e0862"
wb = weibull_min(c=params[0], loc=params[1], scale=params[2])
f, ax = plt.subplots()
# the density=1 argument makes the integral of the histogram equal 1, so it's on the same scale as the PDF
df['MWh'].hist(density=1, ax=ax, bins=50)
ax.plot(x, wb.pdf(x))
plt.show()
```

<!-- #region id="pdf_rghkSpkD" -->
Hey, not a bad fit! It looks like it is like a bi-modal distribution, composed of two or three Weibulls from the different seasons. We chose the Weibull, again, because it's known to represent this sort of data. Also, we can eyeball the histogram and pick a distribution that seems to fit.
<!-- #endregion -->

```python id="db8X4u0YSpkE" executionInfo={"status": "ok", "timestamp": 1633434873130, "user_tz": -330, "elapsed": 1346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
means = []
for i in range(10000):
    sample = np.random.choice(df['MWh'], 1000, replace=True)
    means.append(sample.mean())
```

```python id="DujUmsFeSpkF" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1633434877814, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99d80bac-0d94-40e9-ed8c-eda08a738151"
sns.histplot(means)
plt.show()
```

<!-- #region id="5gwc_rxvSpkG" -->
Yes, looks like it's approaching a normal distribution.
<!-- #endregion -->

<!-- #region id="DQBsRfS1ZHcy" -->
## Statistical Tests
<!-- #endregion -->

<!-- #region id="sMv_zHn1ZIvp" -->
### 1-sample 2-sided T-test
<!-- #endregion -->

```python id="lpnDLTnBZ-g_" executionInfo={"status": "ok", "timestamp": 1633436567751, "user_tz": -330, "elapsed": 426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd

solar_data = pd.read_csv('solar_cell_efficiencies.csv')
```

```python colab={"base_uri": "https://localhost:8080/"} id="dIcZKE0jZMgX" executionInfo={"status": "ok", "timestamp": 1633436569458, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea9fb82a-dd73-4e77-87c6-327b3342a287"
from scipy.stats import ttest_1samp

print(solar_data['efficiency'].mean())
ttest_1samp(solar_data['efficiency'], 14, alternative='two-sided')
```

```python colab={"base_uri": "https://localhost:8080/"} id="eVD2jdJ0Z2Hn" executionInfo={"status": "ok", "timestamp": 1633436582650, "user_tz": -330, "elapsed": 677, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f86a2340-7f37-4844-9c25-853928635ead"
sample = solar_data['efficiency'].sample(30, random_state=1)
print(sample.mean())
ttest_1samp(sample, 14)
```

<!-- #region id="EcAKFc2PaBro" -->
When we are considering full sample, p-value is less than significance level ($\alpha$) of 0.05, so we reject the null hypothesis. This means the solar efficiency is more than 14%. On the other hand, we failed to reject the null hypothesis on a sample data of 30 records.
<!-- #endregion -->

<!-- #region id="WB5npQJfbcWZ" -->
The proper test to use for larger sample sizes is the z-test. This ends up being about the same as a t-test, however. We can use this from the statsmodels package.
<!-- #endregion -->

```python id="Tut43EkjY5a_"
from statsmodels.stats.weightstats import ztest
```

```python id="K4RySnQoY5bA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436759354, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="feb32fa1-7e2c-497d-e34d-442f8dba1478"
ztest(solar_data['efficiency'], value=14)
```

<!-- #region id="jjjrd2SUbicB" -->
Let's say we want to make sure the average efficiency of our latest batch of solar cells is greater than 14%. The sample we used that was measured from a recent production run is in our solar data we've already loaded. We can formulate our null hypothesis as this: the sample mean is less than or equal to the expected mean of 14%. The alternative hypothesis is then: the sample mean is greater than the expected mean of 14%. We can perform this test with scipy like so:
<!-- #endregion -->

```python id="eJslK_flY5bB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436759356, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ffd8a511-2738-4f47-edb9-ea8d1d195a40"
ttest_1samp(solar_data['efficiency'], 14, alternative='greater')
```

<!-- #region id="IscRG7pNbsYq" -->
The alternative argument is set to 'greater', meaning the alternative hypothesis is that the sample mean is greater than the expected mean. Our results show the null hypothesis is rejected, and it looks like our sample mean is greater than 14% with statistical significance.
<!-- #endregion -->

```python id="DKJixYyYY5bC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436759358, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f976331e-3f7a-4150-e8cc-d26c98eac200"
ttest_1samp(solar_data['efficiency'], 14, alternative='less')
```

<!-- #region id="iVlvh4Sfau6c" -->
### A/B testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KfWxh2k6a3lb" executionInfo={"status": "ok", "timestamp": 1633436833050, "user_tz": -330, "elapsed": 1089, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="748a1714-2b06-46ba-c162-bd3974a22fa6"
!wget -q --show-progress https://github.com/PacktPublishing/Practical-Data-Science-with-Python/raw/main/Chapter9/data/ab_sales_data.csv
```

<!-- #region id="dcL2ORrDbxzm" -->
Let's say we have a website selling t-shirts and want to experiment with the design to try and drive more sales. We're going to change the layout in a B version of the site and compare our sales rates to the A version.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="vnG6Q7eLbA1M" executionInfo={"status": "ok", "timestamp": 1633436906113, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1d0b874e-9c40-4b61-c58a-3d71fb74acfb"
ab_df = pd.read_csv('ab_sales_data.csv')
ab_df.head()
```

<!-- #region id="RAnbUXLfb2gw" -->
We have a column for the A design, and each row is a website visitor. A value of 1 represents a sale, while 0 represents no sale. The B design column is the same, and the samples are not paired up (each sample from A and B is individual and independent). We can look at the mean sales rates easily:
<!-- #endregion -->

```python id="NP0Ay-LlY5bG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436907745, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6b986be4-4e63-4e5b-8fa3-5c6b10a93636"
ab_df.mean()
```

<!-- #region id="8HBzxNXsb6D2" -->
This shows us B has a slightly higher sales rate. To test if B is really better than A, we can first try a two-sample, two-sided t-test. The null hypothesis is that the means of the two groups are the same; the alternative is that they are not the same (for a two-sided test).
<!-- #endregion -->

```python id="qReg6PJYY5bI" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436909184, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2ff6c3c1-95f6-4a62-e787-feed2dd9ee01"
ztest(ab_df['a_sale'], ab_df['b_sale'])
```

<!-- #region id="By82QFwIcFN5" -->
Remember that the first value is the z-statistic, and the second value in the tuple is the p-value. In this case, it looks like there is a significant difference in the means, since the p-value of 0.024 is less than our significance threshold of 0.05. We already know from examination that the B sales rate was a little higher, so it appears the B design is better.
<!-- #endregion -->

<!-- #region id="Yuc6BOxecKk6" -->
To be a little more precise, we can also specify the direction of the test. With statsmodels, the options for the alternative argument are two-sided, larger, and smaller. Specifying larger means the alternative hypothesis is that A's mean is larger than B's. The null hypothesis in that case is that A's mean is less than or equal to B's mean. We'll use smaller to carry out our one-sided z-test to see if B's average sales value is greater than A's:
<!-- #endregion -->

```python id="Jd8K_pFMY5bJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436909185, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ccfc08b-4473-42ef-fa5d-a82c39647ea8"
ztest(ab_df['a_sale'], ab_df['b_sale'], alternative='smaller')
```

```python id="_TW0j_rbY5bK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436909842, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="98a5af7d-c672-449d-bbad-f6f5afb5e785"
ztest(ab_df['a_sale'], ab_df['b_sale'], value=-0.01, alternative='smaller')
```

```python id="Hg1kMqWPY5bL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633436910272, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a89fd75-1dcf-4b79-cdd4-9b73bf5c4ddd"
ztest(ab_df['b_sale'], ab_df['a_sale'], value=0.01, alternative='larger')
```

<!-- #region id="eDaVGQV_Y5bM" -->
### Bootstrap A/B
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3t1HL0H_ckVJ" executionInfo={"status": "ok", "timestamp": 1633437248094, "user_tz": -330, "elapsed": 5210, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="958c8d6f-ed72-4e45-f34b-f2574d39881d"
!pip install -q bootstrapped
```

<!-- #region id="WAifUYbuc31Y" -->
Bootstrapping is another method for A/B testing. With this, we can use sampling with replacement (bootstrapping) to calculate many means of our A and B datasets, then get the confidence intervals of the difference in mean values between A and B. If the confidence interval for the difference in means doesn't pass through 0, we can say with a certain percent confidence that the means are different. For example, we can use the bootstrapped package (which you will need to install with pip install bootstrapped) to do this:
<!-- #endregion -->

```python id="rov2xDBNY5bM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633437255708, "user_tz": -330, "elapsed": 5452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="350487e2-4497-4161-9538-f3a88fe29778"
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats

bs.bootstrap_ab(test=ab_df['b_sale'].values,
                ctrl=ab_df['a_sale'].values,
                stat_func=bs_stats.mean,
                compare_func=bs_compare.difference,
                alpha=0.05)
```

<!-- #region id="hEHyDPRBcnAR" -->
The values are small, but we can see the 95% confidence interval doesn't quite pass through 0, so we can say with 95% confidence B is better than A. However, it could be that B is only better than A by 0.0008 in absolute value, which wouldn't be much of an improvement on A.
<!-- #endregion -->

<!-- #region id="InCpuSTFdKv5" -->
### Testing between several groups with ANOVA

Testing one or two samples is useful in many situations, but we can also find ourselves needing to test the means between several groups. We can use multiple t-tests with the Bonferroni correction as one method, but another way is to use ANOVA and post hoc tests.

Let's say we want to test more than one design at a time and compare them all to see which is best: A, B, and C designs. For comparing the means of three or more groups, we can use an ANOVA test. There is also a way to compare several groups with t-tests using what's called the Bonferroni correction; this is available in the scikit_posthocs.posthoc_ttest() function from the scikit-posthocs package (you will need to install this package with conda or pip). This would tell us the difference between all the pairs from our groups of data – we will come back to other ways to do this shortly.

However, ANOVA can be first used to see if there is any difference between any of the groups. Instead of a t-test, it uses an F-test. Again, this method provides a p-value, which we compare to a significant value we choose (usually 0.05).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_cDjt-_EdVdo" executionInfo={"status": "ok", "timestamp": 1633437461019, "user_tz": -330, "elapsed": 2155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="52b68d6b-24c7-493c-a725-7ad5d748df94"
!wget -q --show-progress https://github.com/PacktPublishing/Practical-Data-Science-with-Python/raw/main/Chapter9/data/abc_sales_data.csv
```

<!-- #region id="xrwUe25tdZsO" -->
Because one assumption for ANOVA is that the data comes from normal distributions, we are using data from binomial distributions. This is chunks of 100 website visitors, with a count of how many visitors made a purchase.

Each row is a number between 0 and 100. As we learned in the previous chapter, sampling data from distributions many times tends toward a normal distribution, so if we structure our data in this way, we can approach a normal distribution instead of a binomial distribution like with our other set of A/B sales data.

In this case, a binomial distribution is based on Bernoulli trials (like coin flips), and a collection of binomial distribution samples tends toward a normal distribution. We can load the data with pandas, then conduct an ANOVA test:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Z8S7S4XigSrk" executionInfo={"status": "ok", "timestamp": 1633438240404, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5385f7e1-8803-451c-877a-a7f9dcab6745"
from scipy.stats import f_oneway

abc_df = pd.read_csv('abc_sales_data.csv')
abc_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="V16cvrPigYlF" executionInfo={"status": "ok", "timestamp": 1633438262803, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1882d9c0-0bfb-4804-cf7e-0b6c62e917b7"
f_oneway(abc_df['a_sale'], abc_df['b_sale'], abc_df['c_sale'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="E45ShkXKgeGN" executionInfo={"status": "ok", "timestamp": 1633438264661, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01f59018-afc0-4046-808f-4b80e670273d"
abc_df.mean()
```

<!-- #region id="gabvYeqygejT" -->
Here, we provide as many datasets as we want to our f_oneway() function, which performs an ANOVA test. We get an F-statistic and p-value. As usual, we compare the p-value to our significance level to determine if we can reject the null hypothesis. The null hypothesis here is that the means are all the same; the alternative is that the means are different. Since p < 0.05, we can reject the null hypothesis, and our test shows the means to be different. Looking at the means with abc_df.mean(), we can see they are 4.9, 5.5, and 6.9 for A, B, and C, which look quite different. However, it would be nice to know which differences between the groups are significant. For this, we can use a post hoc test.
<!-- #endregion -->

<!-- #region id="nw6vf3PsgtXt" -->
There are several post hoc tests, but we will use one common post hoc test: the Tukey test. This is named after John Tukey, the legendary statistician who created boxplots and pioneered EDA. Different ANOVA post hoc tests have different subtleties that make them useful in different situations, but Tukey is a decent general test to use as a default.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KZuD7AdxhA99" executionInfo={"status": "ok", "timestamp": 1633438425569, "user_tz": -330, "elapsed": 5131, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ca96984b-3210-47d3-d6f8-6fc38baf1f68"
!pip install -q scikit_posthocs
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="je6HmxXng5Yy" executionInfo={"status": "ok", "timestamp": 1633438425595, "user_tz": -330, "elapsed": 70, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb1abab1-4533-4efc-8ab5-7d3eebf92360"
from scikit_posthocs import posthoc_tukey

melted_abc = abc_df.melt(var_name='groups', value_name='values')
melted_abc.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="6aPA4pGNhAJc" executionInfo={"status": "ok", "timestamp": 1633438430453, "user_tz": -330, "elapsed": 420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea15771d-7c09-4d21-a81e-3e64631ae617"
posthoc_tukey(melted_abc, group_col='groups', val_col='values')
```

<!-- #region id="octeGoAthG_g" -->
These are p-values for the hypothesis we are testing that the means are not different between pairs. Since the p-values are small between all the pairs (0.001, much less than 0.05) we can say the differences between the means of all the groups are significant. It is possible with the test that some differences between groups may be significant while others may not.
<!-- #endregion -->

<!-- #region id="RWmMUPXvhUJ9" -->
### Comparing Winds in different seasons with ANOVA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ae5dgZiNiOFf" executionInfo={"status": "ok", "timestamp": 1633438753332, "user_tz": -330, "elapsed": 1112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="67534f75-8461-4608-a6ca-01cb2eccbca4"
!wget -q --show-progress https://github.com/PacktPublishing/Practical-Data-Science-with-Python/raw/main/Chapter9/test_your_knowledge/data/miso_wind_data.csv
```

```python id="kgYh4J_RiVsn" executionInfo={"status": "ok", "timestamp": 1633438773432, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
from scikit_posthocs import posthoc_tukey
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="9VadVMuBiZ3J" executionInfo={"status": "ok", "timestamp": 1633438784410, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cd22ced2-d4ba-4262-e6f5-f8846af630a1"
df = pd.read_csv('miso_wind_data.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="vPuh_H5gibgU" executionInfo={"status": "ok", "timestamp": 1633438803240, "user_tz": -330, "elapsed": 400, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5460c94-65c1-4cee-94e2-2a1cf190ea06"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="EYjWXAZeihOj" executionInfo={"status": "ok", "timestamp": 1633438819810, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6c4da3c-5c48-4486-e33e-45902e9892dc"
df['MWh'] = df['MWh'].astype('float')
df['Market Day'] = pd.to_datetime(df['Market Day'])
df.set_index('Market Day', inplace=True)

spring = df['3-1-2020': '5-31-2020'][['MWh']]
summer = df['6-1-2020': '8-31-2020'][['MWh']]
fall = df['9-1-2020': '11-30-2020'][['MWh']]
winter = df['12-1-2020':][['MWh']]

spring.columns = ['spring']
summer.columns = ['summer']
fall.columns = ['fall']
winter.columns = ['winter']

spring.reset_index(inplace=True, drop=True)
summer.reset_index(inplace=True, drop=True)
fall.reset_index(inplace=True, drop=True)
winter.reset_index(inplace=True, drop=True)

tukey_df = pd.concat([spring.iloc[:744], summer.iloc[:744], fall.iloc[:744], winter.iloc[:744]], axis=1)
tukey_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="MT-wXeDuilYE" executionInfo={"status": "ok", "timestamp": 1633438831559, "user_tz": -330, "elapsed": 475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="07cf00ee-4702-4045-c550-f3c8aa001126"
melted = tukey_df.melt(var_name='groups', value_name='values')
melted.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="Pbrek4FEioUQ" executionInfo={"status": "ok", "timestamp": 1633438841719, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="62300f5d-c048-45d9-e279-e49ffac44345"
posthoc_tukey(melted, group_col='groups', val_col='values')
```

```python colab={"base_uri": "https://localhost:8080/"} id="TWVsl_Niiq2l" executionInfo={"status": "ok", "timestamp": 1633438851193, "user_tz": -330, "elapsed": 393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb869682-66f8-4967-a4f5-19b1fdca2221"
tukey_df.mean()
```

<!-- #region id="UUmw1t4-is9L" -->
We can see the difference in meanbetween most groups is significant, with winter having the strongest wind power, and the summer the weakest. The only groups to not have a significant different are spring and fall. We can see these two have almost the same MWh value around 8500. We used the Tukey test to test for significant differences between multiple groups.
<!-- #endregion -->
