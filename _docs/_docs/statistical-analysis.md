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

<!-- #region id="PNHRSsdVphjQ" -->
# Statistical Analysis

Statistics is a branch of mathematics dealing with the collection and interpretation of numeric data. It is the precursor of all modern data science. The term *statistic* originally signified “the science of the state” because statistical methods were first developed to analyze the data of state governments. Since ancient times, government agencies have gathered data pertaining to their populace. That data would be used to levy taxes and organize large military campaigns. Hence, critical state decisions depended on the quality of data. Poor record keeping could lead to potentially disastrous results. That is why state bureaucrats were very concerned by any random fluctuations in their records. Probability theory eventually tamed these fluctuations, making the randomness interpretable. Ever since then, statistics and probability theory have been closely intertwined.

Statistics and probability theory are closely related, but in some ways, they are very different. Probability theory studies random processes over a potentially infinite number of measurements. It is not bound by real-world limitations. This allows us to model the behavior of a coin by imagining millions of coin flips. In real life, flipping a coin millions of times is a pointlessly time-consuming endeavor. Surely we can sacrifice some data instead of flipping coins all day and night. Statisticians acknowledge these constraints placed on us by the data-gathering process. Real-world data collection is costly and time consuming. Every data point carries a price. We cannot survey a country’s population without employing government officials. We cannot test our online ads without paying for every ad that’s clicked. Thus, the size of our final dataset usually depends on the size of our initial budget. If the budget is constrained, then the data will also be constrained. This trade-off between data and resourcing lies at the heart of modern statistics. Statistics help us understand exactly how much data is sufficient to draw insights and make impactful decisions. The purpose of statistics is to find meaning in data even when that data is limited in size.
<!-- #endregion -->

<!-- #region id="jfHdYdDE708R" -->
The SciPy library includes an entire module for addressing problems in
probability and statistics; `scipy.stats`. Lets import that module.

**Importing the `stats` module from SciPy**
<!-- #endregion -->

```python id="bhTuUtri708f"
from scipy import stats
```

<!-- #region id="JBZrMWmc708i" -->
Let's say we want to compute the probability of a fair coin producing at-least 16 heads after 20 flips. SciPy allows us
to measure this probability directly using the `stats.binomial_test` method.

**Analyzing extreme head-counts using SciPy**
<!-- #endregion -->

```python id="rL49kKgl708k" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304325534, "user_tz": -330, "elapsed": 987, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2d214eae-0258-4605-f47f-d0e0d04018be"
num_heads = 16
num_flips = 20
prob_head = 0.5
prob = stats.binom_test(num_heads, num_flips, prob_head)
print(f"Probability of observing more than 15 heads or 15 tails is {prob:.17f}")
```

<!-- #region id="ervpQAu8708n" -->
Our method-call returned the probability of seeing a coin-flip sequence where 16 or more coins fell on the same face. If we want the probability seeing exactly 16 heads, then we must utilize the `stats.binom.pmf` method.

**Computing an exact probability using `stats.binom.pmf`**
<!-- #endregion -->

```python id="zPT1G-gQ708o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304342165, "user_tz": -330, "elapsed": 902, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b6adc3a5-5cba-4b6e-b978-3666e7728fce"
prob_16_heads = stats.binom.pmf(num_heads, num_flips, prob_head)
print(f"The probability of seeing {num_heads} of {num_flips} heads is {prob_16_heads}")
```

<!-- #region id="3khDIo-Q708q" -->
We’ve used `stats.binom.pmf` to find the probability of seeing exactly 16 heads. However, that method is also able to compute multiple probabilities simultaneously.

**Computing an array of probabilities using `stats.binom.pmf`**
<!-- #endregion -->

```python id="NPCh9O3O708u"
probabilities = stats.binom.pmf([4, 16], num_flips, prob_head)
assert probabilities.tolist() == [prob_16_heads] * 2
```

<!-- #region id="3VHQ60Cl708x" -->
List-passing allows us to easily compute probabilities across intervals. For example, if we pass `range(21)` into `stats.binom.pmf`, then the outputted array will contain all probabilities across the interval of every possible head-count.

**Computing an interval probability using `stats.binom.pmf`**
<!-- #endregion -->

```python id="9C5enCi1708z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304360906, "user_tz": -330, "elapsed": 668, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f47ff587-27cf-4007-d725-456ad04bd073"
interval_all_counts = range(21)
probabilities = stats.binom.pmf(interval_all_counts, num_flips, prob_head)
total_prob = probabilities.sum()
print(f"Total sum of probabilities equals {total_prob:.14f}")
```

<!-- #region id="qB2qJq5D7080" -->
Plotting `interval_all_counts` versus `probabilities` will reveal the shape of our 20 coin-flip distribution.

**Plotting a 20 coin-flip Binomial Distribution**
<!-- #endregion -->

```python id="iDOPMkzU7081" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304383391, "user_tz": -330, "elapsed": 767, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="826dacd4-efcd-48c4-d541-958e473d6b9d"
import matplotlib.pyplot as plt
plt.plot(interval_all_counts, probabilities)
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

<!-- #region id="J8cV9m6G7083" -->
The `stats.binom.pmf` methods allows to display any distribution associated with an arbitrary coin-flip count. Let's simultaneously plot the distributions for 20, 80, 140, and 200 coin-flips.

**Plotting 5 different Binomial distributions**
<!-- #endregion -->

```python id="VtHc5z6x7084" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304444400, "user_tz": -330, "elapsed": 926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35ca06de-6fcb-48b5-a78d-ad72f241a2fb"
flip_counts = [20, 80, 140, 200]
linestyles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'k']

for num_flips, linestyle, color in zip(flip_counts, linestyles, colors):
    x_values = range(num_flips + 1)
    y_values = stats.binom.pmf(x_values, num_flips, 0.5)
    plt.plot(x_values, y_values, linestyle=linestyle, color=color,
             label=f'{num_flips} coin-flips')
plt.legend()
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

<!-- #region id="iuKl13cK7085" -->
The plotted distributions grow more dispersed around their central positions as these central positions relocate right.

## Mean as a Measure of Centrality

Suppose we've measured noon-time measurements over course of the 7 days.  These temperatures are stored in a NumPy array.

**Storing recorded temperatures in a NumPy array**
<!-- #endregion -->

```python id="Y7sPTlkr7086"
import numpy as np
measurements = np.array([80, 77, 73, 61, 74, 79, 81])
```

<!-- #region id="mtjayBYC7088" -->
We'll now attempt to summarize our measurements using a single central value. First, we'll plot the sorted temperatures in order to evaluate their centrality.

**Plotting the recorded temperatures**
<!-- #endregion -->

```python id="_iUDotSt7089" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637304546556, "user_tz": -330, "elapsed": 966, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="058b22db-f995-4ef9-e090-c02551686295"
measurements.sort()
number_of_days = measurements.size
plt.plot(range(number_of_days), measurements)
plt.scatter(range(number_of_days), measurements)
plt.ylabel('Temperature')
plt.show()
```

<!-- #region id="KW3JwVhw709B" -->
Based on the plot, a central temperature exists somewhere between 60 degrees and 80 degrees. Let’s quantitate our estimate as the mid-point between the lowest value and the highest value in the plot.

**Finding the midpoint temperature**
<!-- #endregion -->

```python id="KucPCSmD709C" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304566468, "user_tz": -330, "elapsed": 785, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="611a8bbd-428f-4240-fb9e-b7155a26015f"
difference = measurements.max() - measurements.min()
midpoint = measurements.min() + difference / 2
assert midpoint == (measurements.max() + measurements.min()) / 2
print(f"The midpoint temperature is {midpoint} degrees")
```

<!-- #region id="4-LzwoAq709D" -->
Let’s mark that midpoint in our plot using a horizontal line. We’ll draw the horizontal line by calling `plt.axhline(midpoint)`.

**Plotting the midpoint temperature**
<!-- #endregion -->

```python id="YQbW1zKg709E" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637304584636, "user_tz": -330, "elapsed": 848, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cd74ea06-ac5e-4ce5-e45a-4addde8b3e3b"
plt.plot(range(number_of_days), measurements)
plt.scatter(range(number_of_days), measurements)
plt.axhline(midpoint, color='k', linestyle='--')
plt.ylabel('Temperature')
plt.show()
```

<!-- #region id="4Q0JijOl709F" -->
Our plotted midpoint seems a little low. Intuitively, our central value should split the measurements more evenly.  The middle array element, which statisticians call the **median**, will split our measurements into exactly equal parts.

**Plotting the median temperature**
<!-- #endregion -->

```python id="MvROZM_l709G" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1637304606143, "user_tz": -330, "elapsed": 1436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e0f1dc19-af32-4bfe-90b6-08cbb235cae4"
median = measurements[3]
print(f"The median temperature is {median} degrees")
plt.plot(range(number_of_days), measurements)
plt.scatter(range(number_of_days), measurements)
plt.axhline(midpoint, color='k', linestyle='--', label='midpoint')
plt.axhline(median, color='g', linestyle='-.', label='median')
plt.legend()
plt.ylabel('Temperature')
plt.show()
```

<!-- #region id="ky5NlZG6709H" -->
Our median split is not as balanced as it could be. Perhaps we can balance the split by penalizing the median for being too far from the minimum. We'll carry out this penalty using **squared distance**. If we penalize our central value based on its distance to the minimum then the squared distance penalty will grow noticeably larger as it drifts away from 61 degrees.

**Penalizing centers using squared distance from minimum**
<!-- #endregion -->

```python id="MHMMCAVY709I" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304616807, "user_tz": -330, "elapsed": 1172, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="44c09236-14c6-4440-ea72-506ff7d1d7cc"
def squared_distance(value1, value2):
    return (value1 - value2) ** 2

possible_centers = range(measurements.min(), measurements.max() + 1)
penalties = [squared_distance(center, 61) for center in possible_centers]
plt.plot(possible_centers, penalties)
plt.scatter(possible_centers, penalties)
plt.xlabel('Possible Centers')
plt.ylabel('Penalty')
plt.show()
```

<!-- #region id="Tik6k6Gx709J" -->
As the plotted centers shift towards the minimum, the penalty will drop, but their distance to the remaining 6 measurements will increase. Thus, we ought to penalize each potential center based on its squared distance to all 7 recorded measurements.

**Penalizing centers using total sum of squared distances**
<!-- #endregion -->

```python id="LX1Nh68x709J" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304622514, "user_tz": -330, "elapsed": 1032, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1453d2df-596f-446a-a57d-f2135b9e1d4d"
def sum_of_squared_distances(value, measurements):
    return sum(squared_distance(value, m) for m in measurements)

penalties = [sum_of_squared_distances(center, measurements) 
             for center in possible_centers]
plt.plot(possible_centers, penalties)
plt.scatter(possible_centers, penalties)
plt.xlabel('Possible Centers')
plt.ylabel('Penalty')
plt.show()
```

<!-- #region id="q61rsWbb709L" -->
Based on our plot, the temperature of 75 degrees incurs the lowest penalty. We’ll informally
refer to this temperature value as our "least-penalized center".

**Plotting the least-penalized temperature**
<!-- #endregion -->

```python id="0uwmC15t709M" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637304627900, "user_tz": -330, "elapsed": 828, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5fe08c0c-32cb-4c10-827f-78069e0a8be4"
least_penalized = 75
assert least_penalized == possible_centers[np.argmin(penalties)]

plt.plot(range(number_of_days), measurements)
plt.scatter(range(number_of_days), measurements)
plt.axhline(midpoint, color='k', linestyle='--', label='midpoint')
plt.axhline(median, color='g', linestyle='-.', label='median')
plt.axhline(least_penalized, color='r', linestyle='-', 
            label='least penalized center')
plt.legend()
plt.ylabel('Temperature')
plt.show()
```

<!-- #region id="5padCWj0709N" -->
Mathematicians have shown that sum of squared distances error is always minimized by the **average** value of a dataset. Thus, we can compute the least-penalized center directly. We simply need to sum all the elements in
`measurements` and then divide that sum by the array size.

**Computing the least-penalized center using an average value**
<!-- #endregion -->

```python id="zZkdrl0w709O"
assert measurements.sum() / measurements.size == least_penalized
```

<!-- #region id="pcPVn4ed709O" -->
A summed array of values divided by array size is  referred to as the **mean** or average of the array. We can compute an array's mean using NumPy.

**Computing the mean using NumPy**
<!-- #endregion -->

```python id="uwBZJZXO709P"
mean = measurements.mean()
assert mean == least_penalized
assert mean == np.mean(measurements)
assert mean == np.average(measurements)
```

<!-- #region id="Qoyb0t3_709P" -->
The `np.average` method differs from the `np.mean` method because it takes as input an optional `weights` parameter. The `weights` parameter is a list of numeric weights that capture the importance of our the measurements relative to each other.

**Passing weights into `np.average`**
<!-- #endregion -->

```python id="YEm3lQdO709Q"
equal_weights = [1] * 7
assert mean == np.average(measurements, weights=equal_weights)

unequal_weights = [100] + [1] * 6
assert mean != np.average(measurements, weights=unequal_weights) 
```

<!-- #region id="I0iFrMtE709R" -->
The `weights` parameter is useful for computing the mean across duplicate measurements. Lets find the mean of a 10-temperature array where 75 degrees appears 9 times, and 77 degrees appears just once.

**Computing the weighted mean of duplicate values**
<!-- #endregion -->

```python id="E3ji6oiv709R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304635960, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e60bbf32-61f0-4239-a169-cb07ae7c826f"
weighted_mean = np.average([75, 77], weights=[9, 1])
print(f"The mean is {weighted_mean}")
assert weighted_mean == np.mean(9 * [75] + [77]) == weighted_mean
```

<!-- #region id="r_--brlj709S" -->
We can convert our absolute counts of 9 and 1 into relative weights of 900 and 100, and the final value of `weighted_mean` will remain the same. This is also true if the weights are converted into relative probabilities of
0.9 and 0.1.

**Computing the weighted mean of relative weights**
<!-- #endregion -->

```python id="0SQjdJqf709T"
assert weighted_mean == np.average([75, 77], weights=[900, 100])
assert weighted_mean == np.average([75, 77], weights=[0.9, 0.1])
```

<!-- #region id="ZjHdCSa3709U" -->
We can treat probabilities as weights. Consequently, this allows us to compute the mean of
any probability distribution.

**Finding the Mean of a Probability Distribution**

We'll compute the mean of a 20 coin-flip Binomial distribution by passing a `probabilities` array to into the `weights` parameter of `np.average`. 

**Computing the mean of a Binomial distribution**
<!-- #endregion -->

```python id="t-3dL_RS709W" colab={"base_uri": "https://localhost:8080/", "height": 296} executionInfo={"status": "ok", "timestamp": 1637304697104, "user_tz": -330, "elapsed": 1009, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a89d457d-a16f-44ba-ba77-0c1c82dd50d5"
num_flips = 20
interval_all_counts = range(num_flips + 1)
probabilities = stats.binom.pmf(interval_all_counts, 20, prob_head)
mean_binomial = np.average(interval_all_counts, weights=probabilities)
print(f"The mean of the Binomial is {mean_binomial:.2f} heads")
plt.plot(interval_all_counts, probabilities)
plt.axvline(mean_binomial, color='k', linestyle='--')
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

<!-- #region id="QRKkcc3w709X" -->
The mean perfectly captures the Binomial’s centrality. That is why SciPy permits us to obtain the mean of any Binomial simply by calling `stats.binom.mean`.

**Computing the Binomial mean using SciPy**
<!-- #endregion -->

```python id="Zq4keMUS709X"
assert stats.binom.mean(num_flips, 0.5) == 10
```

<!-- #region id="r9ErQzOr709X" -->
Using the stats.binom.mean method, we can rigorously analyze the relationship between Binomial centrality and coin-flip count. Let’s plot the Binomial mean across a range of coin-flip counts spanning from 0 to 500.

**Plotting multiple Binomial means**
<!-- #endregion -->

```python id="eRhbqbbN709Y" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304706722, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="73408e65-7528-4e07-86dc-ed77e83258bd"
means = [stats.binom.mean(num_flips, 0.5) for num_flips in range(500)]
plt.plot(range(500), means)
plt.xlabel('Coin Flips')
plt.ylabel('Mean')
plt.show()
```

<!-- #region id="ZIL8DB6V709Y" -->
With this in mind, let's consider the mean of the single coin-flip Binomial distribution (which is commonly called the **Bernoulli** distribution). The Bernoulli distribution has a coin-flip count of 1, so its mean is equal to 0.5

**Predicting the mean of a Bernoulli distribution**
<!-- #endregion -->

```python id="wprozCA-709Z"
num_flips = 1
assert stats.binom.mean(num_flips, 0.5) == 0.5
```

<!-- #region id="An_FqvVc709a" -->
We can leverage the observed linear relationship to predict the mean of a 1000 coin-flip distribution. We expect that mean to equal 500, and for it to be positioned in the distribution’s center.

**Predicting the mean of a 1000 coin-flip distribution**
<!-- #endregion -->

```python id="xX8pK_lP709a" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304713809, "user_tz": -330, "elapsed": 944, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3e9d4b04-2335-4e64-c743-860a8854bdcf"
num_flips = 1000
assert stats.binom.mean(num_flips, 0.5) == 500

interval_all_counts = range(num_flips)
probabilities = stats.binom.pmf(interval_all_counts, num_flips, 0.5)
plt.axvline(500, color='k', linestyle='--')
plt.plot(interval_all_counts, probabilities)
plt.xlabel('Head-count')
plt.ylabel('Probability')
plt.show()
```

<!-- #region id="CUiNCXRf709b" -->
A distribution’s mean serves as an excellent measure of centrality. Let’s now explore the use of variance as a measure of dispersion.
## Variance as a Measure of Dispersion

Consider a scenario where we measure the summer temperatures in California and Kentucky. We’ll store these hypothetical temperature measurements, and then compute their means.

**Measuring the means of multiple temperature arrays**
<!-- #endregion -->

```python id="pnHUgvrx709c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304731362, "user_tz": -330, "elapsed": 913, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5911928-86b9-435f-d560-0dd1e84b348a"
california = np.array([52, 77, 96])
kentucky = np.array([71, 75, 79])

print(f"Mean California temperature is {california.mean()}")
print(f"Mean Kentucky temperatures is {california.mean()}")
```

<!-- #region id="Rpnr-owq709c" -->
The means of the 2 measurement arrays both equal 75.  Despite this, the 2 measurement arrays are far from equal. The California temperatures are much more dispersed and unpredictable.  We'll visualize this difference in dispersion by plotting the 2 measurement arrays.

**Visualizing the difference in dispersion**
<!-- #endregion -->

```python id="zvcKn-Xm709d" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637304736688, "user_tz": -330, "elapsed": 836, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed72055d-1a53-40fd-a99d-c1e6ee44c7d5"
plt.plot(range(3), california, color='b', label='California')
plt.scatter(range(3), california, color='b')
plt.plot(range(3), kentucky, color='r', linestyle='-.', label='Kentucky')
plt.scatter(range(3), kentucky, color='r')
plt.axhline(75, color='k', linestyle='--', label='Mean')
plt.legend()
plt.show()
```

<!-- #region id="b47TA-_2709d" -->
Within the plot, the 3 Kentucky temperatures nearly overlap with the flat mean. Meanwhile, the majority of California temperatures are noticeably more distant from the mean. Let's penalize the California measurements for being
too distant from their center, using a **sum of squares** penalty.

**Computing California’s sum of squares**
<!-- #endregion -->

```python id="z8AEuzSI709e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304737625, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a725be0-aa73-4321-acd5-c7c495a78d98"
def sum_of_squares(data):
    mean = np.mean(data)
    return sum(squared_distance(value, mean) for value in data)

california_sum_squares = sum_of_squares(california)
print(f"California's sum of squares is {california_sum_squares}")
```

<!-- #region id="cBCq0hZ8709e" -->
California’s sum of squares is 974. We expect Kentucky’s sum of squares to be noticeably lower.

**Computing Kentucky’s sum of squares**
<!-- #endregion -->

```python id="N6gb0TYv709f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304741926, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6b8590e-44f6-47de-884d-49713b5c55bf"
kentucky_sum_squares = sum_of_squares(kentucky)
print(f"Kentucky's sum of squares is {kentucky_sum_squares}")
```

<!-- #region id="Fme0uKxS709g" -->
The sum of squares helps us measure dispersion. However, the
measurement is not perfect. Suppose we duplicate the temperatures within the
California array by recording each temperature twice. The level of dispersion will remain the same even though the sum of squares will double.

**Computing sum of squares after array duplication**
<!-- #endregion -->

```python id="9Sl8qiHL709g" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304744055, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5cc0a467-2e23-4e49-d4aa-df5504689e5f"
california_duplicated = np.array(california.tolist() * 2)
duplicated_sum_squares = sum_of_squares(california_duplicated)
print(f"Duplicated California sum of squares is {duplicated_sum_squares}")
assert duplicated_sum_squares == 2 * california_sum_squares
```

<!-- #region id="aOsdQldU709g" -->
The sum of squares is not a good measure of dispersion because it’s influenced by the size of an inputted array. Fortunately, that influence is easy to eliminate. We simply divide the sum of squares by the array size.

**Dividing sum of squares by array size**
<!-- #endregion -->

```python id="xX8lXiFy709h"
value1 = california_sum_squares / california.size
value2 = duplicated_sum_squares / california_duplicated.size
assert value1 == value2
```

<!-- #region id="F-dNQbl5709h" -->
Dividing sum of squares by the number of measurements produces what statisticians call the **variance**. Conceptually, the variance is equal to the average squared distance from the mean.

**Computing the variance from mean squared distance**
<!-- #endregion -->

```python id="JmbxW4CO709i"
def variance(data):
    mean = np.mean(data)
    return np.mean([squared_distance(value, mean) for value in data])

assert variance(california) == california_sum_squares / california.size
```

<!-- #region id="K7MbYcF1709k" -->
The variances for the california and california_duplicated arrays will equal, since their levels of dispersion are identical.

**Computing the variance after array duplication**
<!-- #endregion -->

```python id="C31FkJGN709l"
assert variance(california) == variance(california_duplicated)
```

<!-- #region id="s_ywbSJx709m" -->
Meanwhile, the variances for the California and Kentucky arrays will retain their 30-fold ratio caused by a difference in dispersion.

**Comparing the variances of California and Kentucky**
<!-- #endregion -->

```python id="COOy-zbP709m" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304749341, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="58f8bd20-a023-423c-af93-3ef13a6551b5"
california_variance = variance(california)
kentucky_variance = variance(kentucky)
print(f"California Variance is {california_variance}")
print(f"Kentucky Variance is {kentucky_variance}")
```

<!-- #region id="VXw7E_Wk709n" -->
Variance can be computed by calling `np.var` on a Python list or NumPy array. The variance of a NumPy array can also be computed using the array’s built-in `var` method.

**Computing the variance using NumPy**
<!-- #endregion -->

```python id="8dDjJ_6C709n"
assert california_variance == california.var()
assert california_variance == np.var(california)
```

<!-- #region id="4gk6hfGv709n" -->
Variance is dependent on the mean. If we compute a weighted mean, then we must also compute a weighted variance. The weighted variance is simply the weighted average of all the squared distances from the weighted mean. 

**Computing the weighted variance using np.average**
<!-- #endregion -->

```python id="8GLx29BX709o"
def weighted_variance(data, weights):
    mean = np.average(data, weights=weights)
    squared_distances = [squared_distance(value, mean) for value in data]
    return np.average(squared_distances, weights=weights)

assert weighted_variance([75, 77], [9, 1]) == np.var(9 * [75] + [77]) 
```

<!-- #region id="tS4-47RH709o" -->
The `weighted_variance` function can take as its input an array of probabilities. This allows us to compute the variance of any probability distribution.

**Finding the Variance of a Probability Distribution**

Lets compute variance of the Binomial distribution associated with 20 fair coin-flips. We’ll run the computation by assigning a `probabilities array` to the `weights` parameter of `weighted_variance.`

**Computing the variance of a Binomial distribution**
<!-- #endregion -->

```python id="kUthbBaT709p" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304766807, "user_tz": -330, "elapsed": 782, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="123b2a6c-b20d-4702-c9fe-72d998b4fcfe"
interval_all_counts = range(21)
probabilities = stats.binom.pmf(interval_all_counts, 20, prob_head)
variance_binomial = weighted_variance(interval_all_counts, probabilities)
print(f"The variance of the Binomial is {variance_binomial:.2f} heads")
```

<!-- #region id="tC3yBlFE709p" -->
The Binomial’s variance is 5, which is equal to half the Binomial’s mean. That variance can be computed more directly using SciPy’s `stats.binom.var` method.

**Computing the Binomial variance using SciPy**
<!-- #endregion -->

```python id="jENgdudv709q"
assert stats.binom.var(20, prob_head) == 5.0
assert stats.binom.var(20, prob_head) == stats.binom.mean(20, prob_head) / 2
```

<!-- #region id="QLu5v6ou709q" -->
Using the `stats.binom.var` method, we can rigorously analyze the relationship between Binomial dispersion and coin-flip count. Let's plot the Binomial variance across a range of coin-flip counts spanning from 0 to 500. 

**Plotting multiple Binomial variances**
<!-- #endregion -->

```python id="sG2sayG2709r" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304772340, "user_tz": -330, "elapsed": 1223, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="03aabb12-6498-485b-85c8-23e9740a25a4"
variances = [stats.binom.var(num_flips, prob_head) 
             for num_flips in range(500)]
plt.plot(range(500), variances)
plt.xlabel('Coin Flips')
plt.ylabel('Variance')
plt.show()
```

<!-- #region id="PeCc59tJ709r" -->
The variance is equal to one-fourth the coin-flip count. Thus, the Bernoulli distribution has a
variance of 0.25, because its coin-flip count is 1. By this logic, we can expect a variance of 250 for a 1000 coin-flip distribution.

**Predicting Binomial variances**
<!-- #endregion -->

```python id="ejK81QjQ709s"
assert stats.binom.var(1, 0.5) == 0.25
assert stats.binom.var(1000, 0.5) == 250
```

<!-- #region id="YbwVTxFp709s" -->
The variance is powerful measure of data dispersion. However, statisticians often use an alternative measure, which they call the **standard deviation**. The standard deviation is equal to the square-root of the variance.

**Computing the standard deviation**
<!-- #endregion -->

```python id="xjT1UDcj709s"
data = [1, 2, 3]
standard_deviation = np.std(data)
assert standard_deviation ** 2 == np.var(data)
```

<!-- #region id="BhhpYoBV709t" -->
## Manipulating the Normal Distribution Using SciPy

Let's generate a Normal distribution by plotting a histogram of coin-flip samples. Each sample will represent 10,000 flipped coins. If we use the sample size to divide the sum of values in the sample, we will compute the observed head-count frequency. Conceptually, this frequency is equal to simply taking the sample's mean.

**Computing head-count frequencies**
<!-- #endregion -->

```python id="2oIdNuLc709t"
np.random.seed(0)
sample_size = 10000
sample = np.array([np.random.binomial(1, 0.5) for _ in range(sample_size)])
head_count = sample.sum()
head_count_frequency = head_count / sample_size
assert head_count_frequency == sample.mean()
```

<!-- #region id="i8FCoe9L709u" -->
We can compute 100,000 head-count frequencies in just a single line of code.

**Listing 6.2. Computing 100,000 head-count frequencies**
<!-- #endregion -->

```python id="PDjJ0rfH709u"
np.random.seed(0)
frequencies = np.random.binomial(sample_size, 0.5, 100000) / sample_size
```

<!-- #region id="CTj52PKA709v" -->
Each sampled frequency equals the mean of 10,000 randomly flipped coins. Therefore, we’ll rename our frequencies variable as `sample_means`. We’ll then proceed to visualize our `sample_means` data as a histogram.

**Visualizing sample means in a histogram**
<!-- #endregion -->

```python id="G0Wow9th709v" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304792495, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="73c48a2d-9eb8-44e5-8e2a-4c29740e747f"
sample_means = frequencies
likelihoods, bin_edges, _ = plt.hist(sample_means, bins='auto', 
                                     edgecolor='black', density=True)
plt.xlabel('Binned Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="qWybuogY709v" -->
The histogram is shaped like a Normal distribution. Let’s calculate the distribution’s mean
and standard deviation.

**Computing mean and standard deviation of a histogram**
<!-- #endregion -->

```python id="neF-15kz709w" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304794977, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="804ee717-557d-490a-93ac-e8e766189eef"
mean_normal = np.average(bin_edges[:-1], weights=likelihoods)
var_normal = weighted_variance(bin_edges[:-1], likelihoods)
std_normal = var_normal ** 0.5
print(f"Mean is approximately {mean_normal:.2f}")
print(f"Standard deviation is approximately {std_normal:.3f}")
```

<!-- #region id="jpfQHnfx709w" -->
The distribution’s mean is approximately 0.5, and its standard deviation is approximately 0.005. In a Normal distribution, these values can be computed directly from the distribution’s peak.

**Computing mean and standard deviation from peak coordinates**
<!-- #endregion -->

```python id="TMD4dKOf709x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304806803, "user_tz": -330, "elapsed": 643, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b37199fe-38fe-4441-aa47-62f19de153e1"
import math
peak_x_value = bin_edges[likelihoods.argmax()]
print(f"Mean is approximately {peak_x_value:.2f}")
peak_y_value = likelihoods.max()
std_from_peak = (peak_y_value * (2* math.pi) ** 0.5) ** -1
print(f"Standard deviation is approximately {std_from_peak:.3f}")
```

<!-- #region id="9CaZan3x709x" -->
Additionally, we can compute the mean and standard deviation simply by calling `stats.norm.fit(sample_means)`.

**Computing mean and standard deviation using `stats.norm.fit`**
<!-- #endregion -->

```python id="wZiFZDw_709y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304810304, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35114bfe-b715-4390-8847-5ddd96edc352"
fitted_mean, fitted_std = stats.norm.fit(sample_means)
print(f"Mean is approximately {fitted_mean:.2f}")
print(f"Standard deviation is approximately  {fitted_std:.3f}")
```

<!-- #region id="Tt9QeLIl709y" -->
The computed mean and standard deviation can be used to reproduce our Normal curve. We can regenerate the curve calling `stats.norm.pdf(bin_edges, fitted_mean, fitted_std)`.

**Computing Normal likelihoods using `stats.norm.pdf`**
<!-- #endregion -->

```python id="u3tOhbbs709z" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304818090, "user_tz": -330, "elapsed": 5027, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ff74bffa-462a-4bf7-e366-4689abc4d6c7"
normal_likelihoods = stats.norm.pdf(bin_edges, fitted_mean, fitted_std)
plt.plot(bin_edges, normal_likelihoods, color='k', linestyle='--', 
         label='Normal Curve')
plt.hist(sample_means, bins='auto', alpha=0.2, color='r', density=True)
plt.legend()
plt.xlabel('Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="nVOzzxes7093" -->
The plotted curve's peak sits at an x-axis position of 0.5, and rises to a y-axis position of approximately 80. Lets shift the peak 0.01 units to the right, while also doubling the peak's height. 

**Manipulating a Normal curve’s peak coordinates**
<!-- #endregion -->

```python id="LtJEP6E17093" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304819161, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b8b7da3-86cf-4d39-b0d2-0b0b5f1f1ae8"
adjusted_likelihoods = stats.norm.pdf(bin_edges, fitted_mean + 0.01, 
                                      fitted_std / 2)
plt.plot(bin_edges, adjusted_likelihoods, color='k', linestyle='--')
plt.hist(sample_means, bins='auto', alpha=0.2, color='r', density=True)
plt.xlabel('Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="3CTEAsC87094" -->
**Comparing Two Sampled Normal Curves**

Let's quadruple the coin-flip sample size to 40,000 and plot the resulting distribution changes. Below, we compare the plotted shapes of the old and updated Normal distributions, which we'll label as A and B, respectively. 

**Plotting two curves with different samples sizes**
<!-- #endregion -->

```python id="id1t7j6g7094" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304819164, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82158fd7-cc5d-4384-cd3c-7beb3490a2cf"
np.random.seed(0)
new_sample_size = 40000
new_head_counts = np.random.binomial(new_sample_size, 0.5, 100000)
new_mean, new_std = stats.norm.fit(new_head_counts / new_sample_size)
new_likelihoods = stats.norm.pdf(bin_edges, new_mean, new_std)
plt.plot(bin_edges, normal_likelihoods, color='k', linestyle='--', 
         label='A: Sample Size 10K')
plt.plot(bin_edges, new_likelihoods, color='b', label='B: Sample Size 40K')
plt.legend()
plt.xlabel('Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="bPZA8IsW7095" -->
Both Normal distributions are centered around the sample mean value of 0.5. This represents an estimate of the true Bernoulli mean. Let's calculate the 95% confidence interval for the true Bernoulli mean, using Normal Distribution B. SciPy allows us to compute that interval by calling `stats.norm.interval(0.95, mean, std)`. 

**Computing a confidence interval using SciPy**
<!-- #endregion -->

```python id="rI9MT4FI7095" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304819952, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="135ccd35-b50b-4c60-b7b9-5ffe30baaedb"
mean, std = new_mean, new_std
start, end = stats.norm.interval(0.95, mean, std)
print(f"The true mean of the sampled binomial distribution is between {start:.3f} and {end:.3f}")
```

<!-- #region id="JJxdXKH_7096" -->
We are 95% confident that the true mean of our sampled Bernoulli distribution is between 0.495 and 0.505. In fact, that mean is equal to exactly 0.5. We can confirm this using SciPy.

**Confirming the Bernoulli mean**
<!-- #endregion -->

```python id="DSbh04XC7097"
assert stats.binom.mean(1, 0.5) == 0.5
```

<!-- #region id="EUUJPzNI7098" -->
Let's now attempt to estimate the variance of the Bernoulli distribution based on the plotted Normal curves. The peak of Distribution B is twice as high as the peak of Distribution A. This height is inversely proportional to the standard deviation. Thus, we can infer that the variance of Distribution B is one-fourth the variance of Distribution A. 

**Assessing shift in variance after increased sampling**
<!-- #endregion -->

```python id="EsP73u9K7098" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304823422, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea8c54a1-9941-4033-eab7-48027095d1a2"
variance_ratio = (new_std ** 2) / (fitted_std ** 2) 
print(f"The ratio of variances is approximately {variance_ratio:.2f}")
```

<!-- #region id="8mG6F-Ds7099" -->
It appears that variance is inversely proportional to sample size. If so, than a four-fold decrease in sample size from 10,000 to 2500 should generate a four-fold increase in the variance.

**Assessing shift in variance after decreased sampling**
<!-- #endregion -->

```python id="sANMbyS27099" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304824226, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d221d86d-9ccb-4ba4-be80-61281cdff812"
np.random.seed(0)
reduced_sample_size = 2500
head_counts = np.random.binomial(reduced_sample_size, 0.5, 100000)
_, std = stats.norm.fit(head_counts / float(reduced_sample_size))
variance_ratio = (std ** 2) / (fitted_std ** 2) 
print(f"The ratio of variances is approximately {variance_ratio:.1f}")
```

<!-- #region id="Vhf07ujh709-" -->
A four-fold decrease in the sample size leads to a four-fold increase in the variance. Thus, if we decrease the sample size from 10,000 to 1, we can expect 10,000-fold increase in the variance.

**Predicting variance for a sample size of 1**
<!-- #endregion -->

```python id="0hP4W02z709-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304825985, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af088bd9-1337-44ee-b91c-8718c2566fb3"
estimated_variance = (fitted_std ** 2) * 10000
print(f"Estimated variance for a sample size of 1 is {estimated_variance:.2f}")
```

<!-- #region id="ZTMfCirU709-" -->
If the sample size were 1, then our `sample_means`  array would simply be a sequence of randomly recorded ones and zeroes. By definition, that array would represent the output of the Bernoulli distribution. Consequently, our estimated variance for a sample size of 1 equals the variance of the Bernoulli distribution. 

**Confirming the predicted variance for a sample size of 1**
<!-- #endregion -->

```python id="0AjBr9V3709_"
assert stats.binom.var(1, 0.5) == 0.25
```

<!-- #region id="P1DmjAX-709_" -->
According the Central Limit Theorem, sampling mean-values from almost any distribution will produce a Normal curve. The mean of the Normal curve will approximate the mean of the underlying distribution. Also, the variance of the Normal curve multiplied by the sample size will approximate the variance of the underlying distribution. Using this relationship, we can estimate both mean and variance of almost any distribution through random sampling.

## Determining Mean and Variance of a Population through Random Sampling

Suppose we are tasked with finding the average age of people living in a town. The town's population is exactly 50,000 people. Below, we'll simulate the ages of the townsfolk using the `np.random.randint` module.
<!-- #endregion -->

```python id="udVmi5AW70-A"
np.random.seed(0)
population_ages = np.random.randint(1, 85, size=50000)
```

<!-- #region id="LtGdEs1370-A" -->
The mean of the entire population is the **population mean**. The variance of an entire population is the **population variance**. Let's quickly compute the population mean and population variance of our simulated town.

**Computing population mean and variance**
<!-- #endregion -->

```python id="9bOAqpKc70-B"
population_mean = population_ages.mean()
population_variance = population_ages.var()
```

<!-- #region id="rhdbYyv770-B" -->
Computing the population mean is easy when we have simulated data. However, obtaining that data in real life would be incredibly time consuming. A simpler approach would be to interview 10 randomly chosen people in the town. We'd record the ages from this random sample, and afterwards compute the sample mean. Let's simulate the sampling process by drawing 10 random ages from the `np.random.choice` module.

**Simulating 10 interviewed people**
<!-- #endregion -->

```python id="JIYO_dJu70-B"
np.random.seed(0)
sample_size = 10
sample = np.random.choice(population_ages, size=sample_size)
sample_mean = sample.mean()
```

<!-- #region id="udQozubB70-C" -->
Of course, our sample mean is likely to be noisy and inexact. We can measure that noise by finding the percent difference between sample_mean and population_mean.

**Comparing sample mean to population mean**
<!-- #endregion -->

```python id="ihL4-j1-70-C" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304869287, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d7ffaf47-406d-4a26-92fe-826113f2a459"
percent_diff = lambda v1, v2: 100 * abs(v1 - v2) / v2    
percent_diff_means = percent_diff(sample_mean, population_mean)
print(f"There is a {percent_diff_means:.2f} percent difference between means.")
```

<!-- #region id="De54HywH70-D" -->
Our sample is insufficient to estimate the mean. Perhaps we should raise our sampling to cover 1,000 residents of the town. We can post an ad in the local paper asking for 100 volunteers. Each volunteer will survey 10 random people, in order to sample their ages. Afterwards, every volunteer will send us a computed sample mean. 

**Computing sample means across 1,000 people**
<!-- #endregion -->

```python id="29H7X-Ii70-E"
np.random.seed(0)
sample_means = [np.random.choice(population_ages, size=sample_size).mean() 
                for _ in range(100)]
```

<!-- #region id="zvV1Jwf370-F" -->
According to the Central Limit Theorem, a histogram of sample means should resemble the Normal distribution. Furthermore, the mean of the Normal distribution should approximate the population mean.

**Fitting sample means to a Normal curve**
<!-- #endregion -->

```python id="0UgWOcCv70-F" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304876906, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="38003f9a-df1b-46d3-85cb-5b5d2805b0ee"
likelihoods, bin_edges, _  = plt.hist(sample_means, bins='auto', alpha=0.2, 
                                      color='r', density=True)
mean, std = stats.norm.fit(sample_means)
normal_likelihoods = stats.norm.pdf(bin_edges, mean, std)
plt.plot(bin_edges, normal_likelihoods, color='k', linestyle='--')
plt.xlabel('Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="xt0kY5bA70-G" -->
The histogram’s shape still approximates a Normal distribution. We’ll print that distribution’s mean and compare it to the population mean.

**Comparing Normal mean to population mean**
<!-- #endregion -->

```python id="_DtEH-D270-G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304883520, "user_tz": -330, "elapsed": 912, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="600637ce-e7c3-46b1-b487-8fc1dbf43563"
print(f"Actual population mean is approximately {population_mean:.2f}")
percent_diff_means = percent_diff(mean, population_mean)
print(f"There is a {percent_diff_means:.2f}% difference between means.")
```

<!-- #region id="joIUe-vs70-H" -->
Our result, while not perfect, is still a very good approximation of the actual average age within the town. Now, we’ll briefly turn our attention to the standard deviation computed from the Normal distribution. We simply need to multiply the computed variance by sample size.

**Estimating the population variance**
<!-- #endregion -->

```python id="WpPMKSS070-H"
normal_variance = std ** 2
estimated_variance = normal_variance * sample_size
```

<!-- #region id="iTnyPrAe70-I" -->
Let’s compare the estimated variance to the population variance.

**Comparing estimated variance to population variance**
<!-- #endregion -->

```python id="h3p0ZZK-70-I" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304887545, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9498a488-6d9f-41aa-e4eb-14d7137352ac"
print(f"Estimated variance is approximately {estimated_variance:.2f}")
print(f"Actual population variance is approximately {population_variance:.2f}")
percent_diff_var = percent_diff(estimated_variance, population_variance)
print(f"There is a {percent_diff_var:.2f} percent difference between variances.")
```

<!-- #region id="KyOO0n8070-I" -->
There is approximately a 1.3% difference between the estimated variance and the population variance. We've thus approximated the town's variance to a relative accurate degree, while only sampling 2% of the people living in the town.

## Making Predictions Using Mean and Variance

Let us now consider a new scenario, in which we analyze a fifth grade classroom.  Mrs. Mann has spent 25 years teaching fifth-grade. Her classroom holds 20 students. Thus, over the years, she has taught 500 students total. Each of those students have taken an assessment exam upon the completion of her class. The mean and variance of their exam grades are 84 and 25, respectively. We'll refer to these values as the population mean an population variance, since they cover the entire population of students who've ever been taught by Mrs. Mann.

**Population mean and variance of recorded grades**
<!-- #endregion -->

```python id="r8Canszf70-J"
population_mean = 84
population_variance = 25
```

<!-- #region id="Y2-YbhWe70-K" -->
Lets model the yearly test results of Mrs. Mann's class as a collection of 20 grades randomly drawn from a distribution with mean `population_mean` and variance `population variance`. According to Central Limit Theorem, the likelihood distribution of mean grades will resemble a Normal curve. The mean of the Normal curve will equal `population_mean`. The variance of the Normal curve will equal  the **SEM**. By definition, the SEM equals the population standard deviation divided by the square root of the sample size. We’ll compute the curve parameters, and plot the Normal curve below.

**Plotting a Normal curve using mean and SEM**
<!-- #endregion -->

```python id="mDmz8avR70-K" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304911707, "user_tz": -330, "elapsed": 931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="86beba8b-048e-483c-dd09-bd8e32c313ca"
mean = population_mean
sem = (population_variance / 20) ** 0.5
grade_range = range(101)
normal_likelihoods = stats.norm.pdf(grade_range, mean, sem)
plt.plot(grade_range, normal_likelihoods)
plt.xlabel('Mean Grade of 20 Students (%)')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="ViZCMy3s70-L" -->
The area beneath the plotted curve approaches zero at values higher than 89%. 
Therefore, the probability of observing a mean grade that's at or above 90% is incredibly low. Still, to be sure, we'll need to somehow accurately measure the area under the Normal distribution.

**Computing the Area Beneath a Normal Curve**

We can estimate the area beneath a Normal curve by subdividing it into small, trapezoidal units. This ancient technique referred to as the **trapezoid rule**.  The trapezoid rule is very easy to execute in just a few lines of code. Alternatively, we can utilize NumPy's `np.trapz` method to take the area of an inputted array. Lets apply the trapezoid rule to our Normal distribution. 

**Approximating the area using the trapezoid rule**
<!-- #endregion -->

```python id="jxtjtB5O70-O" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304915769, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f2c5aca-3b43-42c7-e8da-3b7b89332182"
total_area = np.sum([normal_likelihoods[i: i + 2].sum() / 2
                    for i in range(normal_likelihoods.size - 1)])

assert total_area == np.trapz(normal_likelihoods)
print(f"Estimated area under the curve is {total_area}")
```

<!-- #region id="muwAaTwd70-O" -->
The estimated area is very close to 1.0, but its not exactly equal to 1.0. We can access a mathematically exact solution in SciPy, using the `stats.norm.sf` method. 

**Computing the total area using SciPy**
<!-- #endregion -->

```python id="svnM_PGO70-P"
assert stats.norm.sf(0, mean, sem) == 1.0
```

<!-- #region id="RBXYQT0T70-P" -->
We expect `stats.norm.sf(mean, mean, sem)` to equal 0.5. This is because the mean perfectly splits the Normal curve into 2 equal halves. Meanwhile, we expect `np.trapz(normal_likelihoods[mean:])` to approximate but not fully equal 0.5. Lets confirm below.

**Inputting the mean into the survival function**
<!-- #endregion -->

```python id="t26q8zk770-Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304931622, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cc770076-7e6a-45c5-df11-7a708a938a33"
assert stats.norm.sf(mean, mean, sem) == 0.5
estimated_area = np.trapz(normal_likelihoods[mean:])
print(f"Estimated area beyond the mean is {estimated_area}")
```

<!-- #region id="E9t1xohO70-Q" -->
Now, lets execute stats.norm.sf(90, mean, sem). This will return the area over an interval of values lying beyond 90%. The area represents the likelihood of 20 students jointly acing an exam.

**Computing the probability of a good collective grade**
<!-- #endregion -->

```python id="OocXtaQX70-Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304934311, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="43abad05-3a94-40d3-8d47-67903f576ab4"
area  = stats.norm.sf(90, mean, sem)
print(f"Probability of 20 students acing the exam is {area}")
```

<!-- #region id="OP0CvzWT70-R" -->
## Assessing the Divergence Between Sample Mean and Population Mean

Imagine a scenario where we analyze every fifth grade classroom in North Dakota. All fifth graders in the state are given an identical assessment exam. The exam grades are fed into a database. The population mean and variance of the grades are 80 and 100, respectively.

**Population mean and variance of North Dakota's grades**
<!-- #endregion -->

```python id="Pw-xT2-M70-R"
population_mean = 80
population_variance = 100
```

<!-- #region id="UAU1lNdi70-S" -->
Now, suppose we travel to South Dakota. There, we find a class of 18 students that has outperformed North Dakota's population mean by 4 percentage points. If that high performance is just a random anomaly, then the **null hypothesis** is true. Let's temporarily assume that the null hypothesis is true, and that South Dakota's population mean is equal to North Dakota's population mean. Also, we'll assume that South Dakota's population variance is equal to North Dakota's population variance. Consequently, we can model our 18-student classroom as a random sample taken from a Normal distribution. That distribution's mean will equal `population_mean`. Meanwhile, its standard deviation will equal the SEM.

**Normal curve parameters if the null hypothesis is true**
<!-- #endregion -->

```python id="nZ66dh0770-S"
mean = population_mean
sem = (population_variance / 18) ** 0.5
```

<!-- #region id="oXVfNHLJ70-T" -->
If the null hypothesis is true, then the probability of encountering an average exam grade of at-least 84% is equal to `stats.norm.sf(84 mean, sem)`. We'll now print out that probability.

**Finding the probability of a high-performance grade**
<!-- #endregion -->

```python id="cdLV18m370-T" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304950632, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a9a55916-4f76-4001-f281-92558395e73e"
prob_high_grade = stats.norm.sf(84, mean, sem)
print(f"Probability of an average grade >= 84 is {prob_high_grade}")
```

<!-- #region id="Bvo8MDNI70-U" -->
We will now compute the probability of observing an exam-average that’s less than or equal to 76%. The calculation can be carried out with SciPy’s `stats.norm.cdf` method.

**Finding the probability of a low-performance grade**
<!-- #endregion -->

```python id="1zLPU0TL70-U" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304952596, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="49996711-b569-4a8c-9bb3-504b9ddf6218"
prob_low_grade = stats.norm.cdf(76, mean, sem)
print(f"Probability of an average grade <= 76 is {prob_low_grade}")
```

<!-- #region id="4vbOhHnf70-V" -->
It appears that `prob_low_grade` is exactly equal to `prob_high_grade`. The cumulative distribution and the survival function are mirror images of each other. Thus, `stats.norm.sf(mean + x, mean, sem)` will always equal `stats.norm.cdf(mean - x, mean, sem)` for any input `x`. Below, we will demonstrate this symmetry. 

**Comparing the survival and the cumulative distribution functions**
<!-- #endregion -->

```python id="z8p90IpD70-V" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637304961128, "user_tz": -330, "elapsed": 802, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="42264d37-6514-4ae9-b304-9179cdb77784"
for x in range(-100, 100):
    sf_value = stats.norm.sf(mean + x, mean,  sem)
    assert sf_value == stats.norm.cdf(mean - x, mean, sem)

plt.axvline(mean, color='k', label='Mean', linestyle=':')
x_values = range(60, 101)
plt.plot(x_values, stats.norm.cdf(x_values, mean, sem), 
         label='Cumulative Distribution')
plt.plot(x_values, stats.norm.sf(x_values, mean, sem),
         label='Survival Function', linestyle='--', color='r')
plt.xlabel('Sample Mean')
plt.ylabel('Probability')
plt.legend()
plt.show()
```

<!-- #region id="-T25_evR70-W" -->
Conceptually, the sum of `prob_high_grade` and `prob_low_grade` represents the probability of observing an extreme deviation from the population mean when the null hypothesis is true. Statisticians refer to this null hypothesis-driven probability as the **p-value**.

**Computing the null hypothesis driven p-value**
<!-- #endregion -->

```python id="dRlqZRga70-W" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304966621, "user_tz": -330, "elapsed": 827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a53c257f-8b5e-410e-9112-86f407f28002"
p_value = prob_low_grade + prob_high_grade
assert p_value == 2 * prob_high_grade
print(f"The p-value is {p_value}")
```

<!-- #region id="VS65IIck70-X" -->
What if the average of the South Dakotan class had equaled 85%, not 84%? Will that 1-percent shift influence our p-value output? Let’s find out.

**Computing the p-value for an adjusted sample mean**
<!-- #endregion -->

```python id="itR4j5VA70-X" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304967334, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c952e5b-fb9f-4786-f429-df9cac118025"
def compute_p_value(observed_mean, population_mean, sem):
    mean_diff = abs(population_mean - observed_mean)
    prob_high = stats.norm.sf(population_mean + mean_diff, population_mean, sem)
    return 2 * prob_high

new_p_value = compute_p_value(85, mean, sem)
print(f"The updated p-value is {new_p_value}")
```

<!-- #region id="CimpE83v70-Y" -->
The new p-value is below 0.05. The threshold of 0.05 is called the **significance level**, and p-values below that threshold are deemed to be **statistically significant.** We'll temporarily set the significance level to a very stringent value of 0.001. What would be the minimum grade-average required to trigger this p-value threshold? Let's find out.

**Scanning for a stringent p-value result**
<!-- #endregion -->

```python id="ExMp2Kpm70-Y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637304971117, "user_tz": -330, "elapsed": 701, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df42e4e9-84f9-4f5b-a643-3b673cbcd590"
for grade in range(80, 100):
    p_value = compute_p_value(grade, mean, sem)
    if p_value < 0.001:
        break

print(f"An average grade of {grade} leads to a p-value of {p_value}")
```

<!-- #region id="466AR5WJ70-Z" -->
Our lowering of the cutoff has inevitably exposed us to an increased risk of **type II** errors.
Consequently, we’ll maintain the commonly accepted p-value cutoff of 0.05.
However, we will also proceed with excessive caution in order to avoid erroneously
rejecting the null hypothesis.

## Data Dredging: Coming to False Conclusions through Oversampling

Suppose that North Dakota's state-wide test performance does not diverge from the exam results in the other 49 states.  We travel to Montana,  and choose a random fifth-grade classroom of 18 students. We then compute the classroom's average grade. Since the null hypotheses is secretly true, we can simulate the value of that average grade by sampling from a Normal distribution defined by `mean` and `sem`.

**Randomly sampling Montana’s exam performance**
<!-- #endregion -->

```python id="0-ov3G1P70-Z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305000788, "user_tz": -330, "elapsed": 759, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1b23d96a-2d17-45e3-9f6f-ef395fec27b7"
np.random.seed(0)
random_average_grade = np.random.normal(mean, sem)
print(f"Average grade equals {random_average_grade:.2f}")
```

<!-- #region id="JNiFA29G70-a" -->
The average exam grade in the class equaled approximately 84.16. We can determine if that average is statistically significant by checking if its p-value is less than or equal to 0.05.

**Testing significance of Montana’s exam performance**
<!-- #endregion -->

```python id="6C_Vq7j070-a" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305001523, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6ac0115f-7ac6-4c29-9393-8da35b39d407"
if compute_p_value(random_average_grade, mean, sem) <= 0.05:
    print("The observed result is statistically significant")
else:
    print("The observed result is not statistically significant")
```

<!-- #region id="BA5KfiAD70-b" -->
The average-grade is not statistically significant. We will continue our journey. We'll visit a single 18-student classroom in each of the remaining 48 states. Once we discover a statistically significant p-value, our journey will end.

**Randomly searching for a significant state result**
<!-- #endregion -->

```python id="xhifIWjF70-b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305006066, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac045178-8e1c-44d7-f392-4affee4ae8fa"
np.random.seed(0)
for i in range(1, 49):
    print(f"We visited state {i + 1}")
    random_average_grade = np.random.normal(mean, sem)
    p_value = compute_p_value(random_average_grade, mean, sem)
    if p_value <= 0.05:
        print("We found a statistically significant result.")
        print(f"The average grade was {random_average_grade:.2f}")
        print(f"The p-value was {p_value}")
        break

if i == 48:
    print("We visited every state and found no significant results.")

```

<!-- #region id="th005zae70-c" -->
The fifth state that we visited produced a statistically significant result However, our conclusions are erroneous. We have indulged in the cardinal statistical sin of **data dredging**. In data-dredging, experiments are repeated over and over again, until a statistically significant result is found. 

Avoiding data dredging is not difficult. We must simply choose in advance a finite number of experiments to run. Afterwards, we should set our significance level to 0.05 divided by the planned experiment count. This simple technique is known as the **Bonferonni correction**. Let's repeat our analysis of US exam performance using the Bonferonni correction. We'll start by adjusting the significance level.

**Using the Bonferonni correction to adjust significance**
<!-- #endregion -->

```python id="5pCAIXC670-c"
num_planned_experiments = 49
significance_level = .05 / num_planned_experiments
```

<!-- #region id="POkGBy-870-d" -->
We’ll proceed to re-run our previous analysis. The analysis will terminate if we encounter a
p-value that’s less than or equal to significance_level.

**Re-running an analysis using an adjusted significance level**
<!-- #endregion -->

```python id="YTp9CXdS70-d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305017813, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b121a881-95fa-423f-cd15-09d99723586f"
np.random.seed(0)
for i in range(49):
    random_average_grade = np.random.normal(mean, sem)
    p_value = compute_p_value(random_average_grade, mean, sem)
    if p_value <= significance_level:
        print("We found a statistically significant result.")
        print(f"The average grade was {random_average_grade:.2f}")
        print(f"The p-value was {p_value}")
        break

if i == 48:
    print("We visited every state and found no significant results.")
```

<!-- #region id="f7evwGOh70-e" -->
## Bootstrapping with Replacement: Testing a Hypothesis When the

Consider the following scenario, in which we own a very large aquarium. It holds 20 tropical fish of varying lengths. The fish lengths range from 2 cm to nearly 120 cm. The average fish-length equals 27 cm. We'll represent these fish lengths using the `fish_length` array below.

**Defining lengths of fish in an aquarium**
<!-- #endregion -->

```python id="rXozuI2u70-e"
fish_lengths = np.array([46.7, 17.1, 2.0, 19.2, 7.9, 15.0, 43.4, 
                         8.8, 47.8, 19.5, 2.9, 53.0, 23.5, 118.5, 
                         3.8, 2.9, 53.9, 23.9, 2.0, 28.2])
assert fish_lengths.mean() == 27
```

<!-- #region id="dGN5qz3n70-h" -->
The population mean-length of wild, tropical fish equals 37 cm. There is a sizable 10 cm difference between the population mean and our sample mean. Is that difference statistically significant? We would like to find out, but we don't have a population variance. Thus, we can't compute the SEM. So what should we do? Well, we can implement a technique known as **Bootstrapping with Replacement**. We'll begin the Bootstrapping procedure by removing a random fish from the aquarium, and subsequently measuring its length. 

**Sampling a random fish from the aquarium**
<!-- #endregion -->

```python id="Bkw54AzL70-i"
np.random.seed(0)
random_fish_length = np.random.choice(fish_lengths, size=1)[0]
sampled_fish_lengths = [random_fish_length]
```

<!-- #region id="MlLWtduF70-i" -->
Now, we will place the chosen fish back into the aquarium. Afterwards, we'll repeat the sampling procedure 19 more times until 20 random fish-lengths have been measured.

**Sampling 20 random fish with repetition**
<!-- #endregion -->

```python id="QD5V1rkA70-j"
np.random.seed(0)
for _ in range(20):
    random_fish_length = np.random.choice(fish_lengths, size=1)[0]
    sampled_fish_lengths.append(random_fish_length)
```

<!-- #region id="Xd8UIs3u70-j" -->
The `sampled_fish_lengths` list contains 20 measurements. However, the elements of `fish_lengths` and `sampled_fish_lengths` are not identical. Due to random sampling, the mean values of the array and list are likely to differ.

**Comparing the sample mean to the aquariuam mean**
<!-- #endregion -->

```python id="oeq6wWx770-k" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305028884, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9ab10ec9-f801-40ea-a0fb-2a9455d93053"
sample_mean = np.mean(sampled_fish_lengths)
print(f"Mean of sampled fish-lengths is {sample_mean:.2f} cm")
```

<!-- #region id="siIqU_ml70-l" -->
If we sample another 20 measurements from the aquarium, we can expect the resulting sample mean to also deviate from 27 cm. Let’s confirm this by repeating our sampling using a single line of code; `np.random.choice(fish_lengths, size=20, replace=True)`.

**Sampling with replacement using NumPy**
<!-- #endregion -->

```python id="CM3aqV8g70-m" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305029662, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="afcb1437-20c6-4131-af90-46a44428f72d"
new_sampled_fish_lengths = np.random.choice(fish_lengths, size=20, 
                                            replace=True)
new_sample_mean = new_sampled_fish_lengths.mean()
print(f"Mean of the new sampled fish-lengths is {sample_mean:.2f} cm")
```

<!-- #region id="WS4qpl0b70-m" -->
Our sampled mean-values are randomly distributed Let’s explore the shape of this random distribution by repeating our sampling process 150,000 times.

**Plotting the distribution of 150k sampled means**
<!-- #endregion -->

```python id="UARk9-kW70-n" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1637305036874, "user_tz": -330, "elapsed": 6388, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="382a5bb8-e183-4167-9d12-9b4fcb627741"
np.random.seed(0)
sample_means = [np.random.choice(fish_lengths, 
                                size=20,
                                replace=True).mean()
               for _ in range(150000)]
likelihoods, bin_edges, _ = plt.hist(sample_means, bins='auto', 
                                      edgecolor='black', density=True)
plt.xlabel('Binned Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="q9SrBE9k70-n" -->
The histogram's shape is not symmetric, its left side rises steeper than its right side. Mathematicians refer to this asymmetry as a skew. We can confirm the skew within our histogram, by calling `stats.skew(sample_means))`.

**Computing the skew of an asymmetric distribution**
<!-- #endregion -->

```python id="HUOkihUA70-o"
assert abs(stats.skew(sample_means)) > 0.4 
```

<!-- #region id="x9BjzLKR70-o" -->
Our asymmetric histogram cannot be modeled using a Normal distribution. Instead, we can fit the histogram to a generic distribution using `stats.rv_histogram`. The method will return a `random_variable` SciPy object. The object will contain `pdf`, and `cdf`, and `sf` methods, just like `stats.norm.` Lets plot `random_variable.pdf(bin_edges)`.

**Fitting to data to a generic distribution using SciPy**
<!-- #endregion -->

```python id="grKqrgEm70-p" colab={"base_uri": "https://localhost:8080/", "height": 282} executionInfo={"status": "ok", "timestamp": 1637305036882, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cbb5728a-7a64-4d07-9ace-966783f2fa34"
random_variable = stats.rv_histogram((likelihoods, bin_edges))
plt.plot(bin_edges, random_variable.pdf(bin_edges))
plt.hist(sample_means, bins='auto', alpha=0.1,  color='r', density=True)
plt.xlabel('Sample Mean')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="9Pa5ISZ-70-p" -->
As expected, the probability density function perfectly resembles the histogram shape. Let’s now plot both the cumulative distribution function and the survival function associated with `random_variable`.

**Plotting mean and interval-areas for a generic distribution**
<!-- #endregion -->

```python id="_jyU1IdD70-q" colab={"base_uri": "https://localhost:8080/", "height": 296} executionInfo={"status": "ok", "timestamp": 1637305038060, "user_tz": -330, "elapsed": 1206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba491b06-3b16-4d31-a92c-940e2e8debe4"
rv_mean = random_variable.mean()
print(f"Mean of the distribution is approximately {rv_mean:.2f} cm")

plt.axvline(random_variable.mean(), color='k', label='Mean', linestyle=':')
plt.plot(bin_edges, random_variable.cdf(bin_edges), 
         label='Cumulative Distribution')
plt.plot(bin_edges, random_variable.sf(bin_edges),
         label='Survival', linestyle='--', color='r')
plt.xlabel('Sample Mean')
plt.ylabel('Probability')
plt.legend()
plt.show()
```

<!-- #region id="slmCUh_T70-q" -->
What is probability that 20 fish sampled with replacement produce a mean as extreme as the population mean? Extremeness is defined as a sampled output that's at-least 10 cm away from 27. Summing `random_variable.sf(37)` and `random_variable.cdf(17)` will give us our answer.

**Computing the probability of an extreme sample mean**
<!-- #endregion -->

```python id="2SlZL7XO70-r" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305038061, "user_tz": -330, "elapsed": 53, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7bce6729-706c-414f-8b48-1624923e8d36"
prob_extreme= random_variable.sf(37) + random_variable.cdf(17)
print(f"Probability of observing an extreme sample mean is approximately {prob_extreme:.2f}")
```

<!-- #region id="JgR7Jv1F70-r" -->
It has been shown that sampling with replacement approximates a dataset's SEM Thus, if the null hypothesis is true, then our missing SEM is equal to `random_variable.std`. This give us yet another way of computing the p-value.

**Using Bootstrapping to estimate the SEM**
<!-- #endregion -->

```python id="tr0hs2IJ70-r" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305038064, "user_tz": -330, "elapsed": 41, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="63579a08-479a-419a-d443-3ea9f23dde52"
estimated_sem = random_variable.std()
p_value = compute_p_value(27, 37, estimated_sem)
print(f"P-value computed from estimated SEM is approximately {p_value:.2f}")
```

<!-- #region id="sdR99fLD70-s" -->
Furthermore, we can estimate the p-value simply by computing the frequency of extreme observations.

**Computing the  p-value from direct counts**
<!-- #endregion -->

```python id="j68v0s5670-s" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305038066, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0bbb0e66-8b89-45cf-a24e-3cac42d534dd"
number_extreme_values = 0
for sample_mean in sample_means:
    if not 17 < sample_mean < 37:
        number_extreme_values += 1

p_value =  number_extreme_values / len(sample_means)
print(f"P-value is approximately {p_value:.2f}")
```

<!-- #region id="2uGo_IvI70-t" -->
Bootstrapping with Replacement presupposes the knowledge of a population mean. Unfortunately, in real-life situations, the population mean is rarely known. In the next section, we learn how to compare collected samples when both the population mean and the population variance are unknown.

## Permutation Testing: Comparing Means of Samples when the Population Parameters are Unknown

Suppose our neighbor also owns an aquarium. Her aquarium contains 10 fish, whose average length is 46 cm. We’ll represent these new fish-lengths using the `new_fish_length array` below.

**Defining lengths of fish in a new aquarium**
<!-- #endregion -->

```python id="GUDjG_hH70-t"
new_fish_lengths = np.array([51, 46.5, 51.6, 47, 54.4, 40.5, 43, 43.1, 
                             35.9, 47.0])
assert new_fish_lengths.mean() == 46
```

<!-- #region id="8YGCeGCv70-u" -->
We want to compare the contents of our neighbor’s aquarium with our own. We’ll begin by
measuring the difference between `new_fish_length.mean()` and
`fish_length.mean()`.

**Computing difference between 2 sample means**
<!-- #endregion -->

```python id="UttPt7O870-u" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305047052, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8a9a1916-788b-4123-deb4-96776becb7f6"
mean_diff = abs(new_fish_lengths.mean() - fish_lengths.mean())
print(f"There is a {mean_diff:.2f} cm difference between the two means")
```

<!-- #region id="MK1zgftp70-u" -->
We will now to carry out a **Permutation test**, where `mean_diff` is leveraged to compute statistical significance. We'll begin the Permutation test by placing all 30 fish into a single aquarium. The unification of our fish can be modeled using the `np.hstack` method.

**Merging 2 arrays using `np.haystack`**
<!-- #endregion -->

```python id="rJftEoW770-v"
total_fish_lengths = np.hstack([fish_lengths, new_fish_lengths])
assert total_fish_lengths.size == 30
```

<!-- #region id="ZLY0UwhH70-v" -->
Once the fish are grouped together, we will allow them to swim in random directions. We’ll use the `np.random.shuffle` method to shuffle the positions of the fish.

**Shuffling the positions of merged fish**
<!-- #endregion -->

```python id="t6rsvo3B70-w"
np.random.seed(0)
np.random.shuffle(total_fish_lengths)
```

<!-- #region id="ph020bY470-w" -->
Now, we'll choose 20 of our randomly shuffled fish, and place them in a separate aquarium. Once more, we'll have 20 fish in aquarium A and 10 fish in aquarium B. However, the mean-lengths of the fish in each aquarium will probably differ from `fish_lengths.mean(`) and `new_fish_lengths.mean()`. Consequently, the difference between mean fish-lengths will also change.

**Computing difference between 2 random sample means**
<!-- #endregion -->

```python id="I7OAUUL270-x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305051607, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1b2c9be-cab1-4d13-8278-31e7d13ed45e"
random_20_fish_lengths = total_fish_lengths[:20]
random_10_fish_lengths = total_fish_lengths[20:]
mean_diff = random_20_fish_lengths.mean() - random_10_fish_lengths.mean() 
print(f"The sampled difference between mean fish lengths is {mean_diff}")
```

<!-- #region id="FtV_SGYk70-x" -->
Not surprisingly, `mean_diff` is fluctuating random variable. We therefore can proceed to find
its distribution through random sampling. Below, will repeat our fish-shuffling procedure
30,000 times in order to obtain a histogram of `mean_diff` values.

**Plotting the distribution of the fluctuating difference between means**
<!-- #endregion -->

```python id="XQrgpK3q70-y" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637305051611, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cc230db6-5fac-4c04-b5c9-8819d7f14f10"
np.random.seed(0)
mean_diffs = []
for _ in range(30000):
    np.random.shuffle(total_fish_lengths)
    mean_diff = total_fish_lengths[:20].mean() - total_fish_lengths[20:].mean()
    mean_diffs.append(mean_diff)
    
likelihoods, bin_edges, _ = plt.hist(mean_diffs, bins='auto', 
                                      edgecolor='black', density=True)
plt.xlabel('Binned Mean Difference')
plt.ylabel('Relative Likelihood')
plt.show()
```

<!-- #region id="qL2e89v470-y" -->
Next, we will fit the histogram to a random variable using the `stats.rv_hist` method.

**Fitting the histogram to a random variable**
<!-- #endregion -->

```python id="PIadJ6To70-y"
random_variable = stats.rv_histogram((likelihoods, bin_edges))
```

<!-- #region id="6AGIJIbU70-z" -->
We want to know the probability of observing an extreme value when the null hypothesis is true.
We’ll define extremeness as a difference between means whose absolute value is at-least 19
cm. Thus, our p-value will equal `random_variable.cdf(-19) + random_variable.sf(19)`.

**Computing the Permutation p-value**
<!-- #endregion -->

```python id="6Z8Y0nxy70-z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305053371, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e39a1b0-a81e-4c32-b184-3957b66ba834"
p_value = random_variable.sf(19) + random_variable.cdf(-19)
print(f"P-value is approximately {p_value:.2f}")
```

<!-- #region id="K2-y0g-470-0" -->
As an aside, we can simplify our Permutation test by leveraging the Law of Large Numbers. We simply need to compute the frequency of extreme recorded samples, just like we did with Bootstrapping with Replacement.

**Computing the Permutation p-value from direct counts**
<!-- #endregion -->

```python id="VyRXSzyP70-4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305054081, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a7ea8fd5-5182-42d6-84c1-c9bc292a522f"
number_extreme_values = 0.0
for min_diff in mean_diffs:
    if not -19 < min_diff < 19:
        number_extreme_values += 1

p_value =  number_extreme_values / len(mean_diffs)
print(f"P-value is approximately {p_value:.2f}")
```

<!-- #region id="JH5DuDig70-5" -->
## Storing Tables Using Basic Python

Let's define a sample table in Python. The table will store measurements for various species of fish. The measurements will cover both length and width, in centimeters. We'll represent this table as a dictionary. 

**Storing a table using Python data-structures**
<!-- #endregion -->

```python id="j42eoKGZ70-5"
fish_measures = {'Fish': ['Angelfish', 'Zebrafish', 'Killifish', 'Swordtail'],
                 'Length':[15.2, 6.5, 9, 6],
                 'Width': [7.7, 2.1, 4.5, 2]}
```

<!-- #region id="_6Vi0gYC70-6" -->
Suppose we want to know the length of a zebrafish. To obtain the length, we must first access the index of the `'Zebrafish'` element in `fish_measures['Fish']`. Afterwards, we'll need to check index in `fish_measures['Length']`.

**Accessing table columns using a dictionary**
<!-- #endregion -->

```python id="PeUKgrgK70-6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305066601, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75775d22-ff3f-46b6-a30a-efa73915f3ea"
zebrafish_index = fish_measures['Fish'].index('Zebrafish')
zebrafish_length = fish_measures['Length'][zebrafish_index]
print(f"The length of a zebrafish is {zebrafish_length:.2f} cm")
```

<!-- #region id="Q0WrRXkX70-7" -->
Our dictionary representation is functional, but is also difficult to use. A better solution is to use the  external Pandas library.

We'll proceed to install the Pandas library. Once Pandas is installed, we will import it as `pd`, using common Pandas usage convention.

**Importing the Pandas library**
<!-- #endregion -->

```python id="iQ43v4MX70-7"
import pandas as pd
```

<!-- #region id="RtEW5pAE70-7" -->
We’ll now load our fish_measures tables into Pandas. This can be done by calling `pd.DataFrame(fish_measures)`.

**Loading a table into Pandas**
<!-- #endregion -->

```python id="oSmvnXMu70-8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305092948, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ccd713a7-1742-4f54-bc5f-7cb3d3ea3272"
df = pd.DataFrame(fish_measures)
print(df)
```

<!-- #region id="-N_J7mKY70-8" -->
The complete table contents are visible in the printed output. For larger tables, we might prefer to only print the first few rows. Calling `print(df.head(x))` will print out just the first `x` rows within a table.

**Accessing the first 2 rows of a table**
<!-- #endregion -->

```python id="986fPOxS70-8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305109467, "user_tz": -330, "elapsed": 1169, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="37a61aba-182f-45de-91fc-ae30ce01fb38"
print(df.head(2))
```

<!-- #region id="YD63gnqx70-9" -->
Sometimes, the best way to summarize a larger Pandas table is to execute the `pandas.describe()` method. By default, the method will generate statistics for all numeric columns within the table.

**Summarizing the numeric columns**
<!-- #endregion -->

```python id="Bh5aVq1U70-9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305112064, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ced66f9a-75b8-4abc-a429-a4c4de036b81"
print(df.describe())
```

<!-- #region id="wXxWF5aw70--" -->
Myriad statistical information is included in the output. Sometimes, the extra information is not very useful. If all we care about is the mean, then we can omit all other outputs by calling `df.mean()`.

**Computing the column mean**
<!-- #endregion -->

```python id="qkvDK1xL70--" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305112713, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e4bb03b2-a2ea-4da3-e104-a7e47de279f6"
print(df.mean())
```

<!-- #region id="3oUtH7jK70--" -->
The `df.describe()` method is primarily intended to be executed on numeric columns. However, we can force it to process strings by calling `df.describe(include=[np.object])`.

**Summarizing the string columns**
<!-- #endregion -->

```python id="9OsLf80V70-_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305113942, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd96d25f-5636-41b9-f397-da9562abad05"
print(df.describe(include=[np.object]))
```

<!-- #region id="szBt_E2470-_" -->
Pandas stores all data in NumPy for quick manipulation. We can easily retrieve the underlying NumPy array by accessing `df.values`.

**Retrieving the table as a 2D NumPy array**

== 8.3. Retrieving Table Columns
Let’s turn our attention to retrieving individual columns. The columns can be accessed using their column names. We can output all possible column names by calling `print(df.columns)`.

**Accessing all column names**
<!-- #endregion -->

```python id="PKQGW_-h70_A" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305161804, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a2a33d77-0ca9-4d40-d364-891115eb46f3"
print(df.columns)
```

<!-- #region id="RU09fQtv70_A" -->
Now let’s print all data stored in the column Fish. We’ll do this by accessing `df.Fish`.

**Accessing an individual column**
<!-- #endregion -->

```python id="3F97ot0i70_A" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305156356, "user_tz": -330, "elapsed": 1090, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b64bd518-c0f4-4671-d117-512b9c7bef91"
print(df.Fish)
```

<!-- #region id="aVGmwlEM70_B" -->
Please note that the printed output is not a NumPy array. In order to print a NumPy array, we must run `print(df.Fish.values)`.

**Retrieving a column as a NumPy array**
<!-- #endregion -->

```python id="JchFqqqN70_B" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305156357, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f34cc7e2-2be4-4eab-8584-1161ff096546"
print(df.Fish.values)
assert type(df.Fish.values) == np.ndarray
```

<!-- #region id="9nzq728z70_C" -->
We can also access Fish using a  dictionary-style bracket representation. Below, we’ll print `df['Fish']`.

**Accessing a column using brackets**
<!-- #endregion -->

```python id="iS3RO4aa70_C" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305167153, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="247182ad-c822-4695-ce91-977c13c89fe6"
print(df['Fish'])
```

<!-- #region id="tNzXCBJ370_D" -->
The bracket representation allows us to retrieve multiple columns at once. If we wish to retrieve multiple columns, we simply execute `df[name_list]`, where name_list represents a list of column names.

**Accessing multiple column using brackets**
<!-- #endregion -->

```python id="I_IZjT2870_D" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305169882, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5269d2ba-136f-470b-81a7-dde6d8a124dd"
print(df[['Fish', 'Length']])
```

<!-- #region id="8Ar2KMUV70_E" -->
We can analyze data stored within df in variety of useful ways. We could, for instance, sort our rows based on a value of single column.

**Sorting rows by column value**
<!-- #endregion -->

```python id="WI8Ru9jt70_E" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305170959, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea658bd1-a101-4fa2-e740-3429e9298723"
print(df.sort_values('Length'))
```

<!-- #region id="_b9lwU7970_F" -->
Furthermore, we can leverage values within columns to filter out unwanted rows. For example, calling `df[df.Width >= 3]` will return a table whose rows contain a width of at-least 3 cm.

**Filtering rows by column value**
<!-- #endregion -->

```python id="d0XCiUdD70_F" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305171896, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc838fe1-aedd-4e25-9204-5c75ac08602a"
print(df[df.Width >= 3])
```

<!-- #region id="8jCQvQPs70_F" -->
## Retrieving Table Rows

Unlike columns, our rows do not have preassigned label values. To compensate, Pandas assigns a special index value for each row. The index for the Angelfish row is 0, and the index for the Swordtail row is 3. We can access these rows by calling `df.loc[[0, 3]]`.

**Accessing rows by index**
<!-- #endregion -->

```python id="qZkEf87A70_G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305171898, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a052b344-2031-401b-d743-e69502dd1761"
print(df.loc[[0, 3]])
```

<!-- #region id="_in9fM-L70_G" -->
Suppose we want to retrieve those rows whose Fish column contains either `'Angelfish'` or `'Swordtail'`. We need to execute `df[booleans']`, where `booleans` is list of `True` or `False` values that are `True` if they match a row of interest. How do we obtain the `booleans` list? One naïve approach is to iterate over `df.Fish`, returning `True` if a column-value appears in `['Angelfish', 'Swordtail']`. 

**Accessing rows by column value**
<!-- #endregion -->

```python id="90LVjBgH70_H" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305172582, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="41b8f377-e493-4d96-b207-c4a29ae448ba"
booleans = [name in ['Angelfish', 'Swordtail']
            for name in df.Fish]
print(df[booleans])
```

<!-- #region id="6iZ8tA3V70_H" -->
We can more concisely locate these rows by leveraging the Pandas `isin` method. Calling `df.Fish.isin(['Angelfish', Swordtail'])`will return an analogue of our previously computed `booleans` list.

**Accessing rows by column value using `isin`**
<!-- #endregion -->

```python id="SEk3U7D470_I" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305173761, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="964b295b-f9a7-4adf-d035-a66b0e313e1c"
print(df[df.Fish.isin(['Angelfish', 'Swordtail'])])
```

<!-- #region id="5dU8k7t370_L" -->
Let's remedy replace the row indices with species. We'll swap numbers for species-names using the `df.set_index` method.

**Swapping row indices for column values**
<!-- #endregion -->

```python id="sRDCLEHN70_L" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305174772, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="494fea9a-59f7-427c-df56-f07fbcda5812"
df.set_index('Fish', inplace=True)
print(df)
```

<!-- #region id="kJL3DLfo70_M" -->
The left-most index column is no longer numeric. It has been replaced with species-names. We can now access the Angelfish and Swordtail columns by running `df.loc[['Angelfish', 'Swordtail']`.

**Accessing rows by string index**
<!-- #endregion -->

```python id="pCd9L6xR70_M" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305175713, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4fba6411-e0f6-47d9-f888-bd4d169609ef"
print(df.loc[['Angelfish', 'Swordtail']])
```

<!-- #region id="3ILcV3FW70_M" -->
## Modifying Table Rows and Columns

What will happen if we swap our rows and columns? We can find out by running `df.T`. The T stands for **transpose**. In a transpose operation, the elements of a table are flipped around its diagonal so that the rows and columns are switched.

**Swapping rows and columns**
<!-- #endregion -->

```python id="pIjcVwfg70_N" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305225093, "user_tz": -330, "elapsed": 1288, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ce6eca29-8c7f-4012-9330-ce72450bc55e"
df_transposed = df.T
print(df_transposed)
```

<!-- #region id="QkCBsVW-70_N" -->
Each column now refers to an individual species of fish. Meanwhile, each row refers to a particular measurement type. Thus, calling `print(df_transposed.Swordtail)` will print out the swordtail’s length and width.

**Printing a transposed column**
<!-- #endregion -->

```python id="qKgb6GMb70_O" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305228663, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b3e0a7ff-ce4e-425d-f7cf-6695351a8202"
print(df_transposed.Swordtail)
```

<!-- #region id="AZsruV9i70_O" -->
Let's try to modify our transposed table. We'll add the measurements of a clownfish to `df_transposed` by running `df_transposed['Clownfish'] = [10.6, 3.7]`.

**Adding a new column**
<!-- #endregion -->

```python id="DzrBwEFe70_O" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305232529, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b0f9876-8ec2-4966-8d9e-06b73c061d5d"
df_transposed['Clownfish'] = [10.6, 3.7]
print(df_transposed)
```

<!-- #region id="JnOvTzkD70_P" -->
Alternatively, we can assign new columns using the `df_transposed.assign` method. The method lets us add multiple columns by passing in more than one column name.

**Adding multiple new columns**
<!-- #endregion -->

```python id="1YZjrACU70_P" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305236026, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ea55728-2295-4bb4-ad5f-62c9822df50c"
df_new = df_transposed.assign(Clownfish2=[10.6, 3.7], Clownfish3=[10.6, 3.7])
assert 'Clownfish2' not in df_transposed.columns
assert 'Clownfish2' in df_new.columns
print(df_new)
```

<!-- #region id="bGA6xGqT70_Q" -->
Our newly added columns are redundant. We'll delete these columns by calling `df_new.drop(columns=['Clownfish2', 'Clownfish3'], inplace=True)`.

**Deleting multiple columns**
<!-- #endregion -->

```python id="ggK_1CNM70_Q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305237631, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4cc27880-aa89-4a68-f9fa-1df2a4ef6d78"
df_new.drop(columns=['Clownfish2', 'Clownfish3'], inplace=True)
print(df_new)
```

<!-- #region id="clUuTb4R70_Q" -->
Lets utilize our stored measurements in order to compute the surface area of each fish. To find the areas, we must iterate over the values in every column, by executing `df_new.items()`. 

**Iterating over column values**
<!-- #endregion -->

```python id="gJFZKAIY70_R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305238362, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8b2b0d9e-8f13-4660-95c3-391e22b9a610"
areas = []
for fish_species, (length, width) in df_new.items():
    area = math.pi * length * width / 4
    print(f"Area of {fish_species} is {area}")
    areas.append(area)
```

<!-- #region id="2Ks9B8c970_R" -->
Let's add the computed areas to our table. We can augment a new Area row by executing `df_new.loc['Area'] = areas`. Afterwards, we'll need to run `df_new.reindex()` to update the row indices with the added Area name.

**Adding a new row**
<!-- #endregion -->

```python id="ljz9kF1C70_R" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305239204, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c05f46d-5cc9-487e-c46d-4ccf642194a5"
df_new.loc['Area'] = areas
df_new.reindex()
print(df_new)
```

```python id="w14V5UND70_S" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305240131, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a17a6fb3-39e7-4301-980a-44a60dbdc874"
row_count, column_count = df_new.shape
print(f"Our table contains {row_count} rows and {column_count} columns")
```

<!-- #region id="cLc6Y6yC70_S" -->
Our updated table contains 3 rows and 5 columns. We can confirm this using `df_new.shape`.

**Checking table shape**
<!-- #endregion -->

```python id="3XT8AaIR70_T" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305242586, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="78070f88-c387-496f-de69-6720465c7f4d"
df_new.to_csv('Fish_measurements.csv')
with open('Fish_measurements.csv') as f:
    print(f.read())
```

<!-- #region id="MVK3W3_Q70_T" -->
## Saving and Loading Table Data
We've finished making changes to the table. Let's store the table for later use. Calling `df_new.to_csv('Fish_measurements.csv')` will save the table to a CSV file.

**Saving a table to a CSV file**

The CSV file can be loaded into Pandas using the `pd.read_csv` method. Calling `pd.read_csv('Fish_measurements.csv', index_col=0)` will return a data frame containing all our table information. The optional `index_col` parameter will specify which column holds the row-index names

**Loading a table from a CSV file**
<!-- #endregion -->

```python id="mSzxHcJz70_U" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637305280958, "user_tz": -330, "elapsed": 1018, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="734dc5f8-a5ff-4516-d0ef-101de0959c62"
df = pd.read_csv('Fish_measurements.csv', index_col=0)
print(df)
print("\nRow index names when column is assigned:")
print(df.index.values)

df_no_assign = pd.read_csv('Fish_measurements.csv')
print("\nRow index names when no column is assigned:")
print(df_no_assign.index.values)
```

<!-- #region id="ahZCT-Yi70_U" -->
## Visualizing Tables Using Seaborn

Some numeric tables are too large to be viewed as printed output. Such tables are more easily displayed using heatmaps. A **heatmap** is graphical representation of a table, in which numeric cells are colored by value.  The easiest way to create a heatmap is to use the external Seaborn library. Lets now install Seaborn, and import it as `sns`.

**Importing the Seaborn library**
<!-- #endregion -->

```python id="Eq6KX5Rg70_V"
import seaborn as sns
```

<!-- #region id="9ZeEQZGA70_V" -->
Now, we’ll visualize our data frame as heatmap by calling sns.heatmap(df).

**Visualizing a heatmap using Seaborn**
<!-- #endregion -->

```python id="6TyvyBVT70_W" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637305288683, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="00e45bf1-ddf4-4962-b0cf-18f8da9fc457"
sns.heatmap(df)
plt.show()
```

<!-- #region id="3K-YuDMD70_W" -->
We can alter that color-pallet within the heatmap plot by passing in a `cmap` parameter. Below, we'll execute `sns.heatmap(df, cmap='YlGnBu')`. This will create a heatmap where the color-shades transition from yellow to green, and then to blue.

**Adjusting heatmap colors**
<!-- #endregion -->

```python id="yePs5ApZ70_W" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637305293675, "user_tz": -330, "elapsed": 834, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ffc8750-7293-4882-813a-298a0a7ff286"
sns.heatmap(df, cmap='YlGnBu')
plt.show()
```

<!-- #region id="TTTm45sp70_X" -->
Within the updated heatmap, the color-tones have flipped. Now, lighter colors correspond to higher measurements. We can confirm this by annotating the plot with the actual measurement values. 

**Annotating the heatmap**
<!-- #endregion -->

```python id="p3uOWDWe70_X" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637305295043, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e1b3371f-df14-4774-c45d-46cb055fe6fa"
sns.heatmap(df, cmap='YlGnBu', annot=True)
plt.show()
```

<!-- #region id="2L8sDJ3S70_Y" -->
The Seaborn library is built on top of Matplotlib. Consequently, we can use Matplotlib commands to modify elements of the heatmapt. For example, calling `plt.yticks(rotation=0)` will rotate the y-axis measurement labels. 

**Rotating heatmap labels using Matplotlib**
<!-- #endregion -->

```python id="gDj0sE5970_Y" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637305295762, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e25796be-16d8-4605-c1c6-face75fc9d4f"
sns.heatmap(df, cmap='YlGnBu', annot=True)
plt.yticks(rotation=0)
plt.show()
```

<!-- #region id="xlqHEebL70_Y" -->
Finally, we should note that running `sns.heatmap(df.values)` will also create a heatmap plot. However, the y-axis and x-axis labels will be missing from that plot. In order to specify the labels, we will need to set the  `xticklabels` and `yticklabels` parameters within  the `sns.heatmap` method.

**Visualizing a heatmap from a NumPy array**
<!-- #endregion -->

```python id="x0Y5bHj970_Z" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637305296403, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ace1018-7cd3-48f3-9e33-d9d59694a1e8"
sns.heatmap(df.values, 
            cmap='YlGnBu', annot=True,
            xticklabels=df.columns,
            yticklabels=df.index)
plt.yticks(rotation=0)
plt.show()
```
