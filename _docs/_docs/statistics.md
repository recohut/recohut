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

<!-- #region id="dBkS4d-vrNaR" -->
# Statistics and Linear Algebra for Data Science
<!-- #endregion -->

<!-- #region id="DzmJfLBorpUA" -->
## [1. **Mean, Median, and Mode**](https://learning.oreilly.com/scenarios/linear-algebra-for/9781098118006/)
<!-- #endregion -->

<!-- #region id="oB5bqljXrp01" -->
The mean, median, and mode are the most basic descriptive statistics functions in your tool belt. They provide a quick means to summarize data by identifying which values are most frequent.

Let's try them out on this small dataset of golden retriever weights. First, copy this dataset over as a Python list and import the defaultdict:
<!-- #endregion -->

```python id="aMAkAf9ZrtGj"
from collections import defaultdict

# Calculate mean, median, and mode for a sample of 9 golden retriever weights
dog_sample_weights = [ 65.8, 65.5, 64.6, 58.1, 63.6, 66.3,  61.3, 58.1, 64.0]
```

<!-- #region id="avWkZsEirx1A" -->
As we go through the next few pages, feel free to edit the data to see how it affects the mean, median, and mode. Using these measures, we can summarize the data and make basic predictions about what weights to expect for a golden retriever. There are only nine values here, but what if you had 100, 1,000, or 10,000 values?

These measures of central tendency have limitations in the information they describe, but they are a starting point for describing data without trawling thousands of records.
<!-- #endregion -->

<!-- #region id="GIoSQfKvsnxH" -->
**Mean**
<!-- #endregion -->

<!-- #region id="wpxKZpxRsAT1" -->
The mean is the average of a set of values. The operation is simple to perform: sum the values and divide by the number of values. The mean is useful because it shows where the "center of gravity" exists for an observed set of values.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eupTvuGrsDo1" executionInfo={"status": "ok", "timestamp": 1636595424412, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7f3bf11b-7d4c-42d2-8722-1f9f08328f87"
def mean(values):
    mean = sum(values) / len(values)
    return mean

print("MEAN: ", mean(dog_sample_weights))
```

<!-- #region id="L5smpTFfsFP5" -->
The mean is always calculated in the same way, regardless of if it is a sample or population. It plays a central role not just in describing data, but also in more advanced statistical tools, including linear regression.

Before moving on, feel free to edit the dog sample weights and observe how it affects the mean. Notice that an extreme outlier (a value that is much higher or lower than the rest of the values) will cause the mean to shift significantly in the direction of that outlier.
<!-- #endregion -->

<!-- #region id="tE0AwDp2spsv" -->
**Median**
<!-- #endregion -->

<!-- #region id="pvWCEq8QsrHn" -->
The median is the middle-most value in a set of ordered values. You sequentially order the values, and the median is the center-most value. If you have an even number of values, you average the two center-most values. The median is especially useful when a mean is skewed by outliers, or values that are extremely large or small compared to the rest of the values.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LKZ_Gwjtswxu" executionInfo={"status": "ok", "timestamp": 1636595644687, "user_tz": -330, "elapsed": 468, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dec60f91-6dec-40f2-a537-57594fe1ae46"
def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = int(n / 2) - 1 if n % 2 == 0 else int(n/2)

    if n % 2 == 0:
        return (ordered[mid] + ordered[mid+1]) / 2.0
    else:
        return ordered[mid]

print("MEDIAN: ", median(dog_sample_weights))
```

<!-- #region id="q0uUI16zs7Pn" -->
It might be beneficial to compare the mean and median with different sets of data. You will notice that if you skew the data with very large or very small values, the median is fairly unaffected, unlike the mean. When the mean and median have a wide difference between them, then that means the data is skewed.
<!-- #endregion -->

<!-- #region id="bUrGrr5ps_0o" -->
**Mode**
<!-- #endregion -->

<!-- #region id="zoralqxqtDgd" -->
The mode is the most frequently occurring value or set of values. It becomes useful primarily when your data is repetitive and you want to find which values occur the most frequently.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ef8xlMbktD5H" executionInfo={"status": "ok", "timestamp": 1636595795520, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3708c769-4a61-458a-ed45-d44ccf64315f"
def mode(values):
    counts = defaultdict(lambda: 0)

    for s in values:
        counts[s] += 1

    max_count = max(counts.values())
    modes = [v for v in set(values) if counts[v] == max_count]
    return modes

print("MODES: ", mode(dog_sample_weights))
```

<!-- #region id="44heJY84tgEO" -->
Modify the data so that it has several duplicates and notice that the value with the highest number of duplicates becomes the mode when you rerun the script. If any values are tied in the number of duplicates, then those values will be reported as modes together, and the dataset is said to be bimodal.
<!-- #endregion -->

<!-- #region id="JCndFtkFttgf" -->
**Using libraries**
<!-- #endregion -->

<!-- #region id="JfKHRvlItuxE" -->
If you want to explore these three measures using SciPy, NumPy, or the Python statistics package, here are other implementations of all three functions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="S5NZu2a8tvNx" executionInfo={"status": "ok", "timestamp": 1636595867558, "user_tz": -330, "elapsed": 694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5dc5bc55-a5df-4dbe-f6fb-fe1ecd895560"
# Using Python statistics
import statistics as stat

print("\n\nUsing Python Statistics")
print("MEAN: ", stat.mean(dog_sample_weights))
print("MEDIAN: ", stat.median(dog_sample_weights))
print("MODES: ", stat.mode(dog_sample_weights))
```

```python colab={"base_uri": "https://localhost:8080/"} id="o7-w_qpftxoA" executionInfo={"status": "ok", "timestamp": 1636595881197, "user_tz": -330, "elapsed": 687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3380aca8-06d1-43a9-fe0d-4ea5b219d0a0"
# Using NumPy and SciPy
import numpy as np
from scipy.stats import mode

print("\n\nUsing NumPy and SciPy")
print("MEAN: ", np.mean(dog_sample_weights))
print("MEDIAN: ", np.median(dog_sample_weights))
print("MODES: ", mode(dog_sample_weights))
```

<!-- #region id="8sPaJooKt06m" -->
Typically, you will use libraries to calculate these measures of central tendency for you, but it is a good exercise to learn how to implement them from scratch.
<!-- #endregion -->

<!-- #region id="MTlhPgFHt4Qg" -->
## [2. **Variance and Standard Deviation**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098111137/)
<!-- #endregion -->

<!-- #region id="hUk7KeesuRBf" -->
In describing data, we are often interested in measuring the differences between the mean and every data point. This gives us a sense of variance, or how "spread out" the data is.

First, let's declare a small dataset measuring the number of pets each person owns in a sample. By measuring how "spread out" this data is, we can determine how consistently we can encounter a certain number of pets. Let's also declare the mean for comparison later, as that will play a role anchoring our spread:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="o5B44P82uYlz" executionInfo={"status": "ok", "timestamp": 1636596059274, "user_tz": -330, "elapsed": 488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6595aba4-1864-49c6-e15a-46f1113a0acb"
from math import sqrt

# Number of pets each person owns
sample = [1, 3, 2, 5, 7, 0, 2, 3]

def mean(values):
    return sum(values) / len(values)

print("MEAN: ", mean(sample))
```

<!-- #region id="04IjWwm0uge0" -->
You will see that we get a mean of 2.875 pets per person. Let's explore the concept of variance next.
<!-- #endregion -->

<!-- #region id="_5fR5Nkoug5W" -->
**Variance**
<!-- #endregion -->

<!-- #region id="XNOBbwySukKQ" -->
Variance measures spread by calculating the difference between the mean and each element. Then, each of those differences are squared and then averaged (summed and divided by the number of elements). The larger the variance, the more spread out our data is.
<!-- #endregion -->

<!-- #region id="v8qboL9Juys5" -->
Treating the data as a sample, let's calculate the variance:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="T_DDIwXBuoya" executionInfo={"status": "ok", "timestamp": 1636596118701, "user_tz": -330, "elapsed": 517, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3539db94-f2da-49fe-e489-458b0e6c0a1b"
def variance(values, is_sample=False):
    mean = sum(values) / len(values)

    if is_sample:
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    else:
        return sum((v - mean) ** 2 for v in values) / len(values)

print("VARIANCE: ", variance(sample, is_sample=True))
```

<!-- #region id="V_E0mP4Ouu94" -->
Our variance comes out to be 4.9821 pets. The larger this variance value is, the more spread out the data will be, and we will be less likely to see values near the mean of 2.875.

Take care to note that variance has a slight tweak when calculated for samples, and you can see this tweak in how the is_sample parameter is used. For a sample with n number of elements, you will divide by n - 1 instead of n. This is to increase variance, because a sample has more uncertainty in being representative of the population. Feel free to change the is_sample parameter to False so the data is treated as a population. You will notice the variance falls to 4.3593, reflecting greater certainty in a smaller range than a sample.

So a lower variance means the data is more consistent and less varied, while a higher variance has more spread in the data. However, the variance value is not easy to interpret on its own without comparing it to another variance. What does a variance of 4.9821 pets mean? We need to square root it to create the standard deviation.
<!-- #endregion -->

<!-- #region id="Ul8r7SkFvE10" -->
**Standard Deviation**

It's reasonable to conclude that a higher variance means more spread, but how do we relate this back to our data? This number is larger than any of our observations because we did a lot squaring and summing, and so put it on an entirely different metric. How do we squeeze it back down so that it's on the scale we started with?

We undo a square with a square root, so let's take the square root of the variance. This gives us the standard deviation. This is the variance scaled back without the squaring, and we'll express it as "number of pets" which makes it a bit more meaningful:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TMWEHkVQvSao" executionInfo={"status": "ok", "timestamp": 1636596272780, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1b1fb2d1-ecb8-433a-ff08-62dc8e325436"
def std_dev(values, is_sample=False):
    return sqrt(variance(values, is_sample))

print("STD DEV: ", std_dev(sample, is_sample=True))
```

<!-- #region id="zWQ_s4GtvUnm" -->
Treating this data as a sample, you should get 2.232 pets as your standard deviation. We can express our spread on a scale we started with, and this makes our variance a bit easier to interpret. The standard deviation becomes even more useful and informative when plotted into a normal distribution, but this is covered in later sections.

For now, just note that variance and standard deviation are essentially the same metric expressed in two forms. Having a higher variance/standard deviation means you have data that is more spread out and less consistent around the mean. If you have a smaller variance/standard deviation, this means the data is tighter and more consistent around the mean.
<!-- #endregion -->

<!-- #region id="uvQIv1tnvig7" -->
## [3. **The Normal Distribution**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098111144/)
<!-- #endregion -->

<!-- #region id="QjHA8PgywNJx" -->
In statistics, we often encounter data that naturally fits a symmetrical, bell-shaped curve. This is known as a normal distribution, a type of continuous distribution that follows a bell shape. Also called a Gaussian distribution, it has several important features that it useful for several applications.

The normal distribution is important in so many areas of probability and statistics primarily because it has predictive value and occurs frequently in nature and science. It also plays a critical role in hypothesis testing, simulations, optimizations, and other data science models.

Let's say we are doing a veterinary study and are trying to understand what weight to expect for any given golden retriever. The following is a normal distribution for golden retriever weights. By studying this, we can get an idea how much an underweight versus overweight golden retriever would weigh. Notice how most of the mass is around the mean of 64.43 pounds. We would probably consider a golden retriever close to this weight as likely being typical and normal.
<!-- #endregion -->

<!-- #region id="bvPfoo8Bwj6V" -->
The normal distribution has several important properties that make it useful:
1. It's symmetrical; both sides are identically mirrored around the center.
2. Most of the mass is at the center around the mean.
3. It has a spread (being narrow or wide) that is specified by standard deviation.
4. The “tails” are the least likely outcomes and infinitely approach zero but never actually touch it.

The normal distribution resembles a lot of phenomena in nature and daily life and even generalizes non-normal problems because of the central limit theorem, which states that the means of samples form a normal distribution even if the population is not normal.
<!-- #endregion -->

<!-- #region id="PfQvv14uwqgV" -->
Let's generate and plot random golden retriever weights coming from a normal distribution with a mean of 64.43 and standard deviation of 2.99.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TuRd-4lJxB75" executionInfo={"status": "ok", "timestamp": 1636596759743, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd81c810-f6ea-42a7-b010-8d98c93c0b8c"
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

mean = 64.43
standard_deviation = 2.99

# generate 30 random golden retriever weights
# around a normal distribution with a mean 
# of 64.43 and standard deviation of 2.99
random_weights = np.random.normal(mean, standard_deviation, 30)
random_weights
```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} id="uH-0icCcxJGv" executionInfo={"status": "ok", "timestamp": 1636596772342, "user_tz": -330, "elapsed": 1017, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c3ab1c2-0914-4aad-f5e1-423df76aa179"
# Plot between -50 and 80 with .01 steps.
x_axis = np.arange(50.0, 80.0, 0.01)
plt.plot(x_axis, norm.pdf(x_axis, mean, standard_deviation))
plt.scatter(random_weights, [0 for _ in range(0,30)])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="LLeyu_ZaxOFm" executionInfo={"status": "ok", "timestamp": 1636596782790, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5164741a-442c-49d2-8801-52b76f7bc0a9"
# Plot a histogram to visualize distribution
bin_size=20
plt.hist(random_weights, bin_size)
plt.show()
```

<!-- #region id="Ikf2F_vVw6q9" -->
We could do a number of things with this bell curve. We could create a simulation to generate hundreds of golden retrievers with realistic weights. If we were veterinarians, we could also use this bell curve to show patients and say, "Your golden retriever is on the far left side of the bell curve, so we need to help your dog gain some weight."

To really be productive with the normal distribution, you will have to learn about its cumulative distribution function to calculate areas/probabilities under it.
<!-- #endregion -->

<!-- #region id="3g8mIBzlw_mI" -->
## [4. **Central Limit Theorem**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098111168/)

The central limit theorem is a phenomenon where the normal distribution shows up when we take the means of samples, even if the underlying population does not follow a normal distribution.

This opens up a lot of useful concepts and gives license to many statistical tools. Even if the population you are studying does not follow a normal distribution, you can still infer parameters (like the population mean) by measuring its samples. This is a powerful idea and lays an important foundation for inferential statistics, including hypothesis testing.
<!-- #endregion -->

<!-- #region id="Y83FMxCbzB0e" -->
One of the reasons the normal distribution is useful is because it appears a lot in nature, such as in adult golden retriever weights. It shows up in a much more fascinating context, however, outside natural populations. When we start measuring samples from a population, even if that population does not follow a normal distribution, the normal distribution still makes an appearance!

More specifically, the central limit theorem demonstrates that when means of samples are plotted, they will form a normal distribution. There is a bit more nuance to this that we will explore soon, but first let's do the following imports:
<!-- #endregion -->

```python id="4o9IXoCOzMrn"
import matplotlib.pyplot as plt
import pandas as pd
import random
from statistics import mean
```

<!-- #region id="JlLG43YzzOji" -->
In the following script, you will read a single column of data (accessed at https://bit.ly/369dqlF). This data has no particular pattern and does not resemble a normal distribution at all.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="gtEQ0uDIzVQf" executionInfo={"status": "ok", "timestamp": 1636597342842, "user_tz": -330, "elapsed": 534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53cb0c5c-6067-45a8-8837-525e958741dc"
points = pd.read_csv("https://bit.ly/369dqlF", header=None)

# Bring in data that does not fit normal distribution
x_values = [p for p in points[0]]

# Plot the data in histogram with bin range size of 20
plt.hist(x_values, 20)
plt.show()
```

<!-- #region id="dBDRXovRzZ1s" -->
Now, what happens when we group up samples and take the mean of each one?
<!-- #endregion -->

<!-- #region id="1sA3JEhazf_z" -->
We have purely random data. However something remarkable happens when you take random samples from this data, take the mean of each sample, and then plot them. It resembles a normal distribution! Let's append the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="vhNYvqYLzkqn" executionInfo={"status": "ok", "timestamp": 1636597413597, "user_tz": -330, "elapsed": 756, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="975dc311-259c-4b38-8984-5d4854cc30ca"
# Randomly sample 1000 groups of 31 points and take their average
sample_size = 31

sample_avgs = [mean(random.choices(x_values, k=sample_size)) for _ in range(1000)]
plt.hist(sample_avgs, 20)

# Plot the sample means in histogram with bin range size of 20
plt.show()
```

<!-- #region id="Iphzvx19zrEE" -->
This shows a phenomenon when taking samples from a population. Even if the population has no resemblance to the normal distribution, the means of its samples do. Well, there are a few conditions that need to be met for this to happen. Let's talk about those next.
<!-- #endregion -->

<!-- #region id="-M9CSVnGz0ib" -->
You will see this normal distribution behavior if the sample size is at least 31, but as you make that sample size smaller, you will notice that the normal distribution diminishes. This is why we often resort to T-distributions rather than normal distributions for sample sizes less than 31 (which is beyond the scope of this scenario).

But this behavior is important because when we are doing studies about populations or testing a new drug, we do not have to fret about whether the population follows a normal distribution. If we are taking samples (which is almost always the case), whatever we are inferring is going to follow a normal distribution due to the central limit theorem.

Here are the important points about the central limit theorem. Note these carefully, because they are essential to many ideas in statistics and hypothesis testing:
1. The mean of the sample means is equal to the population mean.
2. If the population is normal, then the sample means will be normal.
3. If the population is not normal but the sample size is greater than 30, the sample means will still form a normal distribution!
4. The standard deviation of the sample means equals the population standard deviation divided by the square root of n.
<!-- #endregion -->

<!-- #region id="yoGU1esL0HKC" -->
## [5. **Confidence Intervals**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098111175/)

A *confidence interval* is the probability that a population parameter, such as a mean, falls within a certain range based on a sample. Confidence intervals play a huge role in scientific research, engineering, and many other fields of study inside and outside of data science.
<!-- #endregion -->

<!-- #region id="Kx3mjbR10f77" -->
You may have heard the term confidence interval, which often confuses statistics newcomers and students. A confidence interval is a range calculation showing how confidently we believe a population mean (or other parameter) falls in a range based on a sample.

For example, let's say I have a sample of golden retriever weights, and I want to know with 95% confidence what range of weight values I can expect the population weight mean to fall in. That's essentially the idea of a confidence interval, and it is pretty useful to make claims about a population range based on a sample... but with a percent of confidence. If I have a confidence of 95%, there is, conversely, a 5% chance the population mean will not be in my sample's confidence interval at all.

Here's a more concrete example: based on a sample of 31 golden retrievers with a sample mean of 64.408 and a sample standard deviation of 2.05, I am 95% confident that the population mean lies between 63.686 and 65.1296.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5XEByJm606hC" executionInfo={"status": "ok", "timestamp": 1636597741283, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c1b17e13-06c6-433d-9d30-07631b9f2dcf"
from math import sqrt
from scipy.stats import norm


def critical_z_value(p, mean=0.0, std=1.0):
    norm_dist = norm(loc=mean, scale=std)
    left_area = (1.0 - p) / 2.0
    right_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_area), norm_dist.ppf(right_area)

print("Critical range for .95 confidence: ")
print(critical_z_value(.95))
```

<!-- #region id="utaXA0C407FI" -->
Note the critical_z_value() function. To be 95% confident that my sample mean lies in a range for the population mean, I need to find the area in the middle of the normal distribution that is .95. This will give me .025 area left in each tail. Using the range that gives me this area in the center, I will have the standard deviation value that, plus or minus (±), will provide me an area of .95. This is my critical z-value, and the critical_z_value() function returns the left and right of these boundaries.
<!-- #endregion -->

<!-- #region id="nnH3xvXe1ASV" -->
The critical_z_value() function did most of the hard work. Now all I have to do is convert it back from a standard normal distribution scale (mean of 0 and standard deviation of 1) back into my golden retriever distribution scale (mean of 64.408 and standard deviation of 2.05). Using the central limit theorem, I can achieve this by multiplying the lower and upper boundaries of the critical value range by the sample standard deviation divided by the square root of the sample size n.

Now I can predict with 95% confidence what range the golden retriever population weight mean will be based on my sample.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4lmGmXpC1NyE" executionInfo={"status": "ok", "timestamp": 1636597840720, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9e8f81c6-4400-4d0c-892f-c059fd438b23"
def confidence_interval(p, sample_mean, sample_std, n):
    # Sample size must be greater than 30

    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std / sqrt(n))
    upper_ci = upper * (sample_std / sqrt(n))

    return sample_mean + lower_ci, sample_mean + upper_ci


# What is the confidence interval of my sample of 31 golden retrievers
# with sample mean of 65.13, sample std of 2.05?
print(confidence_interval(p=.95, sample_mean=64.408, sample_std=2.05, n=31))
```

<!-- #region id="bOKhYDx91TSe" -->
Running this, I find that the population mean lies between 63.686 and 65.1296 based on this sample with 95% confidence.


<!-- #endregion -->

<!-- #region id="vtyD9cL61Wou" -->
## [6. **P-values One-Tailed Test**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098111182/)

Hypothesis testing plays a critical role in evaluating whether a finding is significant. We always have to entertain the possibility that our finding is coincidental, and the one-tailed and two-tailed tests help achieve this analysis.
<!-- #endregion -->

<!-- #region id="xfqbgvqe1oP5" -->
When we say something is statistically significant, what do we mean by that?

It has something to do with a concept called the p-value, the probability something occurring by chance rather than because of a hypothesized explanation. This helps us frame our null hypothesis (H0), saying that the variable in question had no impact on the experiment and any positive result is just random luck. The alternative hypothesis (H1) poses that a variable in question (called the controlled variable) is the cause of a positive result.

Let's use one form of hypothesis testing, called the one-tailed test, to see if a new cold drug works. A one-tailed test structures null and alternative hypotheses in a "less than" or "greater than" fashion.

Let's code a hands-on example. Past studies have shown that the mean recovery time for a cold is 18 days, with a standard deviation of 1.5 days, and that it follows a normal distribution. Let's declare that with this code snippet and bring in an import we will use later:
<!-- #endregion -->

```python id="AJLtkmzI2aqv"
from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5
```

<!-- #region id="FabPa8qb2cVs" -->
Now let's say an experimental new drug was given to a test group of 40 people, and it took an average of 16 days for them to recover from the cold. Did the drug work? We need to recognize that this drug falls well outside that .05 tail, and traditionally the threshold for statistical significance is a p-value of 5% or less, or .05. In effect, for the drug to be considered working, the recovery time must be inside the left tail region.
<!-- #endregion -->

<!-- #region id="ErqltVMk2kqG" -->
In a one-tailed test, we typically frame our null (H0) and alternative (H1) hypotheses using inequalities. We structure the alternative hypothesis that the new mean is less than 18, and the null hypothesis has it greater than 18. The grey shaded area below is our p-value from this experiment, but the red area of .05 is the required p-value we need to be within to say the drug works.
<!-- #endregion -->

<!-- #region id="9CFiXrSL3Fc5" -->
Let's calculate that p-value now.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pI7C1sNd3ICG" executionInfo={"status": "ok", "timestamp": 1636598344106, "user_tz": -330, "elapsed": 570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e1ead285-e8f1-4f2a-faf6-035a585b1ff0"
# Probability of 16 or less days
p_value = norm.cdf(16, mean, std_dev)

print("1-tailed P-value: ", p_value) 
print("Required P-value: .05")

if p_value <= norm.ppf(.05):
    print("Passes 1-tailed test")
else:
    print("Fails 1-tailed test")
```

<!-- #region id="ftQSAM103OOW" -->
Our p-value of .0912 is much greater than .05. So our drug has failed to be statistically significant, and we cannot say it worked.
<!-- #endregion -->

<!-- #region id="50lH26r23VDp" -->
## [7. **P-values Two-Tailed Test**](https://learning.oreilly.com/scenarios/statistics-for-data/9781098117030/)

Let's explore the two-tailed test in this scenario, which is often preferable in most experiments.
<!-- #endregion -->

<!-- #region id="OF4pgaI_3mV-" -->
Let's use a form of hypothesis testing called the two-tailed test to see if a new cold drug works. A two-tailed test structures null and alternative hypotheses in an "equals" and "not equals" fashion.

Past studies have shown that the mean recovery time for a cold is 18 days, with a standard deviation of 1.5 days, and that it follows a normal distribution. Consider a critical Z-value for .95, where 95% of probability is between 15 and 21 days of recovery time.

An experimental new drug was given to a test group of 40 people and it took an average of 16 days for them to recover from the cold.

Did the drug work at 95% confidence? We need to recognize that this drug falls well inside this 95% critical Z range (the red area), and traditionally the threshold for statistical significance is a p-value of 5% or less, or .05 (tails outside the red area). In effect, for the drug to be considered working, the recovery time must be outside that shaded region where coincidence is a less likely explanation.
<!-- #endregion -->

<!-- #region id="wAqw1WWR3_gO" -->
To do a two-tailed test, we frame our null and alternative hypothesis in an "equal" and "not equal" structure, as opposed to an inequality in a one-tailed test. In our drug test, we will say the null hypothesis H0 has a mean recovery time of 18 days. But our alternative hypothesis H1 is the mean recovery time is not 18 days thanks to the new drug.

This has an important implication. We are structuring our alternative hypothesis to not test whether the drug improves cold recovery time, but if it had any impact in increasing or decreasing the duration of the cold. We spread our p-value to both tails, not just one.

If we are testing for a statistical significance of 5%, then we split it and give each remaining 2.5% half to each tail. Here is how we calculate the area for the left tail.
<!-- #endregion -->

```python id="h_mjrUma4Rzq"
from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# Probability of 16 or less days
p1 = norm.cdf(16, mean, std_dev)
```

<!-- #region id="lSPzNRUo4UHN" -->
Thanks to the symmetry of the normal distribution, we can simply calculate the boundary of the other tail as being equal. In a two-tailed test, the p-value is going to be the area of both tails.
<!-- #endregion -->

```python id="qjpmEvwX4Y8w"
# Probability of 20 or more days
# Take advantage of symmetry
p2 = p1

# P-value of both tails
# I could have also just multiplied by 2 
p_value = p1 + p2
```

<!-- #region id="Pjc01JNP4asH" -->
If our drug mean recovery time falls in either region below (in red) our test is successful and we reject the null hypothesis:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AoGVEFG74eSt" executionInfo={"status": "ok", "timestamp": 1636598678469, "user_tz": -330, "elapsed": 452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="013811ca-e532-4772-8277-5abf5c5fb188"
print("2-tailed P-value", p_value)
print("Required P-value: .05")


if p_value <= .05:
    print("Passes 2-tailed test")
else:
    print("Fails 2-tailed test")
```

<!-- #region id="jKhcKRPG4f8c" -->
There is an 18.24% probability results were due to random luck, rather than because drug made an impact. This demonstrates that the two-tailed test is often preferable to the one-tailed test because it has a much higher threshold. Again, our p-value is .1824 so that definitely exceeds .05. Therefore we cannot conclude this drug works.
<!-- #endregion -->

<!-- #region id="Sg3W-pk_4ouH" -->
## [8. **Matrix Multiplication**](https://learning.oreilly.com/scenarios/linear-algebra-for/9781098118068/)

When we have multiple linear transformations performed on a vector space, we can effectively combine them into a single linear transformation. This is the idea behind matrix multiplication.

We use the dot product to perform this operation in libraries like NumPy, and along the way.
<!-- #endregion -->

<!-- #region id="ReZeFSbs8ofS" -->
What does it mean when we apply a transformation to another transformation? This is essentially what matrix multiplication is, which is combining multiple linear transformations into a single linear transformation.

When we talk about matrix multiplication, we continue to extend our ideas about linear transformation, giving us another tool for manipulating data. This enables applications like solving systems of equations and machine learning.
<!-- #endregion -->

<!-- #region id="-rSZQMRx98-4" -->
While we could apply these linear transformations one at a time, we could combine them into a single transformation (note that the basis matrix has no effect as a transformation, as it's the linear algebra equivalent of multiplying by "1"). Just like there is a formula for matrix vector multiplication, there is also a formula for matrix multiplication.
<!-- #endregion -->

<!-- #region id="N91LJCcj99pM" -->
Here is how we combine two different transformations/matrices (rotation and shear) into one combined transformation/matrix in NumPy:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tDBNe_yZ-GF9" executionInfo={"status": "ok", "timestamp": 1636600166306, "user_tz": -330, "elapsed": 593, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d1ddcfcb-9cf2-4626-b117-64ff14b344d6"
from numpy import array

# Transformation 1 - Rotate
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

# Transformation 2 - Shear
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# Combine Transformations
combined = transform2 @ transform1

print(combined)
```

<!-- #region id="ZLEVQ2Jy-LJX" -->
Note above that I call the @ to multiply transform2 and tranform1 to effectively combine them. This means that I am applying transformation 2 (the shear) on transformation 1 (the rotation). Now consider this before moving on: does the order you apply the transformation matter?

Yes, The order you perform the dot product matters! Here is the Python code that performs the rotation followed by the shear:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CkQpNpqk-OAQ" executionInfo={"status": "ok", "timestamp": 1636600299344, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad6f2b93-b21c-41f2-a7c6-8fa2d01cf19f"
from numpy import array

# Transformation 1 - Rotate
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

# Transformation 2 - Shear
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# Combine Transformations
combined = transform1 @ transform2

print(combined)
```

<!-- #region id="O_GfI1A4-bDB" -->
If you have three or more dimensions, these rules still apply. You multiply and sum each respective row with each column between the two matrices, and the order you execute them will affect the result:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yxEVEmEy-wp3" executionInfo={"status": "ok", "timestamp": 1636600339765, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4994377f-b995-421f-d03d-65cc0c9dbeb9"
from numpy import array

# Transformation A
i_hat1 = array([1, 0, -1])
j_hat1 = array([1, 1, 3])
k_hat1 = array([-1, 2, 3])
transformA = array([i_hat1, j_hat1, k_hat1]).transpose()

# Transformation B
i_hat2 = array([3, 2, 2])
j_hat2 = array([-4, 2, 9])
k_hat2 = array([2, 1, 4])
transformB = array([i_hat2, j_hat2, k_hat2]).transpose()

# Combine Transformations - A then B
a_then_b = transformB @ transformA 
print("A then B:")
print(a_then_b)

# Combine Transformations - B then A
b_then_a = transformA @ transformB
print("B then A:")
print(b_then_a)
```

<!-- #region id="3eRdedaG-1bs" -->
Why do we care about these kinds of properties with matrix multiplication? Again, if data is a series of vectors packaged into a matrix, we can iteratively transform that data, but all those transformations could be consolidated. While you may not think this abstractly in practice, when you import data from Pandas and Excel, it is good to be mindful of these behaviors as you dive deeper into machine learning and optimization algorithms.
<!-- #endregion -->

<!-- #region id="ROELWYDT_4Nk" -->
## [9. **Determinants**](https://learning.oreilly.com/scenarios/linear-algebra-for/9781098118075/)

When we perform linear transformations, we sometimes “expand” or “squish” space, and the degree to which this happens can be helpful to know. This is what the determinant does, and is one of the most fundamental and useful tools in linear algebra. There may be times before where you're siting down to solve a problem like a system of equations and calculate the determinant to do various checks, such as for linear dependence.
<!-- #endregion -->

<!-- #region id="c68yO9r8_5X-" -->
Take the following linear transformation where we stretch i-hat by a scalar of 3 and j-hat by 2. Notice I sampled an area in yellow. What happened to that yellow area?
<!-- #endregion -->

<!-- #region id="gW0pVF3bAKSS" -->
<!-- #endregion -->

<!-- #region id="kfp5brIKANGK" -->
If you study that yellow area and describe mathematically what happened in the transformation, you can visually see it increases by 6x. In other words, the area increases by a factor of 6. The factor reflecting how much the area increased/decreased in a transformation is known as the determinant.

The easiest way to calculate the determinant on a given matrix/transformation is to use NumPy's det() function from its linear algebra package. Here is the Python code to calculate the determinant for the preceding example:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="17d1-tO-AUQO" executionInfo={"status": "ok", "timestamp": 1636600751523, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d0fd593a-ccbf-40b3-dee4-56d8b965c9fa"
from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant)
```

<!-- #region id="7urXj-SvAZ_P" -->
So why do we care about the determinant? Determinants describe how much a sampled area in a vector space changes in scale with linear transformations, and this can provide helpful information about the transformation. Especially in cases where the determinant becomes negative or 0, the determinant can provide useful insights, especially for problems in linear programming and optimization.

Using some basic geometry, you can infer that simple shears and rotations do not change the determinant, which will remain at 1.0 and reflect no change in area. The following is a visualization and code of a simple shear that should reflect no change in the determinant:
<!-- #endregion -->

<!-- #region id="Kgm9k8e1An6m" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eQmYF4H8Amlt" executionInfo={"status": "ok", "timestamp": 1636600823182, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c9909ff-6e6b-43e9-ed0b-e1c8762e53f2"
from numpy.linalg import det
from numpy import array

i_hat = array([1, 0])
j_hat = array([1, 1])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant)
```

<!-- #region id="JsU5FwAMArgJ" -->
However, scaling will increase or decrease the determinant, as that will increase/decrease the sampled area.

Determinants also apply to cases in more than two dimensions. We just think of it in terms of volume rather than an area (even for high-dimensional vector spaces). The following is a visualization of the determinant for a 3D vector space transformation:
<!-- #endregion -->

<!-- #region id="0gJE1DiWA0I0" -->
<!-- #endregion -->

<!-- #region id="TdH560-4A2Ry" -->
**Zero Determinants**
<!-- #endregion -->

<!-- #region id="5b6s_nk5BGtd" -->
What does a determinant of 0 mean? This is actually an important check when dealing with systems of equations and other types of problems, because it indicates linear dependence.

Linear dependence means that the transformation puts two or more basis vectors on the same underlying line. In many situations, such as solving systems of equations, this creates complications, because we have compressed our space into fewer dimensions.

Take a look at the following linear transformation. Notice that i-hat and j-hat are now collinear, meaning they share the same straight line, making this transformation linearly dependent:
<!-- #endregion -->

<!-- #region id="FWymaBjhBSOp" -->
<!-- #endregion -->

<!-- #region id="vNocqxdzBS8R" -->
Here is the Python code with NumPy showing this zero determinant:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QpfU6mlTBU2_" executionInfo={"status": "ok", "timestamp": 1636601001773, "user_tz": -330, "elapsed": 524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eedfd130-87e6-4c60-ac6c-bdaff38bb392"
from numpy.linalg import det
from numpy import array

i_hat = array([-1, 1])
j_hat = array([1, -1])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant)
```

<!-- #region id="TKqpj8_3BXFc" -->
Another way to think of a zero determinant is that it squishes all of space into a lesser number of dimensions. This makes the transformation problematic for many reasons, as you are now limited to vectors in fewer dimensions.
<!-- #endregion -->

<!-- #region id="r5bER-71BbFL" -->
**Negative Determinants**
<!-- #endregion -->

<!-- #region id="sCGimVMIBeRA" -->
Let's explore negative vectors and what they mean. Here is a visualization of a linear transformation that has a negative determinant. Any guesses on why that is?
<!-- #endregion -->

<!-- #region id="NwR2abTyB2js" -->
<!-- #endregion -->

<!-- #region id="aH-bRwDRB3Rg" -->
Notice how i-hat and j-hat have flipped places in their clockwise positions. When the entire space "flips" and reverses its orientation, that will result in a negative determinant. Here is this negative determinant transformation in Python:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Rtx6-gRgB62e" executionInfo={"status": "ok", "timestamp": 1636601157442, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6a4992e-a594-41a5-a0f5-4a56868718fb"
from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([1, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant)
```

<!-- #region id="Y0B_T9FjB9HO" -->
So a determinant of -5.0 means not only that the area increased by a factor of 5, but it also flipped the space and resulted in a negative determinant.

If you need to visually see whether a 3D vector space would result in a negative determinant, use the right-hand rule as shown in the following. If you cannot relatively orient i-hat, j-hat, and k-hat like my hand here, the orientation has flipped, and you should have a negative determinant. Of course, NumPy can calculate and indicate this for you too by returning a negative determinant:
<!-- #endregion -->

<!-- #region id="lPWyD7OCCJKp" -->
<!-- #endregion -->

<!-- #region id="XRuGYsB4CHLq" -->
In this example, you can see that with some rotation my labelled hand orients with the vector space above, so we can expect a positive determinant.
<!-- #endregion -->

<!-- #region id="QMHAEFhHCOtO" -->
## [10. **Inverse Matrices**](https://learning.oreilly.com/scenarios/linear-algebra-for/9781098118082/)

Suppose you were asked to undo a linear transformation. How do you do it? The way you undo a linear transformation is with another linear transformation, one that does the opposite movement. This is actually what the inverse matrix does, and it has many use cases in solving systems of equations, linear programming, linear regression, and machine learning.
<!-- #endregion -->

<!-- #region id="XNr1yicMFlOs" -->
The most practical way to calculate an inverse matrix is to have NumPy to do it for you. Below, we use the inv() function from NumPy to calculate the inverse of matrix A:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Kdunny8cFlj7" executionInfo={"status": "ok", "timestamp": 1636602119346, "user_tz": -330, "elapsed": 1393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="501b94f9-5ac4-4ae2-b5c6-a4d352a886f6"
from numpy.linalg import inv
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

A = array([i_hat, j_hat]).transpose()

# calculate inverse matrix
inverse = inv(A)
print(inverse)
```

<!-- #region id="pxWDj8ehFnmy" -->
Calculating inverse matrices are a bit tedious to do by hand, and even computers can struggle to do it on large matrices. This is why techniques like matrix decomposition are used.
<!-- #endregion -->

<!-- #region id="wNwiyzV4F0jc" -->
Let's now take this inverse matrix and apply it to the stretched matrix. When we apply it with the dot product, what happens?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="16hp4XhUF8VU" executionInfo={"status": "ok", "timestamp": 1636602203805, "user_tz": -330, "elapsed": 509, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fab4465d-0767-435e-fb29-20fd1e806f47"
from numpy.linalg import inv
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

A = array([i_hat, j_hat]).transpose()

# calculate inverse matrix
inverse = inv(A)

# apply inverse of A to matrix A
identity = inverse @ A 
print(identity)
```

<!-- #region id="L-PyTdGpF8nu" -->
What happens is that we get our basis vectors back to their starting position, effectively undoing the transformation. This matrix where we have a nice diagonal of 1s from corner to corner (and surrounded by 0s) is known as an identity matrix. It effectively shows that no transformation has taken place, and is a key objective in many problems, such as systems of equations.

Now let's apply this knowledge to something slightly more useful. Lets start with a simple vector. Vector v was transformed by matrix A and now lands at [1 2]. What was it before the transformation?
<!-- #endregion -->

<!-- #region id="DQCbREgbGKDC" -->
We need to find another transformation that reverses this one, and see where v started, as visualized in the following:
<!-- #endregion -->

<!-- #region id="Oo8wDDphGQio" -->
<!-- #endregion -->

<!-- #region id="zQ9a0xxLHAmg" -->
Again, we will use the inv() function in NumPy to find the inverse transformation, but this time we will apply it to vector v rather than matrix A:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8YRwscHfGgLa" executionInfo={"status": "ok", "timestamp": 1636602488391, "user_tz": -330, "elapsed": 463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ef4fedb-4d9d-4cbb-feb4-c9963e1098dd"
from numpy.linalg import inv
from numpy import array

new_v = array([1, 2])
i_hat = array([3, 0])
j_hat = array([0, 2])

A = array([i_hat, j_hat]).transpose()

# calculate inverse matrix
inverse = inv(A)

# calculate v before A transformation
old_v = inverse @ new_v
print(old_v)
```

<!-- #region id="K_yqpq0gHCGD" -->
So our vector started at approximately [.333 1] (or [1/3 1], if you make it rational) before it was transformed.

Interesting, right? Inverse matrices are used in many applications from linear programming to linear regression and machine learning. In other scenarios, we will cover how to use them to solve system of equations, as well as linear regressions.

Keep in mind that calculating inverse matrices can be an increasingly computationally expensive operation as the matrix grows larger. This is one reason why matrix decomposition, which breaks a matrix up into simpler components, is used quite a bit in computer science. These simplified matrices derived off the larger matrix can then be used to efficiently do tasks like linear regression and principal component analysis.
<!-- #endregion -->

<!-- #region id="Y8MQHjwCHR_8" -->
## [11. **Matrix Decomposition**](https://learning.oreilly.com/scenarios/linear-algebra-for/9781098118105/)

As you go down the linear algebra rabbit hole, there will be situations in which you inevitably encounter the need for matrix decomposition. This is a process in which you break up a matrix into simpler matrices that are easier to interpret and/or use, depending on your task.

In this scenario, we will explore two matrix decomposition methods, QR decomposition and eigendecomposition, the latter of which creates the famous eigenvectors and eigenvalues.
<!-- #endregion -->

<!-- #region id="dYbYuOjhHyGa" -->
Matrix decomposition, also called matrix factorization, is a process in which you break up a matrix into simpler components for a specific task. The need for this is primarily in circumstances where you run into the limitations that computers have. Computers, even as powerful as they are today, can choke on large complicated matrices full of high-dimensional data. With only so much memory and computational power, even tasks like inverse matrices, solving systems of equations, or calculating determinants can be taxing.

Much like how you cut food into small pieces for a small child, you can cut a matrix into smaller components so the computer can ingest it more easily.

Another way to think about matrix decomposition is by comparing it to number factoring, such as how the number 10 factors into 2 and 5. We are decomposing the matrix into something more basic, although what that looks like depends on the task.

Take, for example, QR Decomposition, which breaks up a matrix A (which can be square or non-square) into two matrices Q and R. When you dot-product Q and R together, it will rebuild matrix A.

QR decomposition is primarily used for solving systems of equations, least squares problems, as well as linear regression. These problems do not always require decomposition, but they do once they reach a certain scale.

In the following, we use QR decomposition to break up a matrix A, and then rebuild it again using the @ operator:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LD3t4dGeIJdM" executionInfo={"status": "ok", "timestamp": 1636602787716, "user_tz": -330, "elapsed": 1547, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31a5c760-520d-44d6-ce17-e8de7d4c7e33"
from numpy import array
from numpy.linalg import qr

# define matrix
A = array([
[5, 2, 7],
[1, 7, 9]
])

# decompose
Q, R = qr(A)

print("Q:")
print(Q)
print("R:")
print(R)

# rebuild
B = Q @ R

print("Reconstructed:")
print(B)
```

<!-- #region id="uA5InIZhIKX-" -->
Eigendecomposition is another form of matrix decomposition that is often used for principal component analysis (PCA) in machine learning.

You might have heard the terms eigenvector and eigenvalues, which are both the components of eigendecomposition.

Without going deep into a rabbit hole, eigenvectors always have a length of 1.0, and eigenvalues are applied to the eigenvectors to scale their length. These two pieces of information provide some useful (if abstract) properties of the matrix.

There is one eigenvector and eigenvalue for each dimension of the parent matrix, and not all matrices can be decomposed into an eigenvector and eigenvalue. Sometimes complex (imaginary) numbers will result!

Here is eigendecomposition performed using NumPy, where matrix A is decomposed into the eigenvector and eigenvalue:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lWy5S096IkXT" executionInfo={"status": "ok", "timestamp": 1636602936223, "user_tz": -330, "elapsed": 1267, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16f87f02-4011-4147-8909-325dc8e14f27"
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([
    [1, 2],
    [4, 5]
])

eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)
```

<!-- #region id="Ncg_3u3-IvFa" -->
To rebuild a matrix from its eigenvectors and eigenvalues, we tweak our formula using some inverse matrix work:

$ A = Q \wedge Q^{-1} $

where,
- $ Q = \text{eigenvectors} $,
- $ \wedge = \text{eigenvalues in diagonal form} $,
- $ Q^{-1} = \text{inverse of matrix Q} $.
<!-- #endregion -->

<!-- #region id="Re0Dxsv5J_hP" -->
Here is how we apply the formula to rebuild the matrix. Note how we repackage our eigenvalues into a diagonal matrix, meaning we orient it diagonally and then pad it with 0s:
<!-- #endregion -->

<!-- #region id="9kp0PpJtKR-y" -->
$
Q = \begin{bmatrix}
0.806 & 0.343 \\
0.590 & -0.939
\end{bmatrix}
$

$
\wedge = \begin{bmatrix}
-0.464 & 0 \\
0 & 6.464
\end{bmatrix}
$

$
Q^{-1} = \begin{bmatrix}
-0.977 & 0.357 \\
-0.614 & -0.839
\end{bmatrix}
$

$
A = Q \wedge Q^{-1} = \begin{bmatrix}
1 & 2 \\
4 & 5
\end{bmatrix}
$
<!-- #endregion -->

<!-- #region id="didhBJnLKF4H" -->
And here is the rebuilding operation appended to our code using NumPy's @ multiplication operator. If you take a look at the following code, you will see it rebuilds the matrix from the eigenvectors and eigenvalues:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bkeqg-TILS5x" executionInfo={"status": "ok", "timestamp": 1636603620170, "user_tz": -330, "elapsed": 1254, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d5dfdb22-754a-49a7-e4f9-020931cc90af"
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([
    [1, 2],
    [4, 5]
])

eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)

print("\nREBUILD MATRIX")
Q = eigenvecs
R = inv(Q)

L = diag(eigenvals)
B = Q @ L @ R

print(B)
```

<!-- #region id="_mVrBg0lLVZT" -->
There are many other matrix decomposition techniques that we did not cover, including LU decomposition, Cholesky decomposition, and the famous singular value decomposition (SVD). The last one, SVD, is used heavily for compressing and denoising data.

The best way to learn each of these is to apply them to the problems they were designed to solve.
<!-- #endregion -->
