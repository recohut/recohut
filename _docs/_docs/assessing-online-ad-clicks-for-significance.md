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

<!-- #region id="aqU4nVYxtdpf" -->
# Assessing Online Ad Clicks for Significance

Fred is a loyal friend, and he needs your help. Fred just launched a burger bistro in the city of Brisbane. The bistro is open for business, but business is slow. Fred wants to entice new customers to come and try his tasty burgers. To do this, Fred will run an online advertising campaign directed at Brisbane residents. Every weekday, between 11:00 a.m. and 1:00 p.m., Fred will purchase 3,000 ads aimed at hungry locals. Every ad will be viewed by a single Brisbane resident. The text of every ad will read, “Hungry? Try the Best Burger in Brisbane. Come to Fred’s.” Clicking the text will take potential customers to Fred’s site. Each displayed ad will cost our friend one cent, but Fred believes the investment will be worth it.

Fred is getting ready to execute his ad campaign. However, he runs into a problem. Fred previews his ad, and its text is blue. Fred believes that blue is a boring color. He feels that other colors could yield more clicks. Fortunately, Fred’s advertising software allows him to choose from 30 different colors. Is there a text color that will bring more clicks than blue? Fred decides to find out.

Fred instigates an experiment. Every weekday for a month, Fred purchases 3,000 online ads. The text of every ad is assigned to one of 30 possible colors. The advertisements are distributed evenly by color. Thus, 100 ads with the same color are viewed by 100 people every day. For example, 100 people view a blue ad, and another 100 people view a green ad. These numbers add up to 3,000 views that are distributed across the 30 colors. Fred’s advertising software automatically tracks all daily views. It also records the daily clicks associated with each of the 30 colors. The software stores this data in a table. That table holds the clicks per day and views per day for every specified color. Each table row maps a color to the views and clicks for all analyzed days.

Fred has carried out his experiment. He obtained ad-click data for all 20 weekdays of the month. That data is organized by color. Now, Fred wants to know if there is a color that draws significantly more ad clicks than blue. Unfortunately, Fred doesn’t know how to properly interpret the results. He’s not sure which clicks are meaningful and which clicks have occurred purely randomly. Fred is brilliant at broiling burgers but has no training in data analysis. This is why Fred has turned to you for help. Fred asks you to analyze his table and to compare the counts of daily clicks. He’s searching for a color that draws significantly more ad clicks than blue. Are you willing to help Fred? If so, he’s promised you free burgers for a year!

### Dataset description

Fred’s ad-click data is stored in the file colored_ad_click_table.csv. The first 99 characters of that line are *Color,Click Count: Day 1,View Count: Day 1,Click Count: Day 2,View Count: Day 2,Click Count: Day 3,*.

Let’s briefly clarify the column labels:

- Column 1: *Color*
    - Each row in the column corresponds to one of 30 possible text colors.
- Column 2: *Click Count: Day 1*
    - The column tallies the times each colored ad was clicked on day 1 of Fred’s experiment.
- Column 3: *View Count: Day 1*
    - The column tallies the times each ad was viewed on day 1 of Fred’s experiment.
    - According to Fred, all daily views are expected to equal 100.
- The remaining 38 columns contain the clicks per day and views per day for the other 19 days of the experiment.

To address the problem at hand, we need to know how to do the following: 1) Measure the centrality and dispersion of sampled data, 2) Interpret the significance of two diverging means through p-value calculation, 3) Minimize mistakes associated with misleading p-value measurements, 4) Load and manipulate data stored in tables using Python.

Our aim is to discover an ad color that generates significantly more clicks than blue. We will do so by following these steps:

1. Load and clean our advertising data using Pandas.
2. Run a permutation test between blue and the other recorded colors.
3. Check the computed p-values for statistical significance using a properly determined significance level.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="S3g09yrUCHPx" executionInfo={"status": "ok", "timestamp": 1637305851357, "user_tz": -330, "elapsed": 577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ca625276-f7d4-46aa-eef9-f9d845a1a089"
!wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T426474/siteC/colored_ad_click_table.csv
```

```python id="2OybbQN9CJh4"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python colab={"base_uri": "https://localhost:8080/"} id="lyZGHnkkCLS6" executionInfo={"status": "ok", "timestamp": 1637305869568, "user_tz": -330, "elapsed": 716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16b3ecdf-17f4-4f3d-a2d2-f634bf997298"
df = pd.read_csv('colored_ad_click_table.csv')
num_rows, num_cols = df.shape
print(f"Table contains {num_rows} rows and {num_cols} columns")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 261} id="TbUwRTRqCQ3n" executionInfo={"status": "ok", "timestamp": 1637305888620, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a80568dd-59b5-49c3-a403-0d8e065b22b1"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="K0jw2FYnCQxR" executionInfo={"status": "ok", "timestamp": 1637305950692, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="241e38c2-385d-4b5d-e8ec-5239c7468aad"
df.Color.values
```

```python colab={"base_uri": "https://localhost:8080/"} id="I0fOXg8lChTr" executionInfo={"status": "ok", "timestamp": 1637305973656, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b02df49-4d4a-4a2d-9679-8c90841a8834"
selected_columns = ['Color', 'Click Count: Day 1', 'View Count: Day 1']
print(df[selected_columns].describe())
```

<!-- #region id="BSkJr7VCD-hh" -->
The values in the Click Count: Day 1 column range from 12 to 49 clicks. Meanwhile, the minimum and maximum values in View Count: Day 1 are both equal to 100 views. Therefore, all the values in that column are equal to 100 views. This behavior is expected. We were specifically informed that each color receives 100 daily views. Let’s confirm that all the daily views equal 100.
<!-- #endregion -->

```python id="cOZpOu6kCx5x"
view_columns = [column for column in df.columns if 'View' in column]
assert np.all(df[view_columns].values == 100)
```

<!-- #region id="sBCzd6lSEHoS" -->
All view counts equal 100. Therefore, all 20 View Count columns are redundant. We can delete them from our table.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tJXDcIqfCyKZ" executionInfo={"status": "ok", "timestamp": 1637306056348, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0beede85-d6bc-4b59-b2ea-c6ded49760b4"
df.drop(columns=view_columns, inplace=True)
print(df.columns)
```

<!-- #region id="8HUD4x_1EKgp" -->
The redundant columns have been removed. Only the color and click-count data remain. Our 20 Click Count columns correspond to the number of clicks per 100 daily views, so we can treat these counts as percentages. Effectively, the color in each row is mapped to the percentage of daily ad clicks. Let’s summarize the percentage of daily ad clicks for blue ads. To generate that summary, we index each row by color and then call df.T.Blue.describe().
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="phYGg4zwC7ej" executionInfo={"status": "ok", "timestamp": 1637306070221, "user_tz": -330, "elapsed": 466, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0120034-725a-4db8-d1c8-7a9855b873ab"
df.set_index('Color', inplace=True)
print(df.T.Blue.describe())
```

<!-- #region id="1EcrP8k1EUyC" -->
The daily click percentages for blue range from 18% to 42%. The mean percent of clicks is 28.35%: on average, 28.35% of blue ads receive a click per view. This average click rate is pretty good. How does it compare to the other 29 colors? We are ready to find out.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 202} id="lpId463sC_4I" executionInfo={"status": "ok", "timestamp": 1637306104172, "user_tz": -330, "elapsed": 696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a909f5c9-52e8-4889-84b3-9377594d2149"
df_not_blue = df.T.drop(columns='Blue')
df_not_blue.head(2)
```

<!-- #region id="swpuqx66EZeu" -->
Our df_not_blue table contains the percent clicks for 29 colors. We would like to compare these percentages to our blue percentages. More precisely, we want to know if there exists a color whose mean click rate is statistically different from the mean click rate of blue. How do we compare these means? The sample mean for every color is easily obtainable, but we do not have a population mean. Thus, our best option is to run a permutation test. To run the test, we need to define a reusable permutation test function. The function will take as input two NumPy arrays and return a p-value as its output.
<!-- #endregion -->

```python id="7M49rBeDDHLl"
def permutation_test(data_array_a, data_array_b):
    data_mean_a = data_array_a.mean()
    data_mean_b = data_array_b.mean()
    extreme_mean_diff = abs(data_mean_a - data_mean_b)
    total_data = np.hstack([data_array_a, data_array_b])
    number_extreme_values = 0.0
    for _ in range(30000):
        np.random.shuffle(total_data)
        sample_a = total_data[:data_array_a.size]
        sample_b = total_data[data_array_a.size:]
        if abs(sample_a.mean() - sample_b.mean())  >= extreme_mean_diff:
            number_extreme_values += 1

    p_value = number_extreme_values / 30000
    return p_value
```

<!-- #region id="Wo3rH6mAEjG5" -->
We’ll run a permutation test between blue and the other 29 colors. Then we’ll sort these colors based on their p-value results. Our outputs are visualized as a heatmap, to better emphasize the differences between p-values.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 592} id="Cf6bAHggDNKy" executionInfo={"status": "ok", "timestamp": 1637306235248, "user_tz": -330, "elapsed": 17899, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba36f4f1-1922-4fdb-dbde-36285f32de61"
np.random.seed(0)
blue_clicks = df.T.Blue.values
color_to_p_value = {}
for color, color_clicks in df_not_blue.items():
    p_value = permutation_test(blue_clicks, color_clicks)
    color_to_p_value[color] = p_value

sorted_colors, sorted_p_values = zip(*sorted(color_to_p_value.items(),
                                             key=lambda x: x[1]))
plt.figure(figsize=(3, 10))
sns.heatmap([[p_value] for p_value in sorted_p_values],
            cmap='YlGnBu', annot=True, xticklabels=['p-value'],
            yticklabels=sorted_colors)
plt.show()
```

<!-- #region id="0YN98mZyE-v4" -->
The majority of colors generate a p-value that is noticeably lower than 0.05. Black has the lowest p-value: its ad-click percentages must deviate significantly from blue. But from a design perspective, black is not a very clickable color. Text links usually are not black, because black links are hard to distinguish from regular text. Something suspicious is going on here: what exactly is the difference between recorded clicks for black and blue? We can check by printing df_not_blue.Black.mean().
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bZaBIc-rFChD" executionInfo={"status": "ok", "timestamp": 1637306610934, "user_tz": -330, "elapsed": 777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b52f2473-6167-4dab-bc01-e7d196d9f47d"
mean_black = df_not_blue.Black.mean()
print(f"Mean click-rate of black is {mean_black}")
```

<!-- #region id="jWsMyCKHFITa" -->
The mean click rate of black is 21.6. This value is significantly lower than the blue mean of 28.35. Hence, the statistical difference between the colors is caused by fewer people clicking black. Perhaps other low p-values are also caused by inferior click rates. Let’s filter out those colors whose mean is less than the mean of blue and then print the remaining colors.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XlCcknHHDUWg" executionInfo={"status": "ok", "timestamp": 1637306252834, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cf9be97a-57f4-4e67-f20b-b6069414272d"
remaining_colors = df[df.T.mean().values > blue_clicks.mean()].index
size = remaining_colors.size
print(f"{size} colors have on average more clicks than Blue.")
print("These colors are:")
print(remaining_colors.values)
```

<!-- #region id="Dn_gdKV_FNXt" -->
Only five colors remain. Each of these colors is a different shade of blue. Let’s print the sorted p-values for the five remaining colors; we also print the mean clicks for easier analysis.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CKPEV1BxDrgm" executionInfo={"status": "ok", "timestamp": 1637306253298, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dada956c-b02b-4752-ddae-6a8bc534651a"
for color, p_value in sorted(color_to_p_value.items(), key=lambda x: x[1]):
    if color in remaining_colors:
        mean = df_not_blue[color].mean()
        print(f"{color} has a p-value of {p_value} and a mean of {mean}")
```

<!-- #region id="9GPyL-TJFe-M" -->
Four of the colors have large p-values. Only one color has a p-value that’s small. That color is ultramarine: a special shade of blue. Its mean of 34.2 is greater than blue’s mean of 28.35. Ultramarine’s p-value is 0.0034. Is that p-value statistically significant? Well, it’s more than 10 times lower than the standard significance level of 0.05. However, that significance level does not take into account our comparisons between blue and 29 other colors. Each comparison is an experiment testing whether a color differs from blue. If we run enough experiments, then we are guaranteed to encounter a low p-value sooner or later. The best way to correct for this is to execute a Bonferroni correction—otherwise, we will fall victim to p-value hacking. To carry out a Bonferroni correction, we lower the significance level to 0.05 / 29.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Z6vlYtavDrsV" executionInfo={"status": "ok", "timestamp": 1637306275447, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5e13e8d0-d939-4c74-80e1-e452dfeecbcd"
significance_level = 0.05 / 29
print(f"Adjusted significance level is {significance_level}")
if color_to_p_value['Ultramarine'] <= significance_level:
    print("Our p-value is statistically significant")
else:
    print("Our p-value is not statistically significant")
```

<!-- #region id="In19EiA0Frpf" -->
Our p-value is not statistically significant—Fred carried out too many experiments for us to draw a meaningful conclusion. Not all of these experiments were necessary. There is no valid reason to expect that black, brown, or gray would outperform blue. Perhaps if Fred had disregarded some of these colors, our analysis would have been more fruitful. Conceivably, if Fred had simply compared blue to the other five variants of blue, we might have obtained a statistically significant result. Let’s explore the hypothetical situation where Fred instigates five experiments and ultramarine’s p-value remains unchanged.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hOeYFcy1Dw-Y" executionInfo={"status": "ok", "timestamp": 1637306281428, "user_tz": -330, "elapsed": 800, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6b211fc3-83be-4b67-c21e-50ceeea53bf0"
hypothetical_sig_level = 0.05 / 5
print(f"Hypothetical significance level is {hypothetical_sig_level}")
if color_to_p_value['Ultramarine'] <= hypothetical_sig_level:
    print("Our hypothetical p-value would have been statistically significant")
else:
    print("Our hypothetical p-value would not not have been statistically significant")
```

<!-- #region id="kEg4t7cSDyfD" -->
Under these hypothetical conditions, our results would be statistically significant. Sadly, we can’t use the hypothetical conditions to lower our significance level. We have no guarantee that rerunning the experiments would reproduce a p-value of 0.0034. P-values fluctuate, and superfluous experiments increase the chance of untrustworthy fluctuations. Given Fred’s high experiment count, we simply cannot draw a statistically significant conclusion.

However, all is not lost. Ultramarine still represents a promising substitute for blue. Should Fred carry out that substitution? Perhaps. Let’s consider our two alternative scenarios. In the first scenario, the null hypothesis is true. If that’s the case, then both blue and ultramarine share the same population mean. Under these circumstances, swapping ultramarine for blue will not affect the ad click rate. In the second scenario, the higher ultramarine click rate is actually statistically significant. If that’s the case, then swapping ultramarine for blue will yield more ad clicks. Therefore, Fred has everything to gain and nothing to lose by setting all his ads to ultramarine.

From a logical standpoint, Fred should definitely swap blue for ultramarine. But if he carries out the swap, some uncertainty will remain; Fred will never know if ultramarine truly returns more clicks than blue. What if Fred’s curiosity gets the best of him? If he really wants an answer, his only choice is to run another experiment. In that experiment, half the displayed ads would be blue and the other displayed ads would be ultramarine. Fred’s software would exhibit the advertisements while recording all the clicks and views. Then we could recompute the p-value and compare it to the appropriate significance level, which would remain at 0.05. The Bonferroni correction would not be necessary because only a single experiment would be run. After the p-value comparison, Fred would finally know whether ultramarine outperforms blue.
<!-- #endregion -->

<!-- #region id="qYb9uRoGGt_X" -->
More data isn’t always better. Running a pointless surplus of analytic tests increases the chance of anomalous results.

It’s worth taking the time to think about a problem before running an analysis. If Fred had carefully considered the 31 colors, he would have realized that it was pointless to test them all. Many colors make ugly links. Colors like black are very unlikely to yield more clicks than blue. Filtering the color set would have led to a more informative test.

Even though Fred’s experiment was flawed, we still managed to extract a useful insight. Ultramarine might prove to be a reasonable substitute for blue, though more testing is required. Occasionally, data scientists are presented with flawed data, but good insights may still be possible.
<!-- #endregion -->

<!-- #region id="ykRdtHaPGr-o" -->
Fred assumed that analyzing every single color would yield more impactful results, but he was wrong. More data isn’t necessarily better: sometimes more data leads to more uncertainty.

Fred is not a statistician. He can be forgiven for failing to comprehend the consequences of overanalysis. The same cannot be said of certain quantitative experts operating in business today. Take, for example, a notorious incident that occurred at a well-known corporation. The corporation needed to select a color for the web links on its site. The chief designer chose a visually appealing shade of blue, but a top-level executive distrusted this decision. Why did the designer choose this shade of blue and not another?

The executive came from a quantitative background and insisted that link color should be selected scientifically via a massive analytic test that would supposedly determine the perfect shade of blue. 41 shades of blue were assigned to company web links completely at random, and millions of clicks were recorded. Eventually, the “optimal” shade of blue was selected based on maximum clicks per view.

The executive proceeded to make the methodology public. Worldwide, statisticians cringed. The executive’s decisions revealed an ignorance of basic statistics, and that ignorance embarrassed both the executive and the company.
<!-- #endregion -->
