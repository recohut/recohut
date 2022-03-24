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

<!-- #region id="jzX21dhMG60W" -->
# Ad-conversion A-B Testing
> Simple T-test to compare 2 samples of ad-conversions

- toc: true
- badges: true
- comments: true
- categories: [ABTest]
- image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vQ13VFgf49GE" outputId="31f18908-5d9a-4b87-d4f2-1bb3dd397b33"
!wget https://github.com/sparsh-ai/reco-data/raw/master/abtest-sample-dayC1C2.csv
```

```python id="6j04Iy0N5C7h"
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="4ytG1m5A5Dy3" outputId="d5152c41-62ec-419b-9967-eae6a685b8b1"
data= pd.read_csv("abtest-sample-dayC1C2.csv")
data.head(10)
```

<!-- #region id="OO0ehoqj5L0j" -->
Let’s plot the distribution of target and control group:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="Jok5xwJD5JAX" outputId="c6bba329-983e-4264-d8b0-c8ed7c93bdc6"
sns.distplot(data.Conversion_A)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="JVZ7sxHM5Nbv" outputId="63f18dc1-c6dd-411e-8207-c139125dee6d"
sns.distplot(data.Conversion_B)
```

<!-- #region id="i00JbDz15QsV" -->
Now, we will perform the t-test:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QOuEKhy05PGB" outputId="c1c4fb0d-8ac9-4e19-c17e-45b4d4f19878"
t_stat, p_val= ss.ttest_ind(data.Conversion_B,data.Conversion_A)
t_stat , p_val
```

<!-- #region id="wEZk66_4GKyQ" -->
For our example, the observed value i.e the mean of the test group is 0.19. The hypothesized value (Mean of the control group) is 0.16. On the calculation of the t-score, we get the t-score as .3787. and the p-value is 0.00036.

SO what does all this mean for our A/B Testing?

Here, our p-value is less than the significance level i.e 0.05. Hence, we can reject the null hypothesis. This means that in our A/B testing, newsletter B is performing better than newsletter A. So our recommendation would be to replace our current newsletter with B to bring more traffic on our website.
<!-- #endregion -->
