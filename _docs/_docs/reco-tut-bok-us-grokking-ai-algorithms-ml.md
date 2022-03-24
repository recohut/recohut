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

```python id="4sv1j8lRtwKQ" executionInfo={"status": "ok", "timestamp": 1630127475157, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-bok"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="n8RiXVLstuoO" executionInfo={"status": "ok", "timestamp": 1630127477779, "user_tz": -330, "elapsed": 1977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e935589-bd11-486e-8461-51ffadab71da"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="drkDUesKtuoT"
!git status
```

```python id="pWkwGc0ltuoT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630133860986, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3157576-0d48-4774-b9d6-5cd1a4b55c33"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="1RxCMJ8JuoZn" executionInfo={"status": "ok", "timestamp": 1630127480995, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c97eb27c-f7fa-401d-fae9-797965228943"
%cd /content/reco-tut-bok/US221014
```

<!-- #region id="zzsGQbcDuMei" -->
---
<!-- #endregion -->

<!-- #region id="Ee3YeafXK5Vf" -->
<!-- #endregion -->

<!-- #region id="Xu4nnCZGK19h" -->
### Collecting and understanding data: Know your context
Collecting and understanding the data you’re working with is paramount to a successful machine learning endeavor. If you’re working in a specific area in the finance industry, knowledge of the terminology and workings of the processes and data in that area is important for sourcing the data that is best to help answer questions for the goal you’re trying to achieve.

Data may also need to be sourced from various systems and combined to be effective. Sometimes, the data we use is augmented with data from outside the organization to enhance accuracy. In this section, we use an example dataset about diamond measurements to understand the machine learning workflow and explore various algorithms.
<!-- #endregion -->

<!-- #region id="HZg9wR1wLbdU" -->
<!-- #endregion -->

```python id="oTDNml7ZLpJy"
# !wget -O ./data/diamonds.csv https://github.com/rishal-hurbans/Grokking-Artificial-Intelligence-Algorithms/raw/master/ch08-machine_learning/diamonds.csv
```

```python id="MASiHSigL1Pp" executionInfo={"status": "ok", "timestamp": 1630129372640, "user_tz": -330, "elapsed": 1083, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8MqOvDnmL79r" executionInfo={"status": "ok", "timestamp": 1630127784751, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8123b8ae-d631-4e3a-8805-510c06c42693"
df = pd.read_csv('./data/diamonds.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="V-5mkkKwL_-H" executionInfo={"status": "ok", "timestamp": 1630127832050, "user_tz": -330, "elapsed": 558, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="39ce79bb-f4a4-49b7-ccc7-f76bdbb25c14"
df.info()
```

<!-- #region id="xBzrxHPdNrvT" -->
- Carat—The weight of the diamond. Out of interest: 1 carat equals 200 mg.
- Cut—The quality of the diamond, by increasing quality: fair, good, very good, premium, and ideal.
- Color—The color of the diamond, ranging from D to J, where D is the best color and J is the worst color. D indicates a clear diamond, and J indicates a foggy one.
- Clarity—The imperfections of the diamond, by decreasing quality: FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, and I3. (Don’t worry about understanding these code names; they simply represent different levels of perfection.)
- Depth—The percentage of depth, which is measured from the culet to the table of the diamond. Typically, the table-to-depth ratio is important for the “sparkle” aesthetic of a diamond.
- Table—The percentage of the flat end of the diamond relative to the X dimension.
- Price—The price of the diamond when it was sold.
- X—The x dimension of the diamond, in millimeters.
- Y—The y dimension of the diamond, in millimeters.
- Z—The z dimension of the diamond, in millimeters.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="7dSgOXXfMSq0" executionInfo={"status": "ok", "timestamp": 1630127863811, "user_tz": -330, "elapsed": 728, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ab499d1-eaf1-4dae-b054-f1ff7e48d75b"
df.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 498} id="Cu7aUu7PMbc-" executionInfo={"status": "ok", "timestamp": 1630127922579, "user_tz": -330, "elapsed": 1650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60c67525-37c6-4ce5-a7df-79cea4100a3a"
fig, ax = plt.subplots(figsize=(12,7))
df.hist(ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="-KBsI-cuMwEL" executionInfo={"status": "ok", "timestamp": 1630127983405, "user_tz": -330, "elapsed": 533, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4c9d362d-39f5-4596-aa5d-bf39f80308c0"
df.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="2wu8qRGEM3wi" executionInfo={"status": "ok", "timestamp": 1630129063625, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0525c23e-43b3-4766-c22b-e3b7daa008c6"
df.cut.astype('str').value_counts().plot(kind='barh');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="NGdAlGUpM7t1" executionInfo={"status": "ok", "timestamp": 1630128060444, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9debe430-78e4-4cc7-bdc5-f4aa23fa304b"
df.color.astype('str').value_counts().plot(kind='barh');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="cTCqWdqOMHWV" executionInfo={"status": "ok", "timestamp": 1630128069737, "user_tz": -330, "elapsed": 481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0abc6674-200f-44c9-8f61-b1f084db8346"
df.clarity.astype('str').value_counts().plot(kind='barh');
```

<!-- #region id="QTkSFirBN-So" -->
### Preparing data: Clean and wrangle
Real-world data is never ideal to work with. Data might be sourced from different systems and different organizations, which may have different standards and rules for data integrity. There are always missing data, inconsistent data, and data in a format that is difficult to work with for the algorithms that we want to use.
<!-- #endregion -->

```python id="pGmIXN8bSGM0" executionInfo={"status": "ok", "timestamp": 1630129384990, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Define the features that we are interested in
feature_x = 'carat'
feature_y = 'price'
```

<!-- #region id="4VVZaoflTvu7" -->
Carat as the independent variable (x)—An independent variable is one that is changed in an experiment to determine the effect on a dependent variable. In this example, the value for carats will be adjusted to determine the price of a diamond with that value.

Price as the dependent variable (y)—A dependent variable is one that is being tested. It is affected by the independent variable and changes based on the independent variable value changes. In our example, we are interested in the price given a specific carat value.
<!-- #endregion -->

```python id="4-rX7VL9SMNV" executionInfo={"status": "ok", "timestamp": 1630129424984, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Filter the data based on the "cut" feature
fair_diamonds = df[df['cut'] == "Fair"]
good_diamonds = df[df['cut'] == "Good"]
very_good_diamonds = df[df['cut'] == "Very Good"]
premium_diamonds = df[df['cut'] == "Premium"]
ideal_diamonds = df[df['cut'] == "Ideal"]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="HlajewdpSTgf" executionInfo={"status": "ok", "timestamp": 1630129488032, "user_tz": -330, "elapsed": 1201, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="44b2454c-65ad-4fa6-d811-6bcd56851464"
# Plot the filtered data as a scatter plot
fig = plt.figure()
plt.title(feature_x + ' vs ' + feature_y)

plt.scatter(fair_diamonds[feature_x], fair_diamonds[feature_y], label="Fair", s=1.8)
plt.scatter(good_diamonds[feature_x], good_diamonds[feature_y], label="Good", s=1.8)
plt.scatter(very_good_diamonds[feature_x], very_good_diamonds[feature_y], label="Very Good", s=1.8)
plt.scatter(premium_diamonds[feature_x], premium_diamonds[feature_y], label="Premium", s=1.8)
plt.scatter(ideal_diamonds[feature_x], ideal_diamonds[feature_y], label="Ideal", s=1.8)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.show()
```

```python id="_K2PkbDHSBD2" executionInfo={"status": "ok", "timestamp": 1630129587909, "user_tz": -330, "elapsed": 700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Encode the string values for "cut", "color", and "clarity" as integer values
encoding_categories = {'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}

df.replace(encoding_categories, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 291} id="rlD4Lx4pReyF" executionInfo={"status": "ok", "timestamp": 1630129589842, "user_tz": -330, "elapsed": 583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ac01bf9b-ec9e-405b-ac19-d9881147f262"
# Plot the filtered df as a heat map based on "cut", "color", and "clarity"
df_subset = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x size', 'y size', 'z size']]
cor = df_subset.corr()
sns.heatmap(cor, square=True)
plt.show()
```

<!-- #region id="Pt8Rf7-HOGIs" -->
### Training a model: Predict with linear regression
Choosing an algorithm to use is based largely on two factors: the question that is being asked and the nature of the data that is available. If the question is to make a prediction about the price of a diamond with a specific carat weight, regression algorithms can be useful. The algorithm choice also depends on the number of features in the dataset and the relationships among those features. If the data has many dimensions (there are many features to consider to make a prediction), we can consider several algorithms and approaches.
<!-- #endregion -->

```python id="WGSu7SnHUzn-" executionInfo={"status": "ok", "timestamp": 1630130095525, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import matplotlib.pyplot as plt
import statistics
```

```python id="U0ZKpggWU2wD" executionInfo={"status": "ok", "timestamp": 1630130167475, "user_tz": -330, "elapsed": 663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Carat values for each diamond
carats = [0.3, 0.41, 0.75, 0.91, 1.2, 1.31, 1.5, 1.74, 1.96, 2.21]

# Scale the carat values for each diamond to be similarly sized to the price
carats = [i * 1000 for i in carats]

# Price values for each diamond
price = [339, 561, 2760, 2763, 2809, 3697, 4022, 4677, 6147, 6535]
```

<!-- #region id="WsoniY74YqnK" -->
<!-- #endregion -->

<!-- #region id="o_cITq1mYwWR" -->
<!-- #endregion -->

<!-- #region id="Dd4dDw10ZZ8R" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ddyDLumFVZav" executionInfo={"status": "ok", "timestamp": 1630130672537, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e153a2b0-04ba-4d9f-bf2c-397835b78f67"
# Calculate the mean for 'price' and 'carat'
mean_X = statistics.mean(carats)
print('Mean of X: {}'.format(mean_X))
mean_Y = statistics.mean(price)
print('Mean of Y: {}'.format(mean_Y))

# Calculate the number of examples in the dataset
number_of_examples = len(carats)

# Print values for x (Carats)
print('\nx', end=': ')
for i in range(number_of_examples):
    print('{0:.0f}'.format(carats[i]), end=' ')

# Print values for x - mean of x
print('\nx - x mean', end=': ')
for i in range(number_of_examples):
    print('{0:.0f}'.format(carats[i] - mean_X), end=' ')

# Print values for y - mean of y
print('\ny - y mean', end=': ')
for i in range(number_of_examples):
    print('{0:.0f}'.format(price[i] - mean_Y), end=' ')

# Print values for x - (x mean)^2
print('\nx - (x mean)^2', end=': ')
sum_x_squared = 0
for i in range(number_of_examples):
    ans = (carats[i] - mean_X) ** 2
    sum_x_squared += ans
    print('{0:.0f}'.format(ans), end=' ')

print('\nSUM squared: ', sum_x_squared)

# Print values for x - x mean * y - y mean
print('\n(x - x mean) * (y - y mean)', end=': ')
sum_multiple = 0
for i in range(number_of_examples):
    ans = (carats[i] - mean_X) * (price[i] - mean_Y)
    sum_multiple += ans
    print('{0:.0f}'.format(ans), end=' ')

print('\nSUM multi: ', sum_multiple)
```

<!-- #region id="vOl4jK7YY9vr" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="R2BHt7mTXGN9" executionInfo={"status": "ok", "timestamp": 1630130696418, "user_tz": -330, "elapsed": 1390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a679c4a3-0bf9-47da-f27f-283d028b69fb"
b1 = sum_multiple / sum_x_squared
print('b1: ', b1)
b0 = mean_Y - (b1 * mean_X)
print('b0: ', b0)
min_x = np.min(carats)
max_x = np.max(carats)
x = np.linspace(min_x, max_x, 10)

# Express the regression line by y = mx + c
y = b0 + b1 * x
```

```python id="PRQsAJYOXUuE" executionInfo={"status": "ok", "timestamp": 1630130754761, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Testing data
carats_test = [220, 330, 710, 810, 1080, 1390, 1500, 1640, 1850, 1910]

price_test = [342, 403, 2772, 2789, 2869, 3914, 4022, 4849, 5688, 6632]
```

<!-- #region id="ukJlB7B7ZmlR" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="k2OSY7nXXeYK" executionInfo={"status": "ok", "timestamp": 1630130807723, "user_tz": -330, "elapsed": 674, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b65c1dd0-3798-4e6c-8559-46486c60f184"
price_test_mean = statistics.mean(price_test)
print('price test mean: ', price_test_mean)
price_test_n = len(price_test)
print('price test difference:')
for i in range(price_test_n):
    print(price_test[i] - price_test_mean)

print('price test difference squared:')
sum_of_price_test_difference = 0
for i in range(price_test_n):
    ans = (price_test[i] - price_test_mean) ** 2
    sum_of_price_test_difference += ans
    print(ans)
print('sum diff: ', sum_of_price_test_difference)

print('predicted values:')
for i in range(price_test_n):
    print('{0:.0f}'.format(b0 + carats_test[i] * b1))
print('predicted values - mean:')
for i in range(price_test_n):
    print('{0:.0f}'.format((b0 + carats_test[i] * b1) - price_test_mean))

print('predicted values - mean squared:')
sum_of_price_test_prediction_difference = 0
for i in range(price_test_n):
    ans = ((b0 + carats_test[i] * b1) - price_test_mean) ** 2
    sum_of_price_test_prediction_difference += ans
    print('{0:.0f}'.format(ans))
print('sum prediction: ', sum_of_price_test_prediction_difference)

# Calculate the R^2 score
ss_numerator = 0
ss_denominator = 0
for i in range(number_of_examples):
    y_predicted = b0 + b1 * carats_test[i]
    ss_numerator += ((price_test[i] - mean_Y) - y_predicted) ** 2
    ss_denominator += (price_test[i] - mean_Y) ** 2
r2 = ss_numerator / ss_denominator
print('R2: ', r2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 902} id="s947CQvhT7ja" executionInfo={"status": "ok", "timestamp": 1630130837473, "user_tz": -330, "elapsed": 1627, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5647782a-23ff-4c9b-d1c9-8ed5798cc1d5"
# Plot the data on a figure to better understand visually
fig = plt.figure()
plt.figure(num=None, figsize=(5, 5), dpi=300, facecolor='w', edgecolor='w')

# Plot the original training data in red
plt.scatter(carats, price, color='red', label='Scatter Plot')
# Plot the testing data in black
plt.scatter(carats_test, price_test, color='black', label='Scatter Plot')
# Plot lines to represent the mean for x and y in gray
plt.axvline(x=mean_X, color='gray')
plt.axhline(y=mean_Y, color='gray')
# Plot the regression line using the min and max for carats
rex_x = [300, 2210]
rex_y = [515.7, 6511.19]
plt.plot(rex_x, rex_y, color='green')
# Label the figure, save it, and show it
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()
```

<!-- #region id="l4NABgVbXo58" -->
### Sklearn linear model
<!-- #endregion -->

```python id="dJU9w6h2X3Xc" executionInfo={"status": "ok", "timestamp": 1630130971866, "user_tz": -330, "elapsed": 628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

```python id="X-MfBKNwX-gm" executionInfo={"status": "ok", "timestamp": 1630130973064, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Data:
    """A data class to read and prepare data into training and testing data"""
    def __init__(self, data_file_name, excluded_features, label, encoded_categories):
        data_file = pd.read_csv(data_file_name)
        data_file.replace(encoded_categories, inplace=True)
        X = data_file.drop(columns=excluded_features)
        y = data_file[label]
        X = X.drop(columns=label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.5)

    def enumerate_categories(self, encoded_categories):
        self.X_train.replace(encoded_categories, inplace=True)
        self.X_test.replace(encoded_categories, inplace=True)
        self.y_train.replace(encoded_categories, inplace=True)
        self.y_test.replace(encoded_categories, inplace=True)
```

```python id="tt9OsYGRbJNX"
# Encode the string values for "cut", "color", and "clarity" as integer values
encoding_categories = {'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}


# Read the data file into a data frame
data = Data('./data/diamonds.csv', ['no'], 'clarity', encoding_categories)
# Initialize linear regression model
regression = linear_model.LinearRegression()
# Filter training data to be used for linear regression, namely, "price" and "carat"
regression_train_x = data.X_train['price'].values[:-1]
regression_train_y = data.X_train['carat'].values[:-1]
# Fit the model based on the data
regression = regression.fit(regression_train_x.reshape(-1, 1), regression_train_y.reshape(-1, 1))
```

<!-- #region id="SWFN1K7vbLg5" -->
### Testing the model: Determine the accuracy of the model
Now that we have determined a regression line, we can use it to make price predictions for other Carat values. We can measure the performance of the regression line with new examples in which we know the actual price and determine how accurate the linear regression model is.

We can’t test the model with the same data that we used to train it. This approach would result in high accuracy and be meaningless. The trained model must be tested with real data that it hasn’t been trained with.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 333} id="zaohRIBZYJxr" executionInfo={"status": "ok", "timestamp": 1630130986114, "user_tz": -330, "elapsed": 1771, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0eddff2b-6ded-4da6-c06e-ca2294438818"
# Filter testing data to be used for linear regression, namely, "price" and "carat"
reg_test_x = data.X_test['price'].values[:]
reg_test_y = data.X_test['carat'].values[:]
# Predict using the trained linear regression model
prediction_y = regression.predict(reg_test_x.reshape(-1, 1))
prediction_y = prediction_y.reshape(-1, 1)
# Print the coefficients
print('Coefficients: \n', regression.coef_)
# Print the mean squared error
print('Mean squared error: ', metrics.mean_squared_error(reg_test_y, prediction_y))
# Print the variance score: 1 is a perfect prediction
print('Variance score: ', metrics.r2_score(reg_test_y, prediction_y))

# Plot the testing data and predicted data
plt.scatter(reg_test_x, reg_test_y,  color='black')
plt.plot(reg_test_x, prediction_y, color='red', linewidth=3)
plt.show()
```

<!-- #region id="3f0W42aXYM7C" -->
## Decision Trees
<!-- #endregion -->

<!-- #region id="11eq0eErblgQ" -->
Decision trees are structures that describe a series of decisions that are made to find a solution to a problem.
If we’re deciding whether or not to wear shorts for the day, we might make a series of decisions to inform the
outcome. Will it be cold during the day? If not, will we be out late in the evening when it does get cold?
We might decide to wear shorts on a warm day, but not if we will be out when it gets cold.
In building a decision tree, all possible questions will be tested to determine which one is the best question to
ask at a specific point in the decision tree. To test a question, the concept of entropy is used. Entropy is the
uncertainty of the dataset.
<!-- #endregion -->

<!-- #region id="6qP39fzqbqRv" -->
### Toy example
<!-- #endregion -->

```python id="ZKw5ZauxbuIk" executionInfo={"status": "ok", "timestamp": 1630131908196, "user_tz": -330, "elapsed": 664, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# The data used for learning
feature_names = ['carat', 'price', 'cut']
feature_examples = [[0.21, 327, 'Average'],
                    [0.39, 897, 'Perfect'],
                    [0.50, 1122, 'Perfect'],
                    [0.76, 907, 'Average'],
                    [0.87, 2757, 'Average'],
                    [0.98, 2865, 'Average'],
                    [1.13, 3045, 'Perfect'],
                    [1.34, 3914, 'Perfect'],
                    [1.67, 4849, 'Perfect'],
                    [1.81, 5688, 'Perfect']]
```

```python id="lOKRWI8Rb3S7" executionInfo={"status": "ok", "timestamp": 1630132273834, "user_tz": -330, "elapsed": 631, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Question:
    """The Question class defines a feature and value that it should satisfy"""
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def filter(self, example):
        value = example[self.feature]
        return value >= self.value

    def to_string(self):
        return 'Is ' + feature_names[self.feature] + ' >= ' + str(self.value) + '?'
```

```python id="xJdqjApldNo4" executionInfo={"status": "ok", "timestamp": 1630132347175, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class ExamplesNode:
    """The ExamplesNode class defines a node in the tree that contains classified examples"""
    def __init__(self, examples):
        self.examples = find_unique_label_counts(examples)
```

```python id="tKwy-k8odVgA" executionInfo={"status": "ok", "timestamp": 1630132347177, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class DecisionNode:
    """The DecisionNode class defines a node in the tree that contains a question, and two branches"""
    def __init__(self, question, branch_true, branch_false):
        self.question = question
        self.branch_true = branch_true
        self.branch_false = branch_false
```

```python id="4coDJfgrde28" executionInfo={"status": "ok", "timestamp": 1630132371779, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Count the unique classes and their counts from a list of examples
def find_unique_label_counts(examples):
    class_count = {}
    for example in examples:
        label = example[-1]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count
```

```python id="_PJblnOIdey2" executionInfo={"status": "ok", "timestamp": 1630132375861, "user_tz": -330, "elapsed": 749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Split a list of examples based on a question being asked
def split_examples(examples, question):
    examples_true = []
    examples_false = []
    for example in examples:
        if question.filter(example):
            examples_true.append(example)
        else:
            examples_false.append(example)
    return examples_true, examples_false
```

```python id="W-CpDOxTdeuX" executionInfo={"status": "ok", "timestamp": 1630132381044, "user_tz": -330, "elapsed": 639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Calculate the Gini Index based on a list of examples
def calculate_gini(examples):
    label_counts = find_unique_label_counts(examples)
    uncertainty = 1
    for label in label_counts:
        probability_of_label = label_counts[label] / float(len(examples))
        uncertainty -= probability_of_label ** 2
    return uncertainty
```

<!-- #region id="5NcdAvpFgXHK" -->
<!-- #endregion -->

```python id="JOXAu0mQdkf4" executionInfo={"status": "ok", "timestamp": 1630132394716, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Calculate the information gain based on the left gini, right gini, and current uncertainty
def calculate_information_gain(left_gini, right_gini, current_uncertainty):
    total = len(left_gini) + len(right_gini)
    gini_left = calculate_gini(left_gini)
    entropy_left = len(left_gini) / total * gini_left
    gini_right = calculate_gini(right_gini)
    entropy_right = len(right_gini) / total * gini_right
    uncertainty_after = entropy_left + entropy_right
    information_gain = current_uncertainty - uncertainty_after
    return information_gain
```

```python id="K5gMIQE6dnaE" executionInfo={"status": "ok", "timestamp": 1630132409672, "user_tz": -330, "elapsed": 758, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Fine the best split for a list of examples based on its features
def find_best_split(examples, number_of_features):
    best_gain = 0
    best_question = None
    current_uncertainty = calculate_gini(examples)
    for feature_index in range(number_of_features):
        values = set([example[feature_index] for example in examples])
        for value in values:
            question = Question(feature_index, value)
            examples_true, examples_false = split_examples(examples, question)
            if len(examples_true) != 0 or len(examples_false) != 0:
                gain = calculate_information_gain(examples_true, examples_false, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question
    return best_gain, best_question
```

```python id="v8XKU1wNdrSW" executionInfo={"status": "ok", "timestamp": 1630132420489, "user_tz": -330, "elapsed": 632, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Build the decision tree
def build_tree(examples):
    gain, question = find_best_split(examples, len(examples[0]) - 1)
    if gain == 0:
        return ExamplesNode(examples)
    print('Best question : ', question.to_string(), '\t', 'Info gain: ', "{0:.3f}".format(gain))
    examples_true, examples_false = split_examples(examples, question)
    branch_true = build_tree(examples_true)
    branch_false = build_tree(examples_false)
    return DecisionNode(question, branch_true, branch_false)
```

```python id="4iCL5RF1ds4N" executionInfo={"status": "ok", "timestamp": 1630132430704, "user_tz": -330, "elapsed": 1722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_tree(node, indentation=''):
    # The examples in the current ExamplesNode
    if isinstance(node, ExamplesNode):
        print(indentation + 'Examples', node.examples)
        return
    # The question for the current DecisionNode
    print(indentation + str(node.question.to_string()))
    # Find the 'True' examples for the current DecisionNode recursively
    print(indentation + '---> True:')
    print_tree(node.branch_true, indentation + '\t')
    # Find the 'False' examples for the current DecisionNode recursively
    print(indentation + '---> False:')
    print_tree(node.branch_false, indentation + '\t')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Vt0KFtV2baCY" executionInfo={"status": "ok", "timestamp": 1630132433039, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be272aa5-71c5-40ae-de3d-5a2e2525e447"
tree = build_tree(feature_examples)
print_tree(tree)
```

<!-- #region id="ofdt0CUlduQ3" -->
### Sklearn model on small sample
<!-- #endregion -->

```python id="SSiqtnhfeFs2" executionInfo={"status": "ok", "timestamp": 1630132528743, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import collections
from sklearn import tree
from sklearn import metrics
import pydotplus
```

```python colab={"base_uri": "https://localhost:8080/"} id="MmiWQMWxd_2w" executionInfo={"status": "ok", "timestamp": 1630132576232, "user_tz": -330, "elapsed": 576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="61136188-c8fa-482b-ae4c-11cb8fbf712d"
data_X = [[0.21, 327],   # 1
          [0.39, 497],   # 1
          [0.50, 1122],  # 2
          [0.76, 907],   # 1
          [0.87, 2757],  # 1
          [0.98, 2865],  # 1
          [1.13, 3045],  # 2
          [1.34, 3914],  # 2
          [1.67, 4849],  # 2
          [1.81, 5688]]  # 2

data_Y = ['1', '1', '2', '1', '1', '1', '2', '2', '2', '2']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_X, data_Y)

dot_data = tree.export_graphviz(clf,
                                feature_names=['carat', ['price']],
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('cyan', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
```

<!-- #region id="2_4RerMlgmcg" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 514} id="6uuPuRN_eRo4" executionInfo={"status": "ok", "timestamp": 1630132633698, "user_tz": -330, "elapsed": 2452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4952ec17-1651-4885-d5db-75687e4d4eb6"
from IPython.display import Image 
Image('tree.png')
```

<!-- #region id="Yxbx1yVufGeJ" -->
### Sklearn model on full load
<!-- #endregion -->

```python id="AKzX9LFCeKsF" executionInfo={"status": "ok", "timestamp": 1630132553044, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Data:
    """A data class to read and prepare data into training and testing data"""
    def __init__(self, data_file_name, excluded_features, label, encoded_categories):
        data_file = pd.read_csv(data_file_name)
        data_file.replace(encoded_categories, inplace=True)
        X = data_file.drop(columns=excluded_features)
        y = data_file[label]
        X = X.drop(columns=label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.5)

    def enumerate_categories(self, encoded_categories):
        self.X_train.replace(encoded_categories, inplace=True)
        self.X_test.replace(encoded_categories, inplace=True)
        self.y_train.replace(encoded_categories, inplace=True)
        self.y_test.replace(encoded_categories, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ykH3YTTEefAk" executionInfo={"status": "ok", "timestamp": 1630132819932, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb81aba3-3501-4732-931b-9cc907fc8f0f"
# Encode the string values for "cut", "color", and "clarity" as integer values
encoding_categories = {'cut': {'Fair': 1, 'Good': 1, 'Very Good': 2, 'Premium': 2, 'Ideal': 2},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}

# "no","carat","cut","color","clarity","depth","table","price","x size","y size","z size"
data = Data('./data/diamonds.csv', ['no', 'color', 'clarity', 'depth', 'table', 'x size', 'y size', 'z size'], 'cut', encoding_categories)
print(data.X_train.head())
# clf = ensemble.RandomForestClassifier()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data.X_train, data.y_train)
prediction = clf.predict(data.X_test)
print("Prediction Accuracy: ", metrics.accuracy_score(prediction, data.y_test))
```

```python id="_h2PI8ZsfNHs"

```
