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

<!-- #region id="EtbAbPPs7A-R" -->
# Concept - Data Mining Similarity Measures
> Understanding basic data science concepts - euclidean distance, pearson correlation, clustering, PCA dimension reduction, supervised learning

- toc: true
- badges: true
- comments: true
- categories: [Concept, PCA, Correlation, KMeans, GradientBoosting]
- image:
<!-- #endregion -->

<!-- #region id="HGucTQev62Tj" -->
### Euclidean Score
<!-- #endregion -->

```python id="chWZr-cI62Tj"
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

```python id="OQIHe3mm62Tk"
#Function to compute Euclidean Distance. 
def euclidean(v1, v2):
    
    #Convert 1-D Python lists to numpy vectors
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    #Compute vector which is the element wise square of the difference
    diff = np.power(np.array(v1)- np.array(v2), 2)
    
    #Perform summation of the elements of the above vector
    sigma_val = np.sum(diff)
    
    #Compute square root and return final Euclidean score
    euclid_score = np.sqrt(sigma_val)
    
    return euclid_score

```

```python id="CdTc0raX62Tl"
#Define 3 users with ratings for 5 movies
u1 = [5,1,2,4,5]
u2 = [1,5,4,2,1]
u3 = [5,2,2,4,4]
```

```python id="8CY-53ou62Tm" colab={"base_uri": "https://localhost:8080/"} outputId="59523801-3dc6-48b6-b0b9-7ebfbdc2ed97"
euclidean(u1, u2)
```

```python id="-EwRppu762Tp" colab={"base_uri": "https://localhost:8080/"} outputId="a23d64c2-ff71-4943-e912-4f26520c7a14"
euclidean(u1, u3)
```

<!-- #region id="Ntevj7uF62Tq" -->
### Pearson Correlation
<!-- #endregion -->

```python id="LPw8S-LZ62Tr" colab={"base_uri": "https://localhost:8080/"} outputId="e2a725c2-4d44-47a3-c27f-8d5ecbcfcb98"
alice = [1,1,3,2,4]
bob = [2,2,4,3,5]

euclidean(alice, bob)
```

```python id="_xYWuWtm62Ts" colab={"base_uri": "https://localhost:8080/"} outputId="69db5778-2544-4bf7-c351-dfefa10fe8db"
eve = [5,5,3,4,2]

euclidean(eve, alice)
```

```python id="PtpDguU162Ts" colab={"base_uri": "https://localhost:8080/"} outputId="99e709a3-6b7b-42a3-977d-493f4ec99f20"
from scipy.stats import pearsonr

pearsonr(alice, bob)
```

```python id="ai-Vi6e462Tt" colab={"base_uri": "https://localhost:8080/"} outputId="61ff2676-f85a-4be3-bf60-47f1fd211fe1"
pearsonr(alice, eve)
```

<!-- #region id="-9SoqphS62Tt" -->
## Clustering

### K-Means
<!-- #endregion -->

```python id="DFuV1mAt62Tu" colab={"base_uri": "https://localhost:8080/", "height": 319} outputId="91b8605c-8254-4c7e-ccc8-24a2ea2ff1f4"
#Import the function that enables us to plot clusters
from sklearn.datasets.samples_generator import make_blobs

#Get points such that they form 3 visually separable clusters
X, y = make_blobs(n_samples=300, centers=3,
                       cluster_std=0.50, random_state=0)


#Plot the points on a scatterplot
plt.scatter(X[:, 0], X[:, 1], s=50);
```

```python id="HTLFs46962Tv" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="9941d1e2-beec-4aa1-8b42-80b7acdd9667"
#Import the K-Means Class
from sklearn.cluster import KMeans

#Initializr the K-Means object. Set number of clusters to 3, 
#centroid initilalization as 'random' and maximum iterations to 10
kmeans = KMeans(n_clusters=3, init='random', max_iter=10)

#Compute the K-Means clustering 
kmeans.fit(X)

#Predict the classes for every point
y_pred = kmeans.predict(X)

#Plot the data points again but with different colors for different classes
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)

#Get the list of the final centroids
centroids = kmeans.cluster_centers_

#Plot the centroids onto the same scatterplot.
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')
```

```python id="zgTse-N962Tx" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="b1b09b91-b790-4f9c-ece2-c6c30cc001a8"
#List that will hold the sum of square values for different cluster sizes
ss = []

#We will compute SS for cluster sizes between 1 and 8.
for i in range(1,9):
    
    #Initlialize the KMeans object and call the fit method to compute clusters 
    kmeans = KMeans(n_clusters=i, random_state=0, max_iter=10, init='random').fit(X)
    
    #Append the value of SS for a particular iteration into the ss list
    ss.append(kmeans.inertia_)

#Plot the Elbow Plot of SS v/s K
sns.pointplot(x=[j for j in range(1,9)], y=ss)
```

```python id="2xbHVux562T0" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="d10b6898-3486-4102-d0be-6eb5fdad6fd4"
#Import the half moon function from scikit-learn
from sklearn.datasets import make_moons

#Get access to points using the make_moons function
X_m, y_m = make_moons(200, noise=.05, random_state=0)

#Plot the two half moon clusters
plt.scatter(X_m[:, 0], X_m[:, 1], s=50);
```

```python id="wSugIX6E62T2" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="c6eb084a-9a87-47e8-c526-82d61feb443b"
#Initialize K-Means Object with K=2 (for two half moons) and fit it to our data
kmm = KMeans(n_clusters=2, init='random', max_iter=10)
kmm.fit(X_m)

#Predict the classes for the data points
y_m_pred = kmm.predict(X_m)

#Plot the colored clusters as identified by K-Means
plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_pred, s=50)
```

```python id="nyLZ5ysK62T3" colab={"base_uri": "https://localhost:8080/", "height": 319} outputId="03067acf-8655-4aee-954a-a8a8ef3a956e"
#Import Spectral Clustering from scikit-learn
from sklearn.cluster import SpectralClustering

#Define the Spectral Clustering Model
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')

#Fit and predict the labels
y_m_sc = model.fit_predict(X_m)

#Plot the colored clusters as identified by Spectral Clustering
plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_sc, s=50);

```

<!-- #region id="h5ElJQLy62T3" -->
## Dimensionality Reduction

### Principal Component Analysis
<!-- #endregion -->

```python id="VpMn4IDl62T3" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="ea0a62ad-5372-4cc8-e4cc-593de8c131c7"
# Load the Iris dataset into Pandas DataFrame
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
                 names=['sepal_length','sepal_width','petal_length','petal_width','class'])

#Display the head of the dataframe
iris.head()
```

```python id="EqRZQsU162T4" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="d8feeb93-493e-4161-c099-63d488b24d67"
#Import Standard Scaler from scikit-learn
from sklearn.preprocessing import StandardScaler

#Separate the features and the class
X = iris.drop('class', axis=1)
y = iris['class']

# Scale the features of X
X = pd.DataFrame(StandardScaler().fit_transform(X), 
                 columns = ['sepal_length','sepal_width','petal_length','petal_width'])

X.head()
```

```python id="fzaNuGrz62T4" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="e747ae3d-3294-43b4-e958-b9eab6d4ac53"
#Import PCA
from sklearn.decomposition import PCA

#Intialize a PCA object to transform into the 2D Space.
pca = PCA(n_components=2)

#Apply PCA
pca_iris = pca.fit_transform(X)
pca_iris = pd.DataFrame(data = pca_iris, columns = ['PC1', 'PC2'])

pca_iris.head()
```

```python id="zGgjEsgr62T5" colab={"base_uri": "https://localhost:8080/"} outputId="11705cc1-6d1a-4211-ccd2-2902142f4446"
pca.explained_variance_ratio_
```

```python id="xxXEE47K62T6"
#Concatenate the class variable
pca_iris = pd.concat([pca_iris, y], axis = 1)
```

```python id="ushWodqA62T6" colab={"base_uri": "https://localhost:8080/", "height": 400} outputId="917e82b9-0db5-4b57-a8f1-5b8d56b40e12"
#Display the scatterplot
sns.lmplot(x='PC1', y='PC2', data=pca_iris, hue='class', fit_reg=False)
```

```python id="0ekDmRUT62T6" colab={"base_uri": "https://localhost:8080/", "height": 400} outputId="184e5db2-a5ea-4684-adaa-8ccb7a315bd7"
#Import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Define the LDA Object to have two components
lda = LinearDiscriminantAnalysis(n_components = 2)

#Apply LDA
lda_iris = lda.fit_transform(X, y)
lda_iris = pd.DataFrame(data = lda_iris, columns = ['C1', 'C2'])

#Concatenate the class variable
lda_iris = pd.concat([lda_iris, y], axis = 1)

#Display the scatterplot
sns.lmplot(x='C1', y='C2', data=lda_iris, hue='class', fit_reg=False)
```

<!-- #region id="71BHyzij62T7" -->
## Supervised Learning

### Gradient Boosting
<!-- #endregion -->

```python id="2CZwhQDa62T7" colab={"base_uri": "https://localhost:8080/"} outputId="ec35504e-b641-4633-f0af-c10e33680bce"
#Divide the dataset into the feature dataframe and the target class series.
X, y = iris.drop('class', axis=1), iris['class']

#Split the data into training and test datasets. 
#We will train on 75% of the data and assess our performance on 25% of the data

#Import the splitting funnction
from sklearn.model_selection import train_test_split

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Import the Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

#Apply Gradient Boosting to the training data
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

#Compute the accuracy on the test set
gbc.score(X_test, y_test)
```

```python id="TOVodJtf62T8" colab={"base_uri": "https://localhost:8080/", "height": 283} outputId="f7375db7-741d-4bc1-9931-b2b161faf0b4"
#Display a bar plot of feature importances
sns.barplot(x= ['sepal_length','sepal_width','petal_length','petal_width'], y=gbc.feature_importances_)
```
