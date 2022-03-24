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

<!-- #region id="kgJ-QEBHvYNq" -->
# Clustering data into groups using K-means and DBSCAN

*Clustering* is the process of organizing data points into conceptually meaningful groups. What makes a given group “conceptually meaningful”? There is no easy answer to that question. The usefulness of any clustered output is dependent on the task we’ve been assigned.

Imagine that we’re asked to cluster a collection of pet photos. Do we cluster fish and lizards in one group and fluffy pets (such as hamsters, cats, and dogs) in another? Or should hamsters, cats, and dogs be assigned three separate clusters of their own? If so, perhaps we should consider clustering pets by breed. Thus, Chihuahuas and Great Danes fall into diverging clusters. Differentiating between dog breeds will not be easy. However, we can easily distinguish between Chihuahuas and Great Danes based on breed size. Maybe we should compromise: we’ll cluster on both fluffiness and size, thus bypassing the distinction between the Cairn Terrier and the similar-looking Norwich Terrier.

Is the compromise worth it? It depends on our data science task. Suppose we work for a pet food company, and our aim is to estimate demand for dog food, cat food, and lizard food. Under these conditions, we must distinguish between fluffy dogs, fluffy cats, and scaly lizards. However, we won’t need to resolve differences between separate dog breeds. Alternatively, imagine an analyst at a vet’s office who’s trying to group pet patients by their breed. This second task requires a much more granular level of group resolution.

Different situations require different clustering techniques. As data scientists, we must choose the correct clustering solution. Over the course of our careers, we will cluster thousands (if not tens of thousands) of datasets using a variety of clustering techniques. The most commonly used algorithms rely on some notion of centrality to distinguish between clusters.
<!-- #endregion -->

<!-- #region id="mktX7lww1Nb6" -->
## Using Centrality to Discover Clusters

Suppose a bull’s-eye is located at a coordinate of `[0, 0]`. A dart is thrown at that coordinate. We’ll model the x and y positions of the dart using 2 Normal distributions.

**Modeling dart coordinates using 2 Normal distributions**
<!-- #endregion -->

```python id="P_R7gS7D1NcE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637419985265, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fff04d3f-6336-4d80-a66f-ae5fcf009911"
import numpy as np
np.random.seed(0)
mean = 0
variance = 2
x = np.random.normal(mean, variance ** 0.5)
y = np.random.normal(mean, variance ** 0.5)
print(f"The x coordinate of a randomly thrown dart is {x:.2f}")
print(f"The y coordinate of a randomly thrown dart is {y:.2f}")
```

<!-- #region id="hMWdIhev1NcK" -->
Lets simulate 5,000 random darts tossed at the bulls'-eye positioned at `[0, 0]`. We'll also simulate 5,000 random darts tossed at a second bull's-eye, positioned at `[0, 6]`. Afterwards, we'll generate a scatter plot of all the random dart coordinates.

**Simulating randomly thrown darts**
<!-- #endregion -->

```python id="6uqFaqb-1NcM" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420018215, "user_tz": -330, "elapsed": 922, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f18d50a-3a4a-405d-bed9-ef2101056ac8"
import matplotlib.pyplot as plt
np.random.seed(1)
bulls_eye1 = [0, 0]
bulls_eye2 = [6, 0]
bulls_eyes = [bulls_eye1, bulls_eye2]
x_coordinates, y_coordinates = [], []
for bulls_eye in bulls_eyes:
    for _ in range(5000):
        x = np.random.normal(bulls_eye[0], variance ** 0.5)
        y = np.random.normal(bulls_eye[1], variance ** 0.5)
        x_coordinates.append(x)
        y_coordinates.append(y)
        
plt.scatter(x_coordinates, y_coordinates)
plt.show()
```

<!-- #region id="SN3h75Dg1NcN" -->
Lets assign each dart to its nearest bull's-eye. We'll measure dart-proximity using **Euclidean distance**, which is the standard straight-line distance between 2 points. 

**Assigning darts to the nearest bull’s-eye**
<!-- #endregion -->

```python id="WPprhNrB1NcO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420041519, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="33066d4b-6302-4fd3-bebf-7bc95ecfefa2"
from scipy.spatial.distance import euclidean
def nearest_bulls_eye(dart):
    distances = [euclidean(dart, bulls_e) for bulls_e in bulls_eyes]
    return np.argmin(distances)

darts = [[0,1], [6, 1]]
for dart in darts:
    index = nearest_bulls_eye(dart)
    print(f"The dart at position {dart} is closest to bulls-eye {index}")
```

<!-- #region id="yCgwN04d1NcP" -->
Now, we will apply the `nearest_bulls_eye` function to all our computed color coordinates. Afterwards, each dart-point will be plotted using one of 2 colors, in order to distinguish between the 2 bull's-eye assignments.

**Coloring darts based on nearest bull’s-eye**
<!-- #endregion -->

```python id="q3YNtlL21NcR" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420084391, "user_tz": -330, "elapsed": 1469, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6719928c-9e31-4345-86c9-652afc2b51c3"
def color_by_cluster(darts):
    nearest_bulls_eyes = [nearest_bulls_eye(dart) for dart in darts]
    for bs_index in range(len(bulls_eyes)):
        selected_darts = [darts[i] for i in range(len(darts))
                          if bs_index == nearest_bulls_eyes[i]]
        x_coordinates, y_coordinates = np.array(selected_darts).T
        plt.scatter(x_coordinates, y_coordinates, 
                    color=['g', 'k'][bs_index])
    plt.show()

darts = [[x_coordinates[i], y_coordinates[i]]  
         for i in range(len(x_coordinates))]
color_by_cluster(darts)
```

<!-- #region id="WDJ5JPU51NcU" -->
The colored darts sensibly split into 2 even clusters. How would we identify such clusters if no central coordinates were provided? Well, one primitive strategy is to simply guess the location of the bull's-eyes.

**Assigning darts to randomly chosen centers**
<!-- #endregion -->

```python id="Z2y3p7NM1NcX" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420103532, "user_tz": -330, "elapsed": 710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3b63e4bc-2d3f-4026-fd6d-02312d6901a8"
bulls_eyes = np.array(darts[:2])
color_by_cluster(darts)
```

<!-- #region id="m--Zx8nj1NcZ" -->
Cluster B on the right seems to be stretching way too far to the left.  Lets remedy our error. We 'll compute the mean coordinates of all the points within the stretched right clustered group, and afterwards utilize these coordinates to adjust our estimation of the group's center. We will also reset the left-most cluster's center to its mean prior to re-running our centrality-based clustering.

**Assigning darts to centers based on mean**
<!-- #endregion -->

```python id="FX7vcu8Y1Nca" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420136031, "user_tz": -330, "elapsed": 1632, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="970a5d0f-425e-4bf8-d2fb-3f26bb4a3e7c"
def update_bulls_eyes(darts):
    updated_bulls_eyes = []
    nearest_bulls_eyes = [nearest_bulls_eye(dart) for dart in darts]
    for bs_index in range(len(bulls_eyes)):
        selected_darts = [darts[i] for i in range(len(darts))
                          if bs_index == nearest_bulls_eyes[i]]
        x_coordinates, y_coordinates = np.array(selected_darts).T
        mean_center = [np.mean(x_coordinates), np.mean(y_coordinates)]
        updated_bulls_eyes.append(mean_center)
        
    return updated_bulls_eyes

bulls_eyes = update_bulls_eyes(darts)
color_by_cluster(darts)
```

<!-- #region id="OuYIalle1Nce" -->
The cluster's centers still appear a little off. Lets remedy the results by repeating the mean-based centrality adjustment over 10 additional iterations.

**Adjusting bull’s-eye positions over 10 iterations**
<!-- #endregion -->

```python id="5pMLXI4X1Ncf" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420169187, "user_tz": -330, "elapsed": 7196, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35f9e2c4-4814-424c-f4cb-e7c8aa25e597"
for i in range(10):
    bulls_eyes = update_bulls_eyes(darts)
    
color_by_cluster(darts)
```

<!-- #region id="9XZhWJ-l1Nch" -->
Now the 2 sets of darts have been perfectly clustered. We have essentially replicated the  **K-means** clustering algorithm, which organizes data using centrality.

## K-Means: A Clustering Algorithm for Grouping Data into K Central Groups

### 10.2.1. K-means Clustering Using Scikit-learn

A speedy implementation of the K-means algorithm is available through the external Scikit-Learn library. Lets import Scikit-learn's `KMeans` clustering class.
<!-- #endregion -->

```python id="HkzEy4SQ1Ncj"
from sklearn.cluster import KMeans
```

<!-- #region id="neaXBMRb1Nck" -->
Now, we'll use the `KMeans` class to cluster our `darts` data.

**K-means clustering using Scikit-learn**
<!-- #endregion -->

```python id="bGwXgO_-1Ncl" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420209456, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5f717cc-0dc5-43bf-ee57-d0b9bdc5356b"
cluster_model = KMeans(n_clusters=2)
assigned_bulls_eyes = cluster_model.fit_predict(darts)

print("Bull's-eye assignments:")
print(assigned_bulls_eyes)
```

<!-- #region id="h0z-pMG11Ncn" -->
Lets quickly color our darts based on their clustering assignments, in order to confirm that the assignments makes sense.

**Plotting K-means cluster assignments**
<!-- #endregion -->

```python id="TCiT_J_h1Nco" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420240188, "user_tz": -330, "elapsed": 925, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="58ec4d53-3995-4d94-aad5-56c4a7552466"
for bs_index in range(len(bulls_eyes)):
    selected_darts = [darts[i] for i in range(len(darts))
                      if bs_index == assigned_bulls_eyes[i]]
    x_coordinates, y_coordinates = np.array(selected_darts).T
    plt.scatter(x_coordinates, y_coordinates, 
                color=['g', 'k'][bs_index])
plt.show()
```

<!-- #region id="rYXCAqPg1Ncr" -->
Our clustering model has located the centroids in the data. Now, we can reuse these centroids to analyze new data-points that the model has not seen before.

**Using `cluster_model` to cluster new data**
<!-- #endregion -->

```python id="DbrnNwNo1Ncs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420252818, "user_tz": -330, "elapsed": 807, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="be9b46d4-4339-4642-bf83-21140504cb82"
new_darts = [[500, 500], [-500, -500]]
new_bulls_eye_assignments = cluster_model.predict(new_darts)
for i, dart in enumerate(new_darts):
    bulls_eye_index = new_bulls_eye_assignments[i]
    print(f"Dart at {dart} is closest to bull's-eye {bulls_eye_index}")
```

<!-- #region id="aLVVa54U1Nct" -->
### Selecting the Optimal K Using the Elbow Method

We estimate an appropriate value for K using a technique known as the **Elbow method**. The Elbow method depends on a calculated value called **inertia**, which is the sum of the squared distances between each point and its closest K-means center. We'll run the technique by plotting the inertia of our dartboard dataset over a large range of K values.

**Plotting the K-means inertia**
<!-- #endregion -->

```python id="kwC9Ir_T1Ncv" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637420271197, "user_tz": -330, "elapsed": 7003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1cad7fc-cbba-476f-d824-ff7ef02b0629"
k_values = range(1, 10)
inertia_values = [KMeans(k).fit(darts).inertia_
                  for k in k_values]

plt.plot(k_values, inertia_values)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()
```

<!-- #region id="2EpBpDDQ1Ncw" -->
The generated plot resembles an arm bent at the elbow. The elbow points directly to a K of 2. What will happen if the number of centers is increased? We can find out by adding an additional bull's-eye to our dart-throwing simulation.

**Plotting inertia for a 3-dartboard simulation**
<!-- #endregion -->

```python id="IvfQk3N61Ncx" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637420279876, "user_tz": -330, "elapsed": 8687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9d1bc931-32b0-4cd4-b209-7e1c60b3ebe7"
new_bulls_eye = [12, 0]
for _ in range(5000):
    x = np.random.normal(new_bulls_eye[0], variance ** 0.5)
    y = np.random.normal(new_bulls_eye[1], variance ** 0.5)
    darts.append([x, y])

inertia_values = [KMeans(k).fit(darts).inertia_
                  for k in k_values]

plt.plot(k_values, inertia_values)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()
```

<!-- #region id="dKycbxyc1Nc0" -->
## Using Density to Discover Clusters

Suppose that an astronomer discovers a new planet at the far-flung edges of the solar system. The plant, much like our Saturn, has multiple rings spinning in constant orbit around its center. Each ring is formed from thousands of rocks. We'll model these rocks as individual points, defined by x and y coordinates.

**Simulating rings around a planet**
<!-- #endregion -->

```python id="fO6YKiXT1Nc1" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420288617, "user_tz": -330, "elapsed": 754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa9ebdd3-41a3-4847-8e2d-e8e5603d199c"
from sklearn.datasets import make_circles

x_coordinates = []
y_coordinates = []
for factor in [.3, .6, 0.99]:
    rock_ring, _ = make_circles(n_samples=800, factor=factor,
                                noise=.03, random_state=1)
    for rock in rock_ring:
        x_coordinates.append(rock[0])
        y_coordinates.append(rock[1])

plt.scatter(x_coordinates, y_coordinates)
plt.show()
```

<!-- #region id="b2OHs04T1Nc2" -->
Three ring-groups are clearly present in the plot. Lets search for these 3 clusters using K-means.

**Using K-means to cluster rings**
<!-- #endregion -->

```python id="XlGbACNk1Nc3" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420308900, "user_tz": -330, "elapsed": 1572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22135779-332e-4013-e5ef-3f7505bd27ce"
rocks = [[x_coordinates[i], y_coordinates[i]]  
          for i in range(len(x_coordinates))]
rock_clusters = KMeans(3).fit_predict(rocks)

colors = [['g', 'y', 'k'][cluster] for cluster in  rock_clusters]
plt.scatter(x_coordinates, y_coordinates, color=colors)
plt.show()
```

<!-- #region id="fDIHU7fz1Nc5" -->
The output is an utter failure! We need to design an algorithm that will cluster data within dense regions of space. One simple definition of density is as follows; a point is in a dense region only if it's located within a distance `epsilon` of `min_points` other points.  Below, we'll set `epsilon` to 0.1 and `min_points` to 10.

**Specifying density parameters**
<!-- #endregion -->

```python id="pWRjoAWr1Nc8"
epsilon=0.1
min_points = 10
```

<!-- #region id="TzH2UFDB1Nc-" -->
Lets analyze the density of the first rock in our `rocks` list. We'll begin by searching for all the other rocks that are within `epsilon` units of `rocks[0]`.

**Finding the neighbors of `rocks[0]`**
<!-- #endregion -->

```python id="8oS3wQkP1Nc_"
neighbor_indices = [i for i, rock in enumerate(rocks[1:])
                    if euclidean(rocks[0], rock) <= epsilon]
```

<!-- #region id="uq29yqOs1NdB" -->
Now, we'll compare the number of neighbors to `min_points`, in order to determine if `rocks[0]` lies in a dense region of space.

**Checking the density of `rocks[0]`**
<!-- #endregion -->

```python id="LirEtIfB1NdC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420311974, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="196571cc-bdeb-4b51-a412-9bf29ff513b9"
num_neighbors = len(neighbor_indices)
print(f"The rock at index 0 has {num_neighbors} neighbors.")

if num_neighbors >= min_points:
    print("It lies in a dense region.")
else:
    print("It does not lie in a dense region.")
```

<!-- #region id="2gdt53KM1NdF" -->
The rock at index 0 lies in a dense region of space. We can combine `rocks[0]` and its neighbors into a single dense cluster.

**Creating a dense cluster**
<!-- #endregion -->

```python id="b3iiwxXn1NdG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420312784, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="20e0eede-d882-4a11-aaa1-a29cdcd2f105"
dense_region_indices = [0] + neighbor_indices
dense_region_cluster = [rocks[i] for i in dense_region_indices]
dense_cluster_size = len(dense_region_cluster)
print(f"We found a dense cluster containing {dense_cluster_size} rocks")
```

<!-- #region id="cZldMuJg1NdK" -->
The rock and index 0 and its neighbors form a single 41-element dense cluster. What about the neighbors of the neighbors? By analyzing additional neighboring points, we expand the size of `dense_region_cluster`.

**Expanding a dense cluster**
<!-- #endregion -->

```python id="iQOAE3ol1NdM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420315827, "user_tz": -330, "elapsed": 2272, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="afaa9696-45f2-4dff-e716-b852db66ffd3"
dense_region_indices = set(dense_region_indices)
for index in neighbor_indices:
    point = rocks[index]
    neighbors_of_neighbors = [i for i, rock in enumerate(rocks)
                              if euclidean(point, rock) <= epsilon]
    if len(neighbors_of_neighbors) >= min_points:
        dense_region_indices.update(neighbors_of_neighbors)
            
dense_region_cluster = [rocks[i] for i in dense_region_indices]
dense_cluster_size = len(dense_region_cluster)
print(f"We expanded our cluster to include {dense_cluster_size} rocks")
```

<!-- #region id="oMkN86R21NdO" -->
We can expand our cluster even further by analyzing the density of newly encountered neighbors. Iteratively repeating our analysis will increase the breadth of our cluster boundary. This precedure is known as **DBSCAN**. The DBSCAN algorithm organizes data based on its density distribution.

## DBSCAN: A Clustering Algorithm for Grouping Data Based on Spatial Density

Scikit-Learn makes DBSCAN available for use. We simply need to import the `DBSCAN` class from `sklearn.cluster`. Afterwards, we can initialize the class by assigning `epsilon` and `min_points` using the `eps` and `min_samples` parameters. Lets utilize `DBSCAN` to cluster our 3 rings.

**Using `DBSCAN` to cluster rings**
<!-- #endregion -->

```python id="0vMALMND1NdP" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420323212, "user_tz": -330, "elapsed": 1159, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cca2a896-037d-4d06-b6a0-93e883608ca1"
from sklearn.cluster import DBSCAN
cluster_model = DBSCAN(eps=epsilon, min_samples=min_points)
rock_clusters = cluster_model.fit_predict(rocks)
colors = [['g', 'y', 'k'][cluster] for cluster in rock_clusters]
plt.scatter(x_coordinates, y_coordinates, color=colors)
plt.show()
```

<!-- #region id="_C9EqEZ21NdQ" -->
DBSCAN has successfully identified the 3 rock rings. The algorithm succeeded where K-means had failed.

### Comparing DBSCAN and K-means

DBSCAN can filter random outliers located in sparse regions of space. For example, if we add an outlier located beyond the boundary of the rings, then DBSCAN will assign it a cluster id of -1. The negative value indicates that the outlier cannot be clustered with the rest of the dataset.

**Finding outliers using DBSCAN**
<!-- #endregion -->

```python id="RJX2Juq91NdR"
noisy_data = rocks + [[1000, -1000]]
clusters = DBSCAN(eps=epsilon, 
                  min_samples=min_points).fit_predict(noisy_data)
assert clusters[-1] == -1
```

<!-- #region id="OZnuNC721NdS" -->
There is one other advantage to the DBSCAN technique that is missing from K-means. DBSCAN does not depend on Euclidean distance.

### Clustering Based on Non-Euclidean Distance

Suppose we are visiting Manhattan. We wish to know the walking distance from the Empire State Building to Columbus Circle. The Empire State Building is located at the intersection of 34th street and 5th avenue. Meanwhile, Columbus Circle is located is located at the intersection of 57th street and 8th avenue. Our route requires us to walk 26 blocks total. Manhattan's average block-length is .17 miles. Lets compute that walking distance directly using a generalized `manhattan_distance` function.

**Computing the Manhattan distance**
<!-- #endregion -->

```python id="F1hBqh011NdU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420334553, "user_tz": -330, "elapsed": 760, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5976bd91-d531-4032-e8f1-7e4375669787"
def manhattan_distance(point_a, point_b):
    num_blocks = np.sum(np.absolute(point_a - point_b))
    return .17 * num_blocks

x = np.array([34, 5])
y = np.array([57, 8])
distance = manhattan_distance(x, y)

print(f"Manhattan distance is {distance} miles")
```

<!-- #region id="CXKtwNZ21NdW" -->
Now, suppose we wish to cluster more than 2 Manhattan locations, using DBSCAN. We will pass `metric= manhattan_distance` into the initialization method. Consequently, the clustering distance will correctly reflect the grid-based constraints within the City.

**Clustering using Manhattan distance**
<!-- #endregion -->

```python id="RhydSyR81NdX" colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"status": "ok", "timestamp": 1637420337214, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="12d4b876-c009-4ba0-f63e-1ea6a3d08015"
points = [[35, 5], [33, 6], [37, 4], [40, 7], [45, 5]]
clusters = DBSCAN(eps=1, min_samples=3,
                  metric=manhattan_distance).fit_predict(points)

for i, cluster in enumerate(clusters):
    point = points[i]
    if cluster == -1:
        print(f"Point at index {i} is an outlier")
        plt.scatter(point[0], point[1], marker='x', color='k')
    else:
        print(f"Point at index {i} is in cluster {cluster}")
        plt.scatter(point[0], point[1], color='g')

plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()

plt.show()
```

<!-- #region id="z8FPFP6z1NdZ" -->
Unlike K-means, the DBSCAN algorithm does not require our distance function to be linearly divisible. Thus, we can easily run DBSCAN clustering using a ridiculous distance metric.

**Clustering using a ridiculous measure of distance**
<!-- #endregion -->

```python id="CgPkZhVd1Ndd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637420343300, "user_tz": -330, "elapsed": 1038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5b6e22c-4ee3-471d-fe97-0e67bc998ba5"
def ridiculous_measure(point_a, point_b):
    is_negative_a = np.array(point_a) < 0
    is_negative_b = np.array(point_b) < 0
    if is_negative_a.all() and is_negative_b.all():
        return 0
    elif is_negative_a.any() and is_negative_b.any():
        return 10
    else:
        return 2

points = [[-1, -1], [-10, -10], [-1000, -13435], [3,5], [5,-7]]
                   
clusters = DBSCAN(eps=.1, min_samples=2,
                  metric=ridiculous_measure).fit_predict(points)

for i, cluster in enumerate(clusters):
    point = points[i]
    if cluster == -1:
        print(f"{point} is an outlier")
    else:
        print(f"{point} falls in cluster {cluster}")
```

<!-- #region id="VkGJe5uX1Ndf" -->
## Analyzing Clusters Using Pandas

We can more intuitively analyze clustered rocks by combining the coordinates and the clusters together in a single Pandas table.

**Storing clustered coordinates in a table**
<!-- #endregion -->

```python id="Us66VK-p1Ndg"
import pandas as pd
x_coordinates, y_coordinates = np.array(rocks).T
df = pd.DataFrame({'X': x_coordinates, 'Y': y_coordinates,
                   'Cluster': rock_clusters})
```

<!-- #region id="sXtnwQRQ1Ndh" -->
Our Pandas table lets us easily access the rocks in any cluster. Lets plot those rocks that fall into cluster zero.

**Plotting a single cluster using Pandas**
<!-- #endregion -->

```python id="4dND2qUy1Ndj" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637420349651, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="55f99cbf-f1d2-4f44-e87c-a7edb9a48e3c"
df_cluster = df[df.Cluster == 0]
plt.scatter(df_cluster.X, df_cluster.Y)
plt.show()
```

<!-- #region id="I6X89py11Ndl" -->
Pandas allows us to obtain a table containing elements from any single cluster. Alternatively, we might want to obtain multiple tables, where each table maps to a cluster id. In Pandas, this can easily be done by calling `df.groupby('Cluster')`.

**Iterating over clusters using Pandas**
<!-- #endregion -->

```python id="n2sAWtMH1Ndm" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1637420352095, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f312fce7-142e-4fac-b2cb-0f4b8d47890c"
for cluster_id, df_cluster in df.groupby('Cluster'):
    if cluster_id == 0:
        print(f"Skipping over cluster {cluster_id}")
        continue
    
    print(f"Plotting cluster {cluster_id}")
    plt.scatter(df_cluster.X, df_cluster.Y)

plt.show()
```
