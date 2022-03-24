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

<!-- #region id="TPlgB0id0uCp" -->
# Tracking Disease Outbreaks using News Headlines

Congratulations! You have just been hired by the American Institute of Health. The Institute monitors disease epidemics in both foreign and domestic lands. A critical component of the monitoring process is analyzing published news data. Each day, the Institute receives hundreds of news headlines describing disease outbreaks in various locations. The news headlines are too numerous to be analyzed by hand.

Your first assignment is as follows: You will process the daily quota of news headlines and extract locations that are mentioned You will then cluster the headlines based on their geographic distribution. Finally, you will review the largest clusters within and outside the United States. Any interesting findings should be reported to your immediate superior.

The file `headlines.txt` contains the hundreds of headlines that you must analyze. Each headline appears on a separate line in the file.

<aside>
📌 To address the problem at hand, we need to know how to do the following: 1) Cluster datasets using multiple techniques and distance measures. 2) Measure distances between locations on a spherical globe. 3) Visualize locations on a map. 4) Extract location coordinates from headline text.

</aside>

Our goal is to extract locations from disease-related headlines to uncover the largest active epidemics within and outside of the United States. We will do as follows:

1. Load the data.
2. Extract locations from the text using regular expressions and the GeoNamesCache library.
3. Check the location matches for errors.
4. Cluster the locations based on geographic distance.
5. Visualize the clusters on a map, and remove any errors.
6. Output representative locations from the largest clusters to draw interesting conclusions.
<!-- #endregion -->

```python id="mIo7ferq-G-J"
!sudo apt-get install libgeos-3.5.0
!sudo apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip

!pip install geonamescache
!pip install unidecode
```

```python colab={"base_uri": "https://localhost:8080/"} id="6iTippHE9oIF" executionInfo={"status": "ok", "timestamp": 1637422147096, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7f223bb4-3e5d-4b7e-b2dd-aee0910f3efe"
!wget -q --show-progress https://raw.githubusercontent.com/sparsh-ai/general-recsys/T426474/siteD/Case_Study3/headlines.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="7b4K8vki9yAT" executionInfo={"status": "ok", "timestamp": 1637422154422, "user_tz": -330, "elapsed": 648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c3b9469c-6f0f-4a71-9adc-99f3cc4887f5"
!head headlines.txt
```

<!-- #region id="WPwfkV6q1NfQ" -->
## Extracting Locations from Headline Data

We'll begin by loading the headline data.

**Listing 12. 1. Loading headline data**
<!-- #endregion -->

```python id="dLPU4qqK1NfR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422538382, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a4c22c0-073b-4587-b651-7d33a84e3ee2"
headline_file = open('headlines.txt','r')
headlines = [line.strip() 
             for line in headline_file.readlines()]
num_headlines = len(headlines)
print(f"{num_headlines} headines have been loaded")
```

<!-- #region id="MiDhA3ro1NfS" -->
We need a mechanism for extracting city and country names from the headline text. One naïve solution is to match the locations in GeoNamesCache against each and every headline.  However, for more optimal matching, we should transform each location name into a case-independent and accent-independent regular expression. Lets execute these transformations using a custom `name_to_regex` function. 

**Listing 12. 2. Converting names to regexes**
<!-- #endregion -->

```python id="gKqzRye01NfS"
def name_to_regex(name):
    decoded_name = unidecode(name)
    if name != decoded_name:
        regex = fr'\b({name}|{decoded_name})\b'
    else:
        regex = fr'\b{name}\b'
    return re.compile(regex, flags=re.IGNORECASE)
```

<!-- #region id="M7rM5ewb1NfT" -->
Using `name_to_regex`, we can create create a mapping between regular expressions and the original names in GeoNamesCache.

**Listing 12. 3. Mapping names to regex**
<!-- #endregion -->

```python id="F0p5G0L0_JNz"
import re
from unidecode import unidecode
from geonamescache import GeonamesCache
gc = GeonamesCache()
```

```python id="j23YOlIv1NfX"
countries = [country['name'] 
             for country in gc.get_countries().values()]
country_to_name = {name_to_regex(name): name 
                   for name in countries}
                   
cities = [city['name'] for city in gc.get_cities().values()]
city_to_name = {name_to_regex(name): name for name in cities}
```

<!-- #region id="8kv0K3eY1NfY" -->
Next, we’ll use our mappings to define a function that will look for location names in text

**Listing 12. 4. Finding locations in text**
<!-- #endregion -->

```python id="PxS2fy_i1NfZ"
def get_name_in_text(text, dictionary):
    for regex, name in sorted(dictionary.items(), 
                              key=lambda x: x[1]):
        if regex.search(text):
            return name
    return None
```

<!-- #region id="lRw1Lkrg1Nfa" -->
We'll utilize `get_names_in_text` to discover the cities and countries that are mentioned in the `headlines` list. Afterwards, we'll store the results in a Pandas table for easier analysis.

**Listing 12. 5. Finding locations in headlines**
<!-- #endregion -->

```python id="ZgeBO1_d1Nfa"
import pandas as pd

matched_countries = [get_name_in_text(headline, country_to_name)
                     for headline in headlines]
matched_cities = [get_name_in_text(headline, city_to_name)
                  for headline in headlines]
data = {'Headline': headlines, 'City': matched_cities, 
        'Country': matched_countries}
df = pd.DataFrame(data)
```

<!-- #region id="44iKBz6K1Nfb" -->
Lets explore our location table. We'll start by summarizing the contents of `df` using the `describe` method.

**Listing 12. 6. Summarizing the location data**
<!-- #endregion -->

```python id="VVZkjwWS1Nfb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422610755, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b23ccc34-c9ab-4806-9d1a-f133caa22bc1"
summary = df[['City', 'Country']].describe()
print(summary)
```

<!-- #region id="vUVFT7aW1Nfc" -->
The most frequently mentioned city is apparently __Of, Turkey__. The 44 instances of __Of__ are more likely
to match the preposition than the rarely referenced Turkish location. We will output some instances of __Of__ in order to confirm the error.

**Listing 12. 7. Fetching cities named __Of__**
<!-- #endregion -->

```python id="41VkcqtQ1Nfc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422681548, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11e073dc-f242-4304-9391-806d315fec75"
of_cities = df[df.City == 'Of'][['City', 'Headline']]
ten_of_cities = of_cities.head(10)
print(ten_of_cities.to_string(index=False))
```

<!-- #region id="b3sp5bwA1Nfd" -->
In all the wrongly matched headlines we matched to __Of__ but not to the actual city
name. The mismatches occurred because we didn't consider potential multiple matches in a headline. How frequently do headlines contain 2 or more city matches? Lets find out. 

**Listing 12. 8. Finding multi-city headlines**
<!-- #endregion -->

```python id="7A8M2JlP1Nfd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422719114, "user_tz": -330, "elapsed": 19300, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e04e5628-5d3c-4e8b-eb09-e92244ce9761"
def get_cities_in_headline(headline):
    cities_in_headline = set()
    for regex, name in city_to_name.items():          
        match = regex.search(headline)
        if match:
            if headline[match.start()].isupper():
                cities_in_headline.add(name)
                
    return list(cities_in_headline)

df['Cities'] = df['Headline'].apply(get_cities_in_headline)
df['Num_cities'] = df['Cities'].apply(len)
df_multiple_cities = df[df.Num_cities > 1]
num_rows, _ = df_multiple_cities.shape
print(f"{num_rows} headlines match multiple cities")
```

<!-- #region id="YHHegj0v1Nfe" -->
71 headlines contain more than one city, representing approximately 10% of the data. Why are so many headlines matching against multiple locations? Perhaps exploring some sample matches will yield an answer.

**Listing 12. 9. Sampling multi-city headlines**
<!-- #endregion -->

```python id="x4oakuO31Nfe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422727553, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7e91b4bf-a3b1-476a-d583-3e82511944c7"
ten_cities = df_multiple_cities[['Cities', 'Headline']].head(10)
print(ten_cities.to_string(index=False))
```

<!-- #region id="PQKGpXtk1Nff" -->
Short, invalid city names are getting matched to the headlines along with longer, more correct location names. One solution is simply to assign the longest city-name as the representative location if more than one matched city is found.

**Listing 12. 10. Selecting the longest city names**
<!-- #endregion -->

```python id="Qsxjz31F1Nfg"
def get_longest_city(cities):
    if cities:
        return max(cities, key=len)
    return None

df['City'] = df['Cities'].apply(get_longest_city)
```

<!-- #region id="LaLYa73T1Nfh" -->
As a sanity check, we'll output those rows in the the table that contain a short city-name (4 characters or less), in order to ensure that no erroneous short name is getting assigned to one of our headlines.

**Listing 12. 11. Printing the shortest city names**
<!-- #endregion -->

```python id="al_XgrH51Nfh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422768444, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="224ad44e-c77d-42ec-bb5d-172e978b9d37"
short_cities = df[df.City.str.len() <= 4][['City', 'Headline']]
print(short_cities.to_string(index=False))
```

<!-- #region id="tXaz76tP1Nfi" -->
Let's now shift our attention from cities to countries. Only 15 of the total headlines contain actual country information. The count is low enough for us to manually examine all these headlines.

**Listing 12. 12. Fetching headlines with countries**
<!-- #endregion -->

```python id="ba_J-EXK1Nfi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422804665, "user_tz": -330, "elapsed": 639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="870d9723-1b77-40f4-c212-e45034219fde"
df_countries = df[df.Country.notnull()][['City', 
                                         'Country', 
                                         'Headline']]
print(df_countries.to_string(index=False))
```

<!-- #region id="xFXqMNLh1Nfj" -->
All of the country-bearing headlines also contain city information. Thus, we can assign a latitude and longitude without relying on the country's central coordinates. Consequently, we can disregard the country names from our analysis.

**Listing 12. 13. Dropping countries from the table**
<!-- #endregion -->

```python id="S01faG-21Nfj"
df.drop('Country', axis=1, inplace=True)
```

<!-- #region id="znxNQi1b1Nfk" -->
We are nearly ready to add latitudes and longitudes to our table. However, we first need to consider those rows where no locations were detected. Lets count the number of unmatched headlines, and then print a subset of that data.

**Listing 12. 14. Exploring unmatched headlines**
<!-- #endregion -->

```python id="TYcyQzaA1Nfl" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637422817474, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e7193c5-ba1a-4c6c-c716-def533f10656"
df_unmatched = df[df.City.isnull()]
num_unmatched = len(df_unmatched)
print(f"{num_unmatched} headlines contain no city matches.")
print(df_unmatched.head(10)[['Headline']].values)
```

<!-- #region id="Q5oW1yo21Nfl" -->
Approximately 6% of the headlines did not match any cities. Given that low frequency, we will delete the missing mentions.

**Listing 12. 15. Dropping unmatched headlines**
<!-- #endregion -->

```python id="qYz1_H_a1Nfn"
df = df[~df.City.isnull()][['City', 'Headline']] 
```

<!-- #region id="ycdArbHC1Nfn" -->
## Visualizing and Clustering the Extracted Location Data

All the rows in our table contain a city-name. Now, we can assign a latitude and longitude to each row.

**Listing 12. 16. Assigning geographic coordinates to cities**
<!-- #endregion -->

```python id="cDiKn0q31Nfo"
latitudes, longitudes = [], []
for city_name in df.City.values:
    city = max(gc.get_cities_by_name(city_name), 
              key=lambda x: list(x.values())[0]['population'])
    city = list(city.values())[0]
    latitudes.append(city['latitude']) 
    longitudes.append(city['longitude'])

df = df.assign(Latitude=latitudes, Longitude=longitudes)
```

<!-- #region id="kyFvcjI-1Nfq" -->
Lets execute K-means across our set of 2D coordinates. We'll use the Elbow method to choose a reasonable value for K.

**Listing 12. 17. Plotting a geographic elbow curve**
<!-- #endregion -->

```python id="eYgqVWgh1Nfr" colab={"base_uri": "https://localhost:8080/", "height": 290} executionInfo={"status": "ok", "timestamp": 1637422922707, "user_tz": -330, "elapsed": 690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9cc8179d-d7de-43a1-b3fa-46a3e2265827"
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

coordinates = df[['Latitude', 'Longitude']].values
k_values = range(1, 10)
inertia_values = []
for k in k_values:
    inertia_values.append(KMeans(k).fit(coordinates).inertia_)

plt.plot(range(1, 10), inertia_values)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()
```

<!-- #region id="nTy_pgMS1Nfs" -->
The "elbow" within our Elbow plot points to a K of 3 That K-value is very low; limiting our scope to at-most 3 different geographic territories.

**Listing 12. 18. Using K-means to cluster cities into 3 groups**
<!-- #endregion -->

```python id="6EO2TKglA4l0"
from mpl_toolkits.basemap import Basemap
```

```python id="v4IMq8aQ1Nft" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637422965546, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eaf5b02f-ddf6-447e-ae92-9118aaaa0757"
def plot_clusters(clusters, longitudes, latitudes):
    fig = plt.figure(figsize=(12, 10))

    map_plotter = Basemap()
    map_plotter.scatter(longitudes, latitudes, c=clusters, latlon=True,
                        marker='o', alpha=1.0)
    map_plotter.drawcoastlines()
    plt.show()
    
df['Cluster'] = KMeans(3).fit_predict(coordinates)
plot_clusters(df.Cluster, df.Longitude, df.Latitude)
```

<!-- #region id="BzIM05Ku1Nft" -->
These continental categories are too broad to actually be useful. Perhaps our K was too low after all. We'll disregard the recommended K-value from the Elbow analysis, and double the size of K to 6.

**Using K-means to cluster cities into 6 groups**
<!-- #endregion -->

```python id="N9PrZy6I1Nfu" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637422996583, "user_tz": -330, "elapsed": 1167, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a9b83625-5860-4c68-fbcd-91ff33de53c7"
df['Cluster'] = KMeans(6).fit_predict(coordinates)
plot_clusters(df.Cluster, df.Longitude, df.Latitude)
```

<!-- #region id="dBH5B4tk1Nfu" -->
K-mean's sense of centrality is unable to properly distinguish between Africa, Europe and Asia. As an alternative approach, we can attempt to execute DBSCAN clustering. The DBSCAN algorithm takes as input any distance metric
of our choosing, allowing us to cluster on the great-circle distance between points.

**Listing 12. 20. Defining a NumPy-based great-circle metric**
<!-- #endregion -->

```python id="hctRVYHLBNPX"
import numpy as np
from math import cos, sin, asin
from sklearn.cluster import DBSCAN
```

```python id="TWBA0qBa1Nfv"
def great_circle_distance(coord1, coord2, radius=3956):
    if np.array_equal(coord1, coord2):
        return 0.0 

    coord1, coord2 = np.radians(coord1), np.radians(coord2)
    delta_x, delta_y = coord2 - coord1
    haversin = sin(delta_x / 2) ** 2 + np.product([cos(coord1[0]),
                                                   cos(coord2[0]), 
                                                   sin(delta_y / 2) ** 2])
    return  2 * radius * asin(haversin ** 0.5)
```

<!-- #region id="1iPIn3Tk1Nfv" -->
With our distance metric in place, we are ready to run the DBSCAN algorithm. 

**Listing 12. 21. Using DBSCAN to cluster cities**
<!-- #endregion -->

```python id="ud7prUKY1Nfw"
metric = great_circle_distance
dbscan = DBSCAN(eps=250, min_samples=3, metric=metric)
df['Cluster'] = dbscan.fit_predict(coordinates)
```

<!-- #region id="5MDeudJC1Nfw" -->
DBSCAN assigns -1 to outlier data-points that do not cluster. Lets remove these outliers from our table. Afterwards, we’ll plot the remaining results.

**Listing 12. 22. Plotting non-outlier DBSCAN clusters**
<!-- #endregion -->

```python id="uyfDIOic1Nfx" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637423121668, "user_tz": -330, "elapsed": 1038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4143a7cc-c3d8-4631-f05e-f05f9e25b6b9"
df_no_outliers = df[df.Cluster > -1]
plot_clusters(df_no_outliers.Cluster, df_no_outliers.Longitude,
              df_no_outliers.Latitude)
```

<!-- #region id="Qls_HcXy1Nfy" -->
DBSCAN has does a decent job of generating discrete clusters within parts of South America, Asia, and Southern Africa. The Eastern United States however, falls into a single overly-dense cluster. Lets cluster US locations independently from the rest of the World. To do so, we will first assign country-codes across each of our cities.

**Listing 12. 23. Assigning country codes to cities**
<!-- #endregion -->

```python id="aKVUJlzn1Nfz"
def get_country_code(city_name):
    city = max(gc.get_cities_by_name(city_name), 
               key=lambda x: list(x.values())[0]['population'])
    return list(city.values())[0]['countrycode']

df['Country_code'] = df.City.apply(get_country_code)
```

<!-- #region id="RFhfgXeo1Nf0" -->
The country-codes allow us to separate the data into 2 distinct `DataFrame` objects. The first object, `df_us`,  which hold all the United States locations. The second object, `df_not_us`, will hold all the remaining global cities.

**Listing 12. 24. Seperating US and global cities**
<!-- #endregion -->

```python id="gqkplKJT1Nf0"
df_us = df[df.Country_code == 'US']
df_not_us = df[df.Country_code != 'US']
```

<!-- #region id="OIr1OTO61Nf2" -->
We've separated US and non-US cities. Now, we will need to re-cluster the coordinates within the 2 separated tables. 

**Listing 12. 25. Re-clustering extracted cities**
<!-- #endregion -->

```python id="-9L25Z5o1Nf2"
def re_cluster(input_df, eps):
    input_coord = input_df[['Latitude', 'Longitude']].values
    dbscan = DBSCAN(eps=eps, min_samples=3, 
                    metric=great_circle_distance)
    clusters = dbscan.fit_predict(input_coord)
    input_df = input_df.assign(Cluster=clusters)
    return input_df[input_df.Cluster > -1]

df_not_us = re_cluster(df_not_us, 250)
df_us = re_cluster(df_us, 125)
```

<!-- #region id="RtfH4aCu1Nf3" -->
## Extracting Insights from Location Clusters

Lets investigate the clustered data within the `df_not_us` table. We'll start by grouping the clustered results using the Pandas `groupby` method.

**Listing 12. 26. Grouping cities by cluster**
<!-- #endregion -->

```python id="X-rV1ye51Nf3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423165127, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16f4f624-c9eb-435b-9e0b-321e91abe1d1"
groups = df_not_us.groupby('Cluster')
num_groups = len(groups)
print(f"{num_groups} Non-US have been clusters detected")
```

<!-- #region id="YTeOtP_q1Nf4" -->
31 global clusters have been detected. Lets sort these groups by size and count the headlines in the largest cluster.

**Listing 12. 27. Finding the largest cluster**
<!-- #endregion -->

```python id="5Jr1C3kW1Nf5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423168616, "user_tz": -330, "elapsed": 416, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="21e69f61-f1c2-4c39-91e0-f8d065982cb8"
sorted_groups = sorted(groups, key=lambda x: len(x[1]), 
                       reverse=True)
group_id, largest_group = sorted_groups[0]
group_size = len(largest_group)
print(f"Largest cluster contains {group_size} headlines")
```

<!-- #region id="kFENEFOc1Nf5" -->
The largest cluster contains 51 total headlines. Reading all these headlines individually will be a time-consuming process. We can save time by outputting just those headlines that represent the most central locations in the cluster.

**Listing 12. 28. Computing cluster centrality**
<!-- #endregion -->

```python id="MEj0qqcs1Nf7"
def compute_centrality(group):
    group_coords = group[['Latitude', 'Longitude']].values
    center = group_coords.mean(axis=0)
    distance_to_center = [great_circle_distance(center, coord)
                          for coord in group_coords]
    group['Distance_to_center'] = distance_to_center
```

<!-- #region id="_jCRIgnH1Nf7" -->
Computing the centrality allows us to sort the grouped locations based on their distance to the centers, in order to output the most central headlines. Lets print the 5 most central headlines within our largest cluster.

**Listing 12. 29. Finding the central headlines in largest cluster**
<!-- #endregion -->

```python id="Lc1fOSpL1Nf9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423175675, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f96842fd-0ebe-40aa-be3d-c0ca9c7d871e"
def sort_by_centrality(group):
    compute_centrality(group)
    return group.sort_values('Distance_to_center', ascending=True)

largest_group = sort_by_centrality(largest_group)
for headline in largest_group.Headline.values[:5]:
    print(headline)
```

<!-- #region id="hqRlnog-1Nf-" -->
The central headlines in largest_cluster focus on an outbreak of Mad Cow Disease within various European cities. We can confirm that the cluster’s locale is centered in Europe by outputting the top countries associated with cities in the cluster.

**Listing 12. 30. Finding the top 3 countries in largest cluster**
<!-- #endregion -->

```python id="HKwmKhPx1Nf-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423178691, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="734a2591-5a8c-45bc-febc-132eacbfb2ad"
from collections import Counter
def top_countries(group):
    countries = [gc.get_countries()[country_code]['name']
                 for country_code in group.Country_code.values]
    return Counter(countries).most_common(3)


print(top_countries(largest_group))
```

<!-- #region id="bcTV--GS1Nf_" -->
Lets repeat this analysis across the next 4 largest global clusters.

**Listing 12. 31. Summarizing content within the largest clusters**
<!-- #endregion -->

```python id="fjC4s0N51NgA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423182660, "user_tz": -330, "elapsed": 708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7275aa2f-f6e0-4a5a-d6ad-be1755281244"
for _, group in sorted_groups[1:5]:
    sorted_group = sort_by_centrality(group)
    print(top_countries(sorted_group))
    for headline in sorted_group.Headline.values[:5]:
        print(headline)
    print('\n')
```

<!-- #region id="PRVKCLHQ1NgB" -->
Lets turn our attention to the US clusters. We'll start by visualizing the clusters on a map of the United States.

**Listing 12. 32. Plotting United States DBSCAN clusters**
<!-- #endregion -->

```python id="fwxvAKSL1NgC" colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"status": "ok", "timestamp": 1637423189720, "user_tz": -330, "elapsed": 3806, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="30d15c33-b8bc-444d-e8c1-0030fbbf7795"
fig = plt.figure(figsize=(12, 10))
map_lcc = Basemap(projection='lcc', llcrnrlon=-119,
                  llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
                   lat_1=33, lat_2=45, lon_0=-95)

map_lcc.scatter(df_us.Longitude.values, df_us.Latitude.values,
                c=df_us.Cluster, latlon=True)
map_lcc.drawcoastlines()
map_lcc.drawstates()
plt.show()
```

<!-- #region id="OmSuaUgU1NgD" -->
The visualized map yields reasonable outputs. We'll proceed to analyze the top 5 US clusters by printing their centrality-sorted headlines.

**Listing 12. 33. Summarizing content within the largest US clusters**
<!-- #endregion -->

```python id="iaf69KtA1NgE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637423195403, "user_tz": -330, "elapsed": 897, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fa372a34-8af0-4da9-d593-f9ba35a00685"
us_groups = df_us.groupby('Cluster')
us_sorted_groups = sorted(us_groups, key=lambda x: len(x[1]),
                         reverse=True)
for _, group in us_sorted_groups[:5]:
    sorted_group = sort_by_centrality(group)
    for headline in sorted_group.Headline.values[:5]:
        print(headline)
    print('\n')
```

<!-- #region id="CiyxftJl1NgF" -->
Lets plot one additional image. It will summarize the menacing scope of the spreading Zika epidemic. The image will display all US and global clusters where Zika is mentioned in more than 50% of article headlines.

**Listing 12. 34. Plotting Zika clusters**
<!-- #endregion -->

```python id="P86PlVB-1NgG" colab={"base_uri": "https://localhost:8080/", "height": 453} executionInfo={"status": "ok", "timestamp": 1637423201594, "user_tz": -330, "elapsed": 1188, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35066afa-87e9-4b13-e4ee-2c14cc27842a"
def count_zika_mentions(headlines):
    zika_regex = re.compile(r'\bzika\b', 
                            flags=re.IGNORECASE)
    zika_count = 0
    for headline in headlines:
        if zika_regex.search(headline): 
            zika_count += 1
    
    return zika_count

fig = plt.figure(figsize=(15, 15))
map_plotter = Basemap()

for _, group in sorted_groups + us_sorted_groups:
    headlines = group.Headline.values
    zika_count = count_zika_mentions(headlines)
    if float(zika_count) / len(headlines) > 0.5:
        map_plotter.scatter(group.Longitude.values, 
                            group.Latitude.values,
                            latlon=True)
map_plotter.drawcoastlines()
plt.show()
```

```python id="jS3mh1P29zw6"

```
