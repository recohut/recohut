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

<!-- #region id="FrxGN2yx1Ndp" -->
## The Great-Circle Distance: A Metric for Computing Distances Between 2 Global Points

People have relied on location information since before the dawn of recorded history. Cave dwellers once carved maps of hunting routes into mammoth tusks. Such maps evolved as civilizations flourished. The ancient Babylonians fully mapped the borders of their vast empire. Much later, in 3000 BC, Greek scholars improved cartography using mathematical innovations. The Greeks discovered that the Earth was round and accurately computed the planet’s circumference. Greek mathematicians laid the groundwork for measuring distances across the Earth’s curved surface. Such measurements required the creation of a geographic coordinate system: a rudimentary system based on latitude and longitude was introduced in 2000 BC.

Combining cartography with latitude and longitude helped revolutionize maritime navigation. Sailors could more freely travel the seas by checking their positions on a map. Roughly speaking, maritime navigation protocols followed these three steps:

Data observation—A sailor recorded a series of observations including wind direction, the position of the stars, and (after approximately AD 1300) the northward direction of a compass.

Mathematical and algorithmic analysis of data—A navigator analyzed all of the data to estimate the ship’s position. Sometimes the analysis required trigonometric calculations. More commonly, the navigator consulted a series of rule-based measurement charts. By algorithmically adhering to the rules in the charts, the navigator could figure out the ship’s coordinates.

Visualizing and decision making—The captain examined the computed location on a map relative to the expected destination. Then the captain would give orders to adjust the ship’s orientation based on the visualized results.

This navigation paradigm perfectly encapsulates the standard data science process. As data scientists, we are offered raw observations. We algorithmically analyze that data. Then, we visualize the results to make critical decisions. Thus, data science and location analysis are linked. That link has only grown stronger through the centuries. Today, countless corporations analyze locations in ways the ancient Greeks could never have imagined. Hedge funds study satellite photos of farmlands to make bets on the global soybean market. Transport-service providers analyze vast traffic patterns to efficiently route fleets of cars. Epidemiologists process newspaper data to monitor the global spread of disease.

The direct path between 2 points along the surface of a sphere is called the **great-circle distance**. That distance  depends on a series of well-known trigonometric operations.
<!-- #endregion -->

<!-- #region id="IrIi2k-V4HFn" -->
<!-- #endregion -->

<!-- #region id="wFmBd9M24r4N" -->
We can compute the great-circle distance given a sphere and two points on that sphere. Any point on the sphere’s surface can be represented using spherical coordinates x and y, where x and y measure the angles of the point relative to the x-axis and y-axis.
<!-- #endregion -->

<!-- #region id="cQQvmiNc4uMA" -->
<!-- #endregion -->

<!-- #region id="KWgJWfAJ4Du4" -->
**Defining a great-circle distance function**
<!-- #endregion -->

```python id="y22Zamht1Ndq"
from math import cos, sin, asin

def great_circle_distance(x1, y1, x2, y2):
    delta_x, delta_y = x2 - x1, y2 - y2
    haversin = sin(delta_x / 2) ** 2 + np.product([cos(x1), cos(x2), 
                                                   sin(delta_y / 2) ** 2])
    return 2 * asin(haversin ** 0.5)
```

<!-- #region id="HrRGNj621Ndu" -->
Lets calculate the great-circle distance between 2 points that lie 180 degrees apart, relative to both the x-axis and the y-axis.

**Computing the great-circle distance**
<!-- #endregion -->

```python id="lykoT3Zd1Ndv"
from math import pi
distance = great_circle_distance(0, 0, pi, pi)
print(f"The distance equals {distance} units")
```

<!-- #region id="i6PffwGm1Ndw" -->
The points are exactly π units apart, half the distance required to circumnavigate a unit-circle. This is akin to traveling between the North and South Poles of any planet. We'll confirm by analyzing the latitudes and longitudes of Earth's North Pole and South Pole. Lets begin by recording the known coordinates of each pole.
<!-- #endregion -->

```python id="3Z61vbq31Ndx"
latitude_north, longitude_north = (90.0, 0)
latitude_south, longitude_south = (-90.0, 0)
```

<!-- #region id="slh9mYi31Ndz" -->
Latitudes and longitudes measure spherical coordinates in degrees, not radians. We'll thus convert to radians from degrees using the `np.radians` function. Afterwards, we'll input the radian results into `great_circle_distance`.

**Computing the great-circle distance between poles**
<!-- #endregion -->

```python id="aqwlU58S1Nd0"
to_radians =  np.radians([latitude_north, longitude_north, 
                          latitude_south, longitude_south])
distance = great_circle_distance(*to_radians.tolist())
print(f"The unit-circle distance between poles equals {distance} units")
```

<!-- #region id="zXC0JXeG1Nd2" -->
As expected, the distance between poles on a unit-sphere is π . Now, let's measure the distance between 2 poles here on Earth. The radius of the Earth is not 1 hypothetical unit, but rather 3956 actual miles.

**Computing the travel distance between Earth’s poles**
<!-- #endregion -->

```python id="eMtFkh7E1Nd3"
earth_distance = 3956 * distance
print(f"The distance between poles equals {earth_distance} miles")
```

<!-- #region id="vzfOtj1-1Nd4" -->
Lets create a general `travel_distance` function to calculate the travel mileage between any 2 terrestrial points.

**Defining a travel distance function**
<!-- #endregion -->

```python id="4D-MDgwc1Nd6"
def travel_distance(lat1, lon1, lat2, lon2):
    to_radians =  np.radians([lat1, lon1, lat2, lon2])
    return 3956 * great_circle_distance(*to_radians.tolist())

assert travel_distance(90, 0, -90, 0) == earth_distance
```

<!-- #region id="aal9I1yj1Nd7" -->
## Plotting Maps Using Basemap

Basemap is a Matplotlib extension for generating maps in Python. Lets install the Basemap library, a import the `Basemap` mapping class. Afterwards, we'll initialize the class as `map_plotter = Basemap()`.

**Initializing the Basemap mapping class**
<!-- #endregion -->

```python id="50kFsy4V5Chm"
!sudo apt-get install libgeos-3.5.0
!sudo apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip
```

```python id="hYmARcJX5fxv"
import matplotlib.pyplot as plt
```

```python id="XlpF-Kv-1Nd8"
from mpl_toolkits.basemap import Basemap
map_plotter = Basemap()
```

<!-- #region id="jasJClwG1Nd8" -->
We are ready to visualize the Earth, by plotting the coastline boundaries of all 7 continents.

**Visualizing the Earth using Basemap**
<!-- #endregion -->

```python id="rVnqYbnl1Nd9" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637421035631, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="94371385-c6cf-4b97-c9d9-8c9bf2eed3bf"
# Basemap usage generates warning outputs within the Jupyter notebook. 
# We'll use `warnings` module to deactivate these distracting outputs.
import warnings
warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(12, 8))
map_plotter.drawcoastlines()
plt.show()
```

<!-- #region id="K5Tb_sCR1Nd_" -->
National boundaries are currently missing from the plot. We can incorporate country boundaries by calling the `map_plotter.drawcountries()` method.

**Mapping coastlines and countries**
<!-- #endregion -->

```python id="sHlZTZ4j1NeA" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637421068031, "user_tz": -330, "elapsed": 1583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1bda7bf9-2159-413f-c1ef-cec4b8f92a69"
fig = plt.figure(figsize=(12, 8))
map_plotter.drawcoastlines()
map_plotter.drawcountries()
plt.show()
```

<!-- #region id="HhCYhHJ21NeB" -->
So far our map looks sparse and uninviting. We can improve the quality by calling `map_plotter.shadedrelief()`. The method-call will color the map using topographic information.

**Coloring a map of the Earth**
<!-- #endregion -->

```python id="BmNCBmhr1NeC" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637421106550, "user_tz": -330, "elapsed": 15344, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c1475771-1069-4325-9b74-1cc82ddc1de7"
fig = plt.figure(figsize=(12, 8))
map_plotter.shadedrelief()
plt.show()
```

<!-- #region id="HxbboExk1NeE" -->
Suppose we are given a list of locations defined by pairs of latitudes and longitudes. We can plot these locations on our map by separating the latitudes from the longitudes and then passing the results into `map_plotter.scatter`.

**Plotting coordinates on a map**
<!-- #endregion -->

```python id="9bkUka6D1NeG" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1637421252673, "user_tz": -330, "elapsed": 11697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01a63584-572f-4857-bd98-9f3fdacd320a"
import numpy as np

fig = plt.figure(figsize=(12, 8))
coordinates = [(39.9526, -75.1652), (37.7749, -122.4194),
               (40.4406, -79.9959), (38.6807, -108.9769),
               (37.8716, -112.2727), (40.7831, -73.9712)]

latitudes, longitudes = np.array(coordinates).T
map_plotter.scatter(longitudes, latitudes, latlon=True)
map_plotter.shadedrelief()
plt.show()
```

<!-- #region id="w5qeKCBG1NeJ" -->
The plotted points all appear within the boundaries of North America. We thusly can simplify the map by zooming in on North America. In order to adjust the map, we must alter our projection.

**Plotting North American Coordinates**
<!-- #endregion -->

```python id="SCb9tNNc1NeK" colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"status": "ok", "timestamp": 1637421254358, "user_tz": -330, "elapsed": 1715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4ac75381-b5c3-4e7e-ec43-64d9691fc088"
fig = plt.figure(figsize=(12, 8))
map_ortho = Basemap(projection='ortho', lat_0=40, lon_0=-95)
map_ortho.scatter(longitudes, latitudes, latlon=True,
                  s=70)
map_ortho.drawcoastlines()
plt.show()
```

<!-- #region id="dfLpgGaB1NeL" -->
We successfully zoomed in on North America. Now, we'll zoom in further, onto the United States.

**Plotting USA Coordinates**
<!-- #endregion -->

```python id="z4LHZsBC1NeM" colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"status": "ok", "timestamp": 1637421257758, "user_tz": -330, "elapsed": 1491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3c7189be-1b31-4f0d-cb70-05c18e81ae7c"
fig = plt.figure(figsize=(12, 8))
map_lcc = Basemap(projection='lcc', lon_0=-95, llcrnrlon=-119, 
                  llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, lat_1=33, 
                  lat_2=45)

map_lcc.scatter(longitudes, latitudes, latlon=True, s=70)
map_lcc.drawcoastlines()
plt.show()
```

<!-- #region id="ICx7g41H1NeO" -->
Our map of the United States is looking a little sparse. Lets add state boundaries to the map by calling `map_lcc.drawstates()`.

**Mapping state boundaries in the USA**
<!-- #endregion -->

```python id="ni_k4gH21NeP" colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"status": "ok", "timestamp": 1637421261722, "user_tz": -330, "elapsed": 3410, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2227dd8-b827-41da-ba39-03add48940f9"
fig = plt.figure(figsize=(12, 8))
map_lcc.scatter(longitudes, latitudes, latlon=True, s=70)
map_lcc.drawcoastlines()
map_lcc.drawstates()
plt.show()
```

<!-- #region id="m5aB28JK1NeQ" -->
Basemap allows us to plot any location on a map. All we need is the location’s latitude and longitude. Thus, we need a mapping between location names and their geographic properties. That mapping is provided by the GeoNamesCache location-tracking library.

## Location Tracking Using GeoNamesCache

GeoNamesCache is designed to efficiently retrieve data pertaining to continents, countries, and cities, as well US counties and US states. Lets install the library and explore its usage in more detail. We'll begin by initializing a `GeonamesCache` location-tracking object.
<!-- #endregion -->

```python id="RKj4ciJY6frv"
!pip install geonamescache
```

```python id="zUnRRH3B1NeR"
from geonamescache import GeonamesCache
gc = GeonamesCache()
```

<!-- #region id="3jReJnPa1NeS" -->
Lets use our `gc` object to explore the 7 continents. We'll run `gc.get_continents()` in order to retrieve a dictionary of continent-related information.

**Fetching all 7 continents from GeoNamesCache**
<!-- #endregion -->

```python id="nXulsH5U1NeT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421295536, "user_tz": -330, "elapsed": 1099, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="37a7c47c-00b9-4aaf-ab62-84d6daf91028"
continents = gc.get_continents()
print(continents.keys())
```

<!-- #region id="WfzXWZtY1NeU" -->
The dictionary keys represent shorthand encoding of continent names, in which __Africa__ is transformed into `'AF'`, and __North America__ is transformed into `'NA'`. Lets check the values mapped to every key by passing in the code for __North America__.

**Fetching North America from GeoNamesCach**
<!-- #endregion -->

```python id="W4Zi05QC1NeX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421303412, "user_tz": -330, "elapsed": 931, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ffbc2ce8-0f96-4627-9ce8-b27166d0efa5"
north_america = continents['NA']
print(north_america.keys())
```

<!-- #region id="pdGiVttV1NeZ" -->
Many of the `north_america` data elements represent the various naming schemes for the North American continent. Such information is not very useful.

**Printing North America’s naming schemas**
<!-- #endregion -->

```python id="LLDkub1L1Neb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421311653, "user_tz": -330, "elapsed": 1834, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="208e322c-1042-4823-c86e-55308bb2f55a"
for name_key in ['name', 'asciiName', 'toponymName']:
    print(north_america[name_key])
```

<!-- #region id="4ge78x6p1Nee" -->
The  `'lat'` and the `'lng'` keys map to the latitude and longitude of the central-most location in the continent. We can utilize these coordinates to plot a map projection centered at the heart of North America.

**Mapping North America’s central coordinates**
<!-- #endregion -->

```python id="c3z-OAqQ1Neg" colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"status": "ok", "timestamp": 1637421314460, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="70a5897f-593d-4a68-d1d9-7c82b2c02b3d"
latitude = float(north_america['lat'])
longitude = float(north_america['lng'])

fig = plt.figure(figsize=(12, 8))
map_plotter = Basemap(projection='ortho',lat_0=40, lon_0=-95)
map_plotter.scatter([longitude], [latitude], latlon=True, s=200)
map_plotter.drawcoastlines()
map_plotter.drawcountries()
plt.show()
```

<!-- #region id="vQKHmW4h1Nei" -->
## Accessing Country Information
We can analyze countries using the `get_countries` method. It returns a dictionary whose 2-character keys encode the names of 252 different countries. Accessing `gc.get_countries()['US']` will return a dictionary containing useful USA statistics. Lets output all the non-city information pertaining to the United States.

**Fetching US data from GeoNamesCache**
<!-- #endregion -->

```python id="bUuq9P9Y1Nej" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421322479, "user_tz": -330, "elapsed": 767, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4fb21d9e-3cdf-4eb5-89f3-6848576cfce9"
countries = gc.get_countries()
num_countries = len(countries)
print(f"GeonamesCache holds data for {num_countries} countries.")

us_data = countries['US']
print("The following data pertains to the United States:")
print(us_data)
```

<!-- #region id="zjP-vXm31Nem" -->
There is valuable information within each country's `'neighbours'` element. It maps to a comma-delimited string
of country codes that signify all neighboring territories. We can obtain more details about each neighbor by splitting the string and passing the codes into the `'countries'` dictionary.

**Fetching neighboring countries**
<!-- #endregion -->

```python id="utaZO6fk1Nen" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421325102, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93658168-915a-4629-eff4-928d273854c9"
us_neighbors = us_data['neighbours']
for neighbor_code in us_neighbors.split(','):
    print(countries[neighbor_code]['name'])
```

<!-- #region id="-8JIHEpI1Nep" -->
We can also query all countries by name using the `get_countries_by_names` method. This method returns a dictionary whose elements are country names rather than codes.

**Fetching countries by name**
<!-- #endregion -->

```python id="EoSgid211Ner" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421327037, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11f8eebd-67cd-4ae0-9c8f-6f3413ca26b1"
result = gc.get_countries_by_names()['United States']
assert result == countries['US']
countries['US']
```

<!-- #region id="JBdYSIFA1Net" -->
## Accessing City Information

The `get_cities` method returns a dictionary whose keys are unique ids mapping back to city data.

**Fetching cities from GeoNamesCache**
<!-- #endregion -->

```python id="uSNqyMes1Nev" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421341502, "user_tz": -330, "elapsed": 1851, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bf5ced26-3d28-4dd7-f7d5-239dc17ca0ce"
cities = gc.get_cities()
num_cities = len(cities)
print(f"GeoNamesCache holds data for {num_cities} total cities")
city_id = list(cities.keys())[0]
print(cities[city_id])
```

<!-- #region id="2lwmRhpN1Nex" -->
The data for each city contains the reference code for the country where that city is located. By utilizing the country code, we can create a new mapping between a country and all its territorial cities.

**Fetching US cities from GeoNamesCache**
<!-- #endregion -->

```python id="nbf1Y7ka1Nez" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421343106, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1fa0dbaa-a2c3-406f-ba29-61dc1ce3f6d7"
us_cities = [city for city in cities.values() 
             if city['countrycode'] == 'US']
num_us_cities = len(us_cities)
print(f"GeoNamesCache holds data for {num_us_cities} US cities.")
```

<!-- #region id="rgDB7NTp1Ne1" -->
Lets find the average US latitude and longitude. This average will approximate the central coordinates of the United States.

**Approximating US central coordinates**
<!-- #endregion -->

```python id="IPFzhO7G1Ne2" colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"status": "ok", "timestamp": 1637421375672, "user_tz": -330, "elapsed": 4369, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a1ee6cf5-1134-4c3e-81ed-d2b86be9c8c5"
center_lat = np.mean([city['latitude'] 
                      for city in us_cities])
center_lon = np.mean([city['longitude'] 
                       for city in us_cities])

fig = plt.figure(figsize=(12, 8))
map_lcc = Basemap(projection='lcc', lon_0=-95, llcrnrlon=-119, 
                  llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, lat_1=33, 
                  lat_2=45)
map_lcc.scatter([center_lon], [center_lat], latlon=True, s=200)
map_lcc.drawcoastlines()
map_lcc.drawstates()
plt.show()
```

<!-- #region id="BGFuDgBG1Ne5" -->
The `get_cities` method is suitable for iterating over city information, but not for querying cities by name. To run a name-based city search, we must rely on `get_cities_by_name`. 

**Fetching cities by name**
<!-- #endregion -->

```python id="GLGrRUMz1Ne6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421381912, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a0d086f-224a-4bb4-bf4e-5427e94f8bc2"
matched_cities_list = gc.get_cities_by_name('Philadelphia')
print(matched_cities_list)
```

<!-- #region id="p77N5ptP1Ne8" -->
The `get_cities_by_name` method may return more than one city, because city-names are not always unique. For example, GeoNamesCache contains 6 different instances of the city __San Francisco__, spanning across 5 different countries.

**Fetching multiple cities with a shared name**
<!-- #endregion -->

```python id="nh9nl0T31Ne9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421385747, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ab72ab7-5775-4407-f346-49b1fd817657"
matched_cities_list = gc.get_cities_by_name('San Francisco')

for i, san_francisco in enumerate(matched_cities_list):
    city_info = list(san_francisco.values())[0]
    country_code = city_info['countrycode']
    country = countries[country_code]['name']
    print(f"The San Francisco at index {i} is located in {country}")
```

<!-- #region id="ftS4mdmJ1Ne-" -->
Its common for multiple cities to share the same name. Choosing among such cities is quite difficult. Usually, the safest guess is the city with the largest population. 

**Mapping the most populous San Francisco**
<!-- #endregion -->

```python id="D-GxVlcc1Ne_" colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"status": "ok", "timestamp": 1637421392550, "user_tz": -330, "elapsed": 4232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="88b8d5f2-9d2b-47ae-9f8e-19c6cb08be94"
best_sf = max(gc.get_cities_by_name('San Francisco'), 
              key=lambda x: list(x.values())[0]['population'])
sf_data = list(best_sf.values())[0]
sf_lat = sf_data['latitude']
sf_lon = sf_data['longitude']

fig = plt.figure(figsize=(12, 8))
map_lcc = Basemap(projection='lcc', lon_0=-95, llcrnrlon=-119, 
                  llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, lat_1=33, 
                  lat_2=45)
map_lcc.scatter([sf_lon], [sf_lat], latlon=True, s=200)
map_lcc.drawcoastlines()
map_lcc.drawstates()

x, y = map_lcc(sf_lon, sf_lat)
plt.text(x, y, ' San Francisco', fontsize=16)
plt.show()
```

<!-- #region id="ziyM_RoB1NfA" -->
### Limitations of the GeoNamesCache Library

The `get_cities_by_name` method maps only one version of a city's name to its geographic data. This poses a problem for cities like __New York__, which carry more than one commonly referenced name.

**Fetching New York City from GeoNamesCache**
<!-- #endregion -->

```python id="Agvx320e1NfA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421401416, "user_tz": -330, "elapsed": 950, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4404570c-37ab-4f32-eeb7-35fa72f16f07"
for ny_name in ['New York', 'New York City']:
    if not gc.get_cities_by_name(ny_name):
        print(f"'{ny_name}' is not present in GeoNamesCache database.")
    else:
        print(f"'{ny_name}' is present in GeoNamesCache database.")
```

<!-- #region id="F6KLuWux1NfB" -->
The limits of single references become particularly obvious when we examine diacritics in city names. Diacritics are accent marks that designate the proper pronunciation of foreign-sounding words. 

**Fetching accented cities from GeoNamesCache**
<!-- #endregion -->

```python id="1iNnKsoG1NfC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421401418, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d5c3482-1e5f-49dc-8915-e42bf72b80bd"
print(gc.get_cities_by_name(u'Cañon City'))
print(gc.get_cities_by_name(u'Hagåtña'))
```

<!-- #region id="OTc1BkBY1NfD" -->
How many of the cities stored in GeoNamesCache contain diacritics in their name? We can find out using the `unidecode` function from the external Unidecode library.

**Counting all accented cities in GeoNamesCache**
<!-- #endregion -->

```python id="2i6BwjUx6-9D"
!pip install unidecode
```

```python id="xuN-5CsE1NfD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421423508, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb559c2a-3fe4-4cb2-d70a-997b4e34861e"
from unidecode import unidecode
accented_names = [city['name'] for city in gc.get_cities().values()
                  if city['name'] != unidecode(city['name'])]
num_accented_cities = len(accented_names)

print(f"An example accented city name is '{accented_names[0]}'")
print(f"{num_accented_cities} cities have accented names") 
```

<!-- #region id="q_hCtCz11NfF" -->
We can now match the stripped dictionary keys against all inputted text by passing the accented dictionary values into GeoNamesCache, whenever a key-match is found.

**Finding accent-free city-names in text**
<!-- #endregion -->

```python id="H0IiOJuJ1NfG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421428484, "user_tz": -330, "elapsed": 771, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6babe8cf-b453-43b8-c4d7-2bae96cd0f4d"
alternative_names = {unidecode(name): name 
                     for name in accented_names}
print(gc.get_cities_by_name(alternative_names['Hagatna']))
```

<!-- #region id="pEvePa8T1NfH" -->
## Matching Location Names in Text

In Python, we can easily determine if one string is a substring of another, or if the start of a string contains some predefined text.

**Basic string matching**
<!-- #endregion -->

```python id="_DCYfTbW1NfJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637421435340, "user_tz": -330, "elapsed": 1665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d82d1bc3-af77-43c5-ff54-c2f988085ef2"
text = u'This sentence matches Hagatna'
for key, value in alternative_names.items():
    if key in text:
        print(gc.get_cities_by_name(value))
        break
```

<!-- #region id="HlExsSQO1NfJ" -->
In more complex analyses, Python's basic string syntax can be quite limiting. For example, Python's string methods can't directly distinguish between sub-characters in a string and sub-phrases in a sentence.

**Basic sub-string matching errors**
<!-- #endregion -->

```python id="8z_KjlWT1NfK"
assert 'in a' in 'sin apple'
assert 'in a' in 'win attached'
```

<!-- #region id="-8RQZFSK1NfL" -->
To overcame these limitations, we must rely on Python's built-in regular expression processing library, `re`. A **regular expression** (or **regex** for short) is a string-encoded pattern that can be compared against some text. Most regex-matching in Python can be executed with the `re.search` function. 

**String matching using regexes**
<!-- #endregion -->

```python id="kgodG0Ra1NfL"
import re
regex = 'Boston'
random_text = 'Clown Patty'
match = re.search(regex, random_text)
assert match is None

matchable_text = 'Boston Marathon'
match = re.search(regex, matchable_text)
assert match is not None
start, end = match.start(), match.end()
matched_string = matchable_text[start: end]
assert matched_string == 'Boston'
```

<!-- #region id="d24O-Dqc1NfM" -->
Case-insensitive string matching is a breeze with `re.search`. We simply pass `re.IGNORECASE` as an added `flags` parameter.

**Case-insensitive matching using regexes**
<!-- #endregion -->

```python id="O-R2wEDj1NfM"
for text in ['BOSTON', 'boston', 'BoSTOn']:
    assert re.search(regex, text, flags=re.IGNORECASE) is not None
```

<!-- #region id="Hx18Kaid1NfN" -->
Also, regexes permit us to match exact words, and not just substrings, using word boundary detection.

**Word boundary matching using regexes**
<!-- #endregion -->

```python id="IITaFmf_1NfN"
for regex in ['\\bin a\\b', r'\bin a\b']:
    for text in ['sin apple', 'win attached']:
        assert re.search(regex, text) is None
        
    text = 'Match in a string'
    assert re.search(regex, text) is not None
```

<!-- #region id="cndslWUO1NfO" -->
Now, let us carry out a more complicated match. We'll match against the sentence `f'I visited {city} yesterday`, where `{city}` represents one of 3 possible locations; `'Boston'`, `'Philadelphia'`, or `'San Francisco'`.

**Multi-city matching using regexes**
<!-- #endregion -->

```python id="LxxuJqZi1NfP"
regex = r'I visited \b(Boston|Philadelphia|San Francisco)\b yesterday.'
assert re.search(regex, 'I visited Chicago yesterday.') is None

cities = ['Boston', 'Philadelphia', 'San Francisco']
for city in cities:
    assert re.search(regex, f'I visited {city} yesterday.') is not None
```

<!-- #region id="7bVYLuYJ1NfP" -->
Suppose we want to match a regex against 100 strings. For every match, `re.search`  will transform the regex into Python `PatternObject.` Each such transformation is computationally costly. We're better off executing the transformation only once using `re.compile`.

**String matching using compiled regexes**
<!-- #endregion -->

```python id="w1VvaPsW1NfQ"
compiled_re = re.compile(regex)
text = 'I visited Boston yesterday.'
for i in range(1000):
    assert compiled_re.search(text) is not None
```

```python id="Y4S-FHwY7GUM"

```
