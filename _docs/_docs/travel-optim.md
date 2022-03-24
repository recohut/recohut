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

<!-- #region id="kIvLTO2oAb1c" -->
# Utilizing Undirected Graphs to Optimize the Travel-Time Between Towns

In business logistics, product delivery time can impact certain critical decisions. Consider the following scenario, in which you’ve opened your own kombucha brewery. Your plan is to deliver batches of the delicious fermented tea to all the towns within a reasonable driving radius. More specifically, you’ll only deliver to a town if it’s within a two-hour driving distance of the brewery; otherwise, the gas costs won’t justify the revenue from that delivery. A grocery store in a neighboring county is interested in regular deliveries. What is the fastest driving time between your brewery and that store?

Normally, you could obtain the answer by searching for directions on a smartphone, but we’ll assume that existing tech solutions are not available (perhaps the area is remote and the local maps have not been scanned into an online database). In other words, you need to replicate the travel time computations carried out by existing smartphone tools. To do this, you consult a printed map of the local area. On the map, roads zigzag between towns, and some towns connect directly via a road. Conveniently, the travel times between connected towns are illustrated clearly on the map. We can model these connections using undirected graphs.

Suppose that a road connects two towns, _Town 0_ and _Town 1_. The driving time between the towns is 20 minutes. Lets record this information in an undirected graph.

**Creating a 2-node undirected graph**
<!-- #endregion -->

```python id="l4tSQuhDINaV"
import networkx as nx
```

```python id="xnibQcQ7Ab1d"
G = nx.Graph()
G.add_edge(0, 1)
G[0][1]['travel_time'] = 20
```

<!-- #region id="iU287vcIAb1e" -->
Our travel-time is an attribute of the edge `(0, 1)`. Given an attribute `k` of edge `(i, j)`, we can access that attribute by running `G[i][j][k]`. Hence, we can access the travel-time by running `G[0][1]['travel_time']`.

**Checking the edge attribute of a graph**
<!-- #endregion -->

```python id="BiPV_GnhAb1e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508803058, "user_tz": -330, "elapsed": 1571, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f30d8327-9585-4bf0-85fc-70f08266bd3a"
for i, j in [(0, 1), (1, 0)]:
    travel_time = G[i][j]['travel_time']
    print(f"It takes {travel_time} minutes to drive from Town {i} to Town {j}.")
```

<!-- #region id="TPmNTBi0Ab1f" -->
Imagine an additional _Town 2_  that is connected to _Town 1_ but not _Town 0_. There is no road between _Town 0_ and _Town 2_. There is a road between  _Town 1_ and _Town 2_. The travel-time on that road is 15 minutes. Let's add this new connection to our graph. Afterwards, we'll visualize the graph using `nx.draw`.

**Visualizing a path between multiple towns**
<!-- #endregion -->

```python id="kTvcmnvbIZAk"
import numpy as np
import matplotlib.pyplot as plt
```

```python id="DC0cOlCjAb1f" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508831384, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a3df581-c655-4d44-ec40-222d99209f4c"
np.random.seed(0)
G.add_edge(1, 2, travel_time=15)
nx.draw(G, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="5onLpytuAb1g" -->
Traveling from _Town 0_ to _Town 2_ requires us to first traverse _Town 1_. Hence, the total travel-time is equal to the sum of `G[0][1]['travel_time']` and `G[1][2]['travel_time']`.

**Computing the travel-time between towns**
<!-- #endregion -->

```python id="CcQhZ1yGAb1g" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508835314, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6fac2d2-3f1f-4f7e-a8db-2e494cdd70d8"
travel_time = sum(G[i][1]['travel_time'] for i in [0, 2])
print(f"It takes {travel_time} minutes to drive from Town 0 to Town 2.")
```

<!-- #region id="7N2XlMWLAb1h" -->
Our computation was trivial, since there is just one route between _Town 0_ and _Town 2_. However, in real-life, many routes can exist between localized towns. Lets build a graph containing more than a dozen towns. These cities will be spread across multiple counties. Within our graph model, the travel-time between towns will increase when cities are in different counties. We'll assume that:

A. Our towns cover six different counties.

B. Each county contains 3 - 10 towns.

C. 90% of towns within a single county are directly connected by road.
* The average travel-time on a county road is 20 minutes.

D. 5% of cities across different counties are directly connected by a road.
*  The average travel-time across an intra-county road is 45 minutes

### Modeling a Complex Network of Cities and Counties

Lets start by modeling a single county that contains five towns.

**Modeling five towns in the same county**
<!-- #endregion -->

```python id="7yOHpPKEAb1h"
G = nx.Graph()
G.add_nodes_from((i, {'county_id': 0}) for i in range(5))
```

<!-- #region id="coXYBne3Ab1i" -->
Next, we'll assign random roads to our five towns.

**Modeling random intra-county roads**
<!-- #endregion -->

```python id="V4Wrw_68Ab1i" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508849938, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1a864a2-e5e9-461a-88b6-b0bff4a6d4dc"
np.random.seed(0)
def add_random_edge(G, node1, node2, prob_road=0.9, 
                    mean_drive_time=20):
    if np.random.binomial(1, prob_road):
        drive_time = np.random.normal(mean_drive_time)
        G.add_edge(node1, node2, travel_time=round(drive_time, 2))


nodes = list(G.nodes())
for node1 in nodes[:-1]:
    for node2 in nodes[node1 + 1:]:
        add_random_edge(G, node1, node2)
            
nx.draw(G, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="XpBAWD-QAb1j" -->
We've connected most of the towns in _County 0_. In this same manner, we can randomly generate roads and towns for a second county; _County 1_.

**Modeling a second random county**
<!-- #endregion -->

```python id="iL3zood2Ab1k" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508858259, "user_tz": -330, "elapsed": 890, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8304f5f3-a1f7-498a-c79b-21301acae419"
np.random.seed(0)
def random_county(county_id):
    num_towns = np.random.randint(3, 10)
    G = nx.Graph()
    nodes = [(node_id, {'county_id': county_id})
            for node_id in range(num_towns)]
    G.add_nodes_from(nodes)
    for node1, _ in nodes[:-1]:
        for node2, _ in nodes[node1 + 1:]:
            add_random_edge(G, node1, node2)

    return G

G2 = random_county(1)
nx.draw(G2, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="Qrf0VKbKAb1k" -->
Currently, _County 1_ and _County 2_ are stored  in two separate graphs; `G` and `G2`. We can combine these graphs together by executing `nx.disjoint_union(G, G2)`.

**Merging two separate graphs**
<!-- #endregion -->

```python id="3uoI4LqGAb1k" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508866064, "user_tz": -330, "elapsed": 593, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f95ddac-4aa3-4110-e9f7-55682822cf15"
np.random.seed(0)
G = nx.disjoint_union(G, G2)
nx.draw(G, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="FWs1XL0OAb1l" -->
Our two counties appear in the same graph. Each town in the graph is assigned a unique id. Now, it's time to generate random roads between the counties.


**Adding random inter-county roads**
<!-- #endregion -->

```python id="VwSswgfyAb1l" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508869645, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="715ae194-96a7-4206-f44c-eb0f64ad4298"
np.random.seed(0)
def add_intracounty_edges(G):
    nodes = list(G.nodes(data=True))
    for node1, attributes1 in nodes[:-1]:
        county1 = attributes1['county_id']
        for node2, attributes2 in nodes[node1:]:
            if county1 != attributes2['county_id']:
                add_random_edge(G, node1, node2,
                                prob_road=0.05, mean_drive_time=45)
    return G

G = add_intracounty_edges(G)
np.random.seed(0)
nx.draw(G, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="nRlc75OsAb1m" -->
We’ve successfully simulated two interconnected counties. Now, we’ll simulate six interconnected counties.

**Simulating six interconnected counties**
<!-- #endregion -->

```python id="CEWEkQvWAb1m" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508918596, "user_tz": -330, "elapsed": 1027, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a2e6cc77-ddfd-418f-bbc1-abf09ccafe20"
np.random.seed(1)
G = random_county(0)
for county_id in range(1, 6):
    G2 = random_county(county_id)
    G = nx.disjoint_union(G, G2)

G = add_intracounty_edges(G)
np.random.seed(1)
nx.draw(G, with_labels=True, node_color='khaki')
plt.show()
```

<!-- #region id="Rp2657wLAb1n" -->
We’ve visualized our six-county graph. However, individual counties are tricky to decipher in the visualization. Fortunately, we can improve our plot by coloring each node based on county id. Doing so requires that we modify our input into the `node_color` parameter.

**Coloring nodes by county**
<!-- #endregion -->

```python id="MJ4R32joAb1n" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637508923087, "user_tz": -330, "elapsed": 1313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f47b5905-4653-4e6e-9d62-8d17326e9e85"
np.random.seed(1)
county_colors = ['salmon', 'khaki', 'pink', 'beige', 'cyan', 'lavender']
county_ids = [G.nodes[n]['county_id'] 
              for n in G.nodes]
node_colors = [county_colors[id_] 
               for id_ in county_ids]
nx.draw(G, with_labels=True, node_color=node_colors)
```

<!-- #region id="RGOwjAWQAb1o" -->
The individual counties are now visible. Most counties form tight clumps within the network. Later, we’ll proceed to extract these clumps automatically, using network clustering. For now, however, we’ll focus our attention on computing the fastest travel-time between nodes.

### Computing the Fastest Travel-Time Between Nodes

Suppose we want to know the fastest travel-time between _Town 0_ and _Town 30_. In the process, we’ll need to compute the fastest travel-time between _Town 0_ and every other town. Initially, all we know is the trivial travel-time between _Town 0_ and itself; 0 minutes. Let’s record this travel-time in a `fastest_times` dictionary. 

**Tracking the fastest-known travel times**
<!-- #endregion -->

```python id="_xpbghwlAb1o"
fastest_times = {0: 0}
```

<!-- #region id="hOM4-VfSAb1o" -->
Next, we can answer a simple question; what is the known travel-distance between _Town 0_ and its neighboring towns? In NetworkX, we can access the neighbors of _Town 0_ by executing `G.neighbors(0)`. Alternatively, we can access the neighbours simply by running `G[0]`.

**Accessing the neighbors of _Town 0_**
<!-- #endregion -->

```python id="9U2j6cFnAb1o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508930638, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4702e7f3-03d1-4def-c2da-082d00e7ee60"
neighbors = list(G.neighbors(0))
assert list(neighbors) == list(G[0])
print(f"The following towns connect directly with Town 0:\n{neighbors}")
```

<!-- #region id="ChLb-VS5Ab1p" -->
Now, we’ll record the travel-times between _Town 0_ and each of its five neighbors. We’ll use these times to update `fastest_times`.

**Tracking the travel-times to neighboring towns**
<!-- #endregion -->

```python id="6F0M6qcFAb1p" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508931144, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a1e40c53-6eb9-4233-a0f4-182f83200ce7"
time_to_neighbor = {n: G[0][n]['travel_time'] for n in neighbors}
fastest_times.update(time_to_neighbor)
for neighbor, travel_time in sorted(time_to_neighbor.items(), 
                                    key=lambda x: x[1]):
    print(f"It takes {travel_time} minutes to drive from Town 0 to Town "
          f"{neighbor}.")
```

<!-- #region id="PO-OA-72Ab1p" -->
It takes approximately 45 minutes to drive from _Town 0_ to _Town 13_. Is this the fastest travel-time between these two towns? Not necessarily! It’s possible that a detour through another town will speed-up travel. Consider, for instance, a detour through _Town 7_. It’s our most proximate town, with a drive-time of only 18 minutes. What if there’s a road between _Town 7_ and _Town 13_?  If that road exists, and its travel-time is under 27 minutes, then a faster route to _Town 13_ is possible! We can potentially shave-off minutes from travel, if we examine the neighbors of _Town 7_. Lets carry-out that examination thusly:

1. First, we’ll obtain the neighbors of _Town 7_.

2. Next, we’ll obtain travel-time between _Town 7_ and every neighboring _Town N_.

3. We’ll add 18.4 minutes to the travel-time obtained in the previous step. This represents the travel-time between _Town 0_ and _Town N_, when we take a detour through _Town 7_.

4. If _N_ is present in `fastest_times`, we’ll check if the detour is faster than `fastest_times[N]`.

5. If _N_ is not present in `fastest_times`, then we will update that dictionary with the travel-time computed in Step 3.

**Searching for faster detours through _Town 7_**
<!-- #endregion -->

```python id="bsmf9Al5Ab1q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508932290, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="821e9457-4845-4142-d7c9-f1f84177ca92"
def examine_detour(town_id):
    detour_found = False
    
    travel_time = fastest_times[town_id]
    for n in G[town_id]:
        detour_time = travel_time + G[town_id][n]['travel_time']
        if n in fastest_times:
            if detour_time < fastest_times[n]:
                detour_found = True
                print(f"A detour through Town {town_id} reduces "
                      f"travel-time to Town {n} from "
                      f"{fastest_times[n]:.2f} to "
                      f"{detour_time:.2f} minutes.")
                fastest_times[n] = detour_time
        
        else:
            fastest_times[n] = detour_time
    
    return detour_found

if not examine_detour(7):
    print("No detours were found.")
    
added_towns = len(fastest_times) - 6
print(f"We've computed travel-times to {added_towns} additional towns.")
```

<!-- #region id="zlKDyCmeAb1q" -->
We’ve uncovered travel-times to three additional towns. However, we have not uncovered any faster detours for travel to neighbors of _Town 0_. Lets choose another viable detour candidate. We’ll select a town that’s proximate to _Town 0_, whose neighbors we have not examined. Doing so requires that we:

1. Combine the neighbours of _Town 0_ and _Town 7_ into a pool of detour candidates.
2. Remove _Town 0_ and _Town 7_ from the pool of candidates, leaving behind a set of unexamined towns.
2. Select an unexamined town with the fastest known-travel distance to _Town 0_.

**Selecting an alternative detour candidate**
<!-- #endregion -->

```python id="lOEb0RxNAb1q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508934133, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2a674700-cffc-43be-8f12-4adcc1e44d86"
candidate_pool = set(G[0]) | set(G[7])
examined_towns = {0, 7}
unexamined_towns = candidate_pool - examined_towns
detour_candidate = min(unexamined_towns, 
                       key=lambda x: fastest_times[x])
travel_time = fastest_times[detour_candidate]
print(f"Our next detour candidate is Town {detour_candidate}, "
      f"which is located {travel_time} minutes from Town 0.")
```

<!-- #region id="87rgxAuKAb1r" -->
Our next detour candidate is _Town 3_. We’ll proceed to check _Town 3_ for detours.

**Searching for faster detours through _Town 3_**
<!-- #endregion -->

```python id="9hWzPsUtAb1r" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508937726, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2896b60e-63a2-407c-b9f4-91693b0fa63b"
if not examine_detour(detour_candidate):
    print("No detours were found.")

def new_neighbors(town_id):
    return set(G[town_id]) - examined_towns

def shift_to_examined(town_id):
    unexamined_towns.remove(town_id)
    examined_towns.add(town_id)

unexamined_towns.update(new_neighbors(detour_candidate))
shift_to_examined(detour_candidate)
num_candidates = len(unexamined_towns)
print(f"{num_candidates} detour candidates remain.")
```

<!-- #region id="oIQWIyM-Ab1r" -->
Once again, no detours were discovered. Nonetheless, nine detour candidates remain in our `unexamined_towns` set. Lets iteratively examine the remaining candidates. The code below will iteratively:

1. Select an unexamined town with the fastest known travel-time to _Town 0_.
2. Check that town for detours using `examine_detour`.
2. Shift the town’s id from `unexamined_towns` to `examined_towns`.
4. Repeat Step 1 if any unexamined towns remain. Terminate otherwise.

**Examining every town for faster detours**
<!-- #endregion -->

```python id="WYGoToUUAb1s" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508938857, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="398e245f-8561-46eb-d095-ab465b35dfa7"
while unexamined_towns:
    detour_candidate = min(unexamined_towns, 
                       key=lambda x: fastest_times[x])
    examine_detour(detour_candidate)
    shift_to_examined(detour_candidate)
    unexamined_towns.update(new_neighbors(detour_candidate))
```

<!-- #region id="8Lm0ozaAAb1s" -->
We’ve examined the travel-time to every single town, and discovered five possible detours. Two of the detours reduce travel-times to _Towns 29_ and _30_ from 2.1 hours to 1.8 hours. How many other towns are within two hours of _Town 0_? Lets find out.

**Counting all the towns within a 2-hour driving range**
<!-- #endregion -->

```python id="XQKKJRuxAb1s" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508939623, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="def52b42-cd7f-42f5-c127-17d43a97a58a"
close_towns = {town for town, drive_time in fastest_times.items()
               if drive_time <= 2 * 60}

num_close_towns = len(close_towns)
total_towns = len(G.nodes)
print(f"{num_close_towns} of our {total_towns} towns are within two "
      "hours of our brewery.")
```

<!-- #region id="or7zGQMzAb1t" -->
All but two of our towns are within two hours of the brewery. We’ve figured this out by solving the **shortest path length problem**. The problem applies to graphs whose edges contain numeric attributes, which are called **edge weights**. Additionally, a sequence of node-transitions in the graph is called a **path**. Each path occurs over a sequence of edges. The sum of edge weights in that sequence is called the **path length**. The problem asks us to compute the shortest path-lengths between some node _N_ and every single node within the graph.  A shortest path length detection algorithm is included in NetworkX.

**Computing shortest path-lengths with NetworkX**
<!-- #endregion -->

```python id="CSIgwLO_Ab1t"
shortest_lengths = nx.shortest_path_length(G, weight='travel_time', 
                                           source=0)
for town, path_length in shortest_lengths.items():
    assert fastest_times[town] == path_length
```

<!-- #region id="oWRDweRzAb1u" -->
In many real-word circumstances, we want to know the path that minimizes the distance between nodes. Calling `nx.shortest_path_length(G, weight=’weight’, source=N)` will compute all shortest paths from node _N_ to every node in G.

**Computing shortest paths with NetworkX**
<!-- #endregion -->

```python id="kx6qPry5Ab1u" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508943777, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3fadc765-50b4-4390-d776-46b1c50cba0f"
shortest_path = nx.shortest_path(G, weight='travel_time', source=0)[30]
print(shortest_path)
```

<!-- #region id="AzB_jUpxAb1v" -->
Driving time is minimized if we travel from _Town 0_ to _Town 13_ to _Town 28_ and finally to _Town 30_. We expect that travel-time to equal `fastest_times[30]`. Lets confirm.

**Verifying the length of a shortest path**
<!-- #endregion -->

```python id="ZpzUoXQAAb1v" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637508944287, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f205b060-2ed6-4251-d67e-c7ca255c9353"
travel_time = 0
for i, town_a in enumerate(shortest_path[:-1]):
    town_b = shortest_path[i + 1]
    travel_time += G[town_a][town_b]['travel_time']

print("The fastest travel time between Town 0 and Town 30 is "
      f"{travel_time} minutes.")
assert travel_time == fastest_times[30]
```

<!-- #region id="xxkx1oZWAb1v" -->
### Uncovering Central Nodes Based on Expected Traffic within a Network

Suppose our business is growing at an impressive rate. We wish to expand our customer base by putting up a billboard advertisement in one of the local towns represented by G.nodes. To maximize billboard views, we’ll choose the town with the heaviest traffic. Intuitively, traffic is determined by the number of cars that pass through town every day. Can we rank the 31 towns in G.nodes based on the expected daily traffic? Yes, we can! Using simple modeling, we can predict traffic flow from the network of roads between the towns. Later, we’ll expand on these traffic-flow techniques to identify local counties in an automated manner.

We need a way of ranking the towns based on expected traffic. One naive solution is to just count the number of inbound roads into each town. The edge-count of a node within an undirected graph is simply called the node’s **degree**. We can compute the degree of any node `i` by summing over the ith column of the graph’s adjacency matrix. Or, we can measure the degree by running `len(G.nodes[i])`. Alternatively, we can utilize the NetworkX degree method by calling `G.degree(i)`.

**Computing the degree of a single node**
<!-- #endregion -->

```python id="z3ig9a1mAb1w" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509373963, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eae3d38d-cf5d-4591-be8f-409556f5b623"
adjacency_matrix = nx.to_numpy_array(G)
degree_town_0 = adjacency_matrix[:,0].sum()
assert degree_town_0 == len(G[0])
assert degree_town_0 == G.degree(0)
print(f"Town 0 is connected by {degree_town_0:.0f} roads.")
```

<!-- #region id="5jqp6KgnAb1w" -->
In graph theory, any measure of a node’s importance is commonly called **node centrality**. Furthermore, ranked importance based on node’s degree is called the **degree centrality**. We’ll now select the node with the highest degree centrality in `G`.
 
**Selecting a central node using degree centrality** 
<!-- #endregion -->

```python id="4qBzVKBdAb1w" colab={"base_uri": "https://localhost:8080/", "height": 336} executionInfo={"status": "ok", "timestamp": 1637509492935, "user_tz": -330, "elapsed": 981, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f9e800cd-f2a8-4ece-c163-963e82b0e24a"
np.random.seed(1)
central_town = adjacency_matrix.sum(axis=0).argmax()
degree = G.degree(central_town)
print(f"Town {central_town} is our most central town. It has {degree} "
       "connecting roads.")
node_colors[central_town] = 'k'
nx.draw(G, with_labels=True, node_color=node_colors)
plt.show()
```

<!-- #region id="CtQEnhkDAb1x" -->
_Town 3_ our most central town. How does _Town 3_ compare with the second most central town? We’ll quickly check by outputting the second highest degree in `G`.

**Selecting a node with the second highest degree centrality**
<!-- #endregion -->

```python id="FYatps9oAb1x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509514756, "user_tz": -330, "elapsed": 646, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c0d918d1-5011-4fe7-d384-0733bef67757"
second_town = sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)[1]
second_degree = G.degree(second_town)
print(f"Town {second_town} has {second_degree} connecting roads.")
```

<!-- #region id="dlhhbxMZAb1x" -->
_Town 12_ has eight connecting roads. It lags behind _Town 3_ by just one road. What would we do if these two central towns had equal degree rankings? Lets challenge ourselves to find out, by removing an edge connecting _Towns 3_ and _9_.

**Removing an edge from the most central node**
<!-- #endregion -->

```python id="u53wJO_1Ab1x" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509519536, "user_tz": -330, "elapsed": 968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9271846d-ebfb-4653-8342-bf27c3a424e5"
np.random.seed(1)
G.remove_edge(3, 9)
assert G.degree(3) == G.degree(12)
nx.draw(G, with_labels=True, node_color=node_colors)
plt.show()
```

<!-- #region id="cdLKxz1NAb1y" -->
Removal of the road has partially isolated _Town 3_ as well as its neighboring towns. Hence, we can expect _Town 12_ to garner more traffic than _Town 3_ , even though their degrees are equal. In fact, we can quantitate these differences in traffic using random simulations.

### Measuring Centrality Using Traffic Simulations

Lets simulate the random path of a single car. The car will start its journey in a random town `i`. Afterwards, the driver will randomly select one of the  `G.degree(i)` roads that cut through town.  The process will repeat itself until the car has driven through 10 towns.

**Simulating the random route of a single car**
<!-- #endregion -->

```python id="U4Feyy-IAb1y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509529476, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d2462f2-cd35-47c2-f4e5-2365ed11f4be"
np.random.seed(0)
def random_drive(num_stops=10):
    town = np.random.choice(G.nodes)
    for _ in range(num_stops):
        town = np.random.choice(G[town])
        
    return town

destination = random_drive()
print(f"After driving randomly, the car has reached Town {destination}.")
```

<!-- #region id="W5P7EuUwAb1y" -->
Now, we’ll repeat this simulation with 20,000 cars.

**Simulating traffic using 20,000 cars**

<!-- #endregion -->

```python id="f9vaYzHSAb1z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509538465, "user_tz": -330, "elapsed": 4062, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="15f9e5b9-413e-4598-d511-91c147c1d0b5"
import time
np.random.seed(0)
car_counts = np.zeros(len(G.nodes))
num_cars = 20000

start_time = time.time()
for _ in range(num_cars):
    destination = random_drive()
    car_counts[destination] += 1

central_town = car_counts.argmax()
traffic = car_counts[central_town]
running_time = time.time() - start_time
print(f"We ran a {running_time:.2f} second simulation.")
print(f"Town {central_town} has the most traffic.")
print(f"There are {traffic:.0f} cars within that town.")
```

<!-- #region id="U--c8nqVAb1z" -->
_Town 12_ has the most traffic, with over 1000 cars. This is not surprising, given that _Town 12_ has the highest degree centrality, along with _Town 3_. Based on our previous discussion, we expect _Town 12_ to have more heavy traffic than _Town 3_.

**Checking the traffic in _Town 3_**
<!-- #endregion -->

```python id="FEVfMhPzAb1z" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509558274, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2f98aef0-e5f9-48f9-fb0c-6d21fad6fa97"
print(f"There are {car_counts[3]:.0f} cars in Town 3.")
```

<!-- #region id="iz63a5ZNAb10" -->
As expected, _Town 3_ has less than 1000 cars. We should note that car-counts can be cumbersome to compare, especially when `num_cars` is large. Hence, it's preferable to replace these direct counts with probabilities, through division by the simulation count.

**Converting traffic counts to probabilities**
<!-- #endregion -->

```python id="EC2R5gK2Ab10" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509560542, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad3e7554-1a9b-4d79-a678-141e5b08b50c"
probabilities = car_counts / num_cars
for i in [12, 3]:
    prob = probabilities[i]
    print(f"The probability of winding up in Town {i} is {prob:.3f}.")
```

<!-- #region id="QcF_sUaVAb10" -->
W’ve shown that _Town 12_ is more central than _Town 3_. Unfortunately, our simulation process is slow, and doesn’t scale well to larger graphs. In the next sub-subsection we’ll show how to compute the traffic probabilities more efficiently using straightforward matrix multiplication.

## Computing Travel Probabilities Using Matrix Multiplication

Our traffic simulation can be modeled mathematically, using matrices and vectors. We’ll break down this process into simple, manageable parts. Consider for instance, a car that is about to leave _Town 0_ for one of the neighboring towns. The probability of traveling from _Town 0_ to any neighboring town is `1 / G.degree(0)`. It is also equal to `1 / M[:,0].sum()`, where `M` is the adjacency matrix.

**Computing the probability of travel to a neighboring town**
<!-- #endregion -->

```python id="IcQWnRCiAb11" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509573911, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b89c9a1a-a553-48c7-8d5a-defa6be11a18"
prob_travel = 1 / G.degree(0)
assert prob_travel == 1 / adjacency_matrix[:,0].sum()
print("The probability of traveling from Town 0 to one of its "
      f"{G.degree(0)} neighboring towns is {prob_travel}")
```

<!-- #region id="PbdN16SnAb11" -->
If we’re in _Town 0_ and _Town i_ is a neighboring town, then there’s a 20% chance of us traveling from _Town 0_ to _Town i_. Of course, if _Town i_ is not a neighboring town, then the probability drops to zero. We can track the probabilities for every possible `i` using a vector `v`. This vector is called a **transition vector**, since it tracks the probability of transitioning from  _Town 0_  to other towns. 

**Computing a transition vector**
<!-- #endregion -->

```python id="8l8iszEXAb11" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509574686, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="279a6500-ab72-4503-eec7-fc59ac9c7519"
transition_vector = np.array([0.2 if i in G[0] else 0 for i in G.nodes])

adjacency_matrix = nx.to_numpy_array(G)
v2 = np.array([1 if i in G[0] else 0 for i in G.nodes]) * 0.2
v3 = adjacency_matrix[:,0] * 0.2
v4 = adjacency_matrix[:,0] / adjacency_matrix[:,0].sum()

for v in [v2, v3, v4]:
    assert np.array_equal(transition_vector, v)

print(transition_vector)
```

```python id="KH7FYe4-Ab14"
assert np.array_equal(transition_vector,
                      adjacency_matrix[:,0] / adjacency_matrix[:,0].sum())
```

<!-- #region id="JEcSzZX6Ab14" -->
Executing `M / M.sum(axis=0)` will divide each column of the adjacency matrix by the associated degree. The end-result is a matrix whose columns correspond to transition vectors. This matrix is referred to as a **transition matrix**.

**Computing a transition matrix**
<!-- #endregion -->

```python id="yTEsw624Ab14"
transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)
assert np.array_equal(transition_vector, transition_matrix[:,0])
```

<!-- #region id="SZy8-vERAb15" -->
Our transition matrix allows us to compute the traveling probability to every town in just a few lines of code. If we want to know the probability of winding up in _Town i_ after 10 stops, then we simply need to:

1. Initialize a vector `v` where `v` equals `np.ones(31) / 31`.
2. Update `v` to equal `transition_matrix @ v` over 10 iterations and return `v[i]`.

**Computing travel probabilities using the transition matrix**
<!-- #endregion -->

```python id="6tjxAShsAb15" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509579289, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5d677c27-32a1-4af0-c3ab-feb0c9bd27fe"
v = np.ones(31) / 31
for _ in range(10):
    v = transition_matrix @ v
    
for i in [12, 3]:
    print(f"The probability of winding up in Town {i} is {v[i]:.3f}.")
```

<!-- #region id="r84sRAK5Ab15" -->
We can model traffic-flow using a series of matrix multiplications. These multiplications serve as the basis for **PageRank centrality**. PageRank centrality is easy to compute, but not so easy to derive. Nonetheless, with basic probability theory, we can demonstrate why repeated `transition_matrix` multiplications directly yield the travel probabilities.

### Deriving PageRank Centrality from Probability Theory

`transition_matrix[i][j]` equals the probability of traveling from _Town j_ directly to _Town i_. Of course, that probability assumes that our car is located in _Town j_. Generally, if the probability of our current location is `p`, then the probability of travel from the current location `j` to new location `i` equals `p * transition_matrix[i][j]`. Suppose a car begins its journey in a random town, and travels one town over. What is the probability that the car will travel from _Town 3_ to _Town 0_ ? Well, the car can start the journey in anyone of 31 different towns. Consequently the probability of traveling from _Town 3_ to _Town 0_ is `transition_matrix[0][3] / 31`. 


**Computing a travel-likelihood from a random starting location**
<!-- #endregion -->

```python id="SFjuZRIJAb16" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509585472, "user_tz": -330, "elapsed": 769, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="498e6249-6f6f-4d1c-f38e-c25213b8c50d"
prob = transition_matrix[0][3] / 31
print("Probability of starting in Town 3 and driving to Town 0 is "
      f"{prob:.2}")
```

<!-- #region id="1mDr_t2SAb16" -->
The probability of that particular path is very low. However, there are multiple paths to reaching _Town 0_ from a random starting location. Lets print all non-zero instances of  `transition_matrix[0][i] / 31` for every possible _Town i_.

**Computing travel likelihoods of random paths leading to _Town 0_**
<!-- #endregion -->

```python id="nWI4iFvdAb16" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509585998, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c761831b-bbf0-452f-a1b7-cd565173c1bc"
for i in range(31):
    prob = transition_matrix[0][i] / 31
    if not prob:
        continue
        
    print(f"Probability of starting in Town {i} and driving to Town 0 is "
          f"{prob:.2}")
    
print("\nAll remaining transition probabilities are 0.0")
```

<!-- #region id="MxPvkHgxAb16" -->
Five different routes take us to _Town 0_. Each route has a different probability. The sum of these probabilities equals the likelihood of starting at any random town and traveling directly to _Town 0_.

**Computing the probability that the first stop is _Town 0_**
<!-- #endregion -->

```python id="iO28pVSBAb17" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509589646, "user_tz": -330, "elapsed": 2519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a0079fe-506d-4919-ece5-219b06e87746"
np.random.seed(0)
prob = sum(transition_matrix[0][i] / 31 for i in range(31))
frequency = np.mean([random_drive(num_stops=1) == 0
                     for _ in range(50000)])

print(f"Probability of making our first stop in Town 0: {prob:.3f}")
print(f"Frequency with which our first stop is Town 0: {frequency:.3f}")
```

<!-- #region id="08AEt-ciAb17" -->
Our computed probability is consistent with the observed frequency. It’s worth noting that the probability can be computed more concisely as vector dot product operation. We just need to run `transition_matrix[0] @ v`, where `v` is a 31-element vector whose elements all equal `1 / 31`.

**Computing a travel probability using a vector dot product**
<!-- #endregion -->

```python id="UB5bprJxAb17"
v = np.ones(31) / 31
assert transition_matrix[0] @ v == prob
```

<!-- #region id="dxLlsKxXAb17" -->
Executing `transition_matrix[i] @ v` will return the likelihood of making our first stop in _Town i_. We can compute this likelihood for every town by `[transition_matrix[i] @ v for i in range(31)`. Of course, this operation is equivalent to the matrix product between `transition_matrix` and `v`. Hence, `transition_vector @ v` returns all first-stop probabilities. 


**Computing all first stop probabilities**
<!-- #endregion -->

```python id="bbyQhTrWAb18" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509598988, "user_tz": -330, "elapsed": 1347, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b6aeed2-6ce3-400c-aff5-5e05fd30f848"
np.random.seed(0)
stop_1_probabilities = transition_matrix @ v
prob = stop_1_probabilities[12]
frequency = np.mean([random_drive(num_stops=1) == 12
                     for _ in range(50000)])

print('First stop probabilities:')
print(np.round(stop_1_probabilities, 3))
print(f"\nProbability of making our first stop in Town 12: {prob:.3f}")
print(f"Frequency with which our first stop is Town 12: {frequency:.3f}")
```

<!-- #region id="Qp7OswX4Ab18" -->
We’ve established that `transition_matrix @ v` returns a vector of first stop probabilities. Furthermore, we can easily show that `transition_matrix[i] @ stop_1_probabilities` returns a vector of second-stop probabilities. However,  `stop_1_probabilities` is  equal to `transition_matrix @ v`. Consequently, the second-stop probabilities are also equal to `transition_matrix @ transition_matrix @ v`. 

**Computing all second stop probabilities**
<!-- #endregion -->

```python id="MgTfHPctAb18" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509601647, "user_tz": -330, "elapsed": 2662, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7be61c77-d9b1-4dcc-92d8-9ece5540de87"
np.random.seed(0)
stop_2_probabilities = transition_matrix @ transition_matrix @ v
prob = stop_2_probabilities[12]
frequency = np.mean([random_drive(num_stops=2) == 12
                     for _ in range(50000)])

print('Second stop probabilities:')
print(np.round(stop_2_probabilities, 3))
print(f"\nProbability of making our second stop in Town 12: {prob:.3f}")
print(f"Frequency with which our second stop is Town 12: {frequency:.3f}")
```

<!-- #region id="6YEPggEfAb18" -->
We were able to derive our second-stop probabilities directly from our first stop probabilities. If we repeat our derivation, then we can easily show that `stop_3_probabilities` equals  `transition_matrix @ stop_2_probabilities`.  Of course, this vector also equals `M @ M @ M @ v`, where `M` is the transition matrix.  We can repeat this process to compute the fourth-stop probabilities, and then fifth-stop probabilities, and eventually the Nth stop-probabilities.  

**Computing the Nth-stop probabilities**
<!-- #endregion -->

```python id="R_dITN-AAb19" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509601648, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fef31fb2-cae8-4a9d-bc77-5cb68efef7ad"
def compute_stop_likelihoods(M, num_stops):
    v = np.ones(M.shape[0]) / M.shape[0]
    for _ in range(num_stops):
        v = M @ v
        
    return v
    

stop_10_probabilities = compute_stop_likelihoods(transition_matrix, 10)
prob = stop_10_probabilities[12]
print('Tenth stop probabilities:')
print(np.round(stop_10_probabilities, 3))
print(f"\nProbability of making our tenth stop in Town 12: {prob:.3f}")
```

<!-- #region id="76BdOPVGAb19" -->
As we’ve discussed, our iterative matrix multiplications form the basis for PageRank centrality. 

### Computing PageRank Centrality Using NetworkX
A function to compute PageRank centrality is included in NetworkX.

**Computing PageRank centrality using NetworkX**
<!-- #endregion -->

```python id="5zrhda32Ab19" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509609732, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a320cded-3135-4c5e-9881-55aa97bfee1b"
centrality = nx.pagerank(G)[12]
print(f"The PageRank centrality of Town 12 is {centrality:.3f}.")
```

<!-- #region id="CTD3-BW-Ab1-" -->
The printed PageRank value is 0.048, which is slightly lower than expected. This is due to the concept of _teleportation_ which is built into the PageRank algorithm. The algorithm assumes that a traveller will teleport to random node in 15% of all transitions. Imagine that in 15% of our town visits, we call for a helicopter, which takes us to a totally random town. Hence, 15% of the time, we’ll fly from _Town i_ to _Town j_ with a probability of `1 / 31`. In the remaining 85% of instances, we’ll drive from _Town i_ to _Town j_ with a probability of `transition_matrix[j][i]`. Consequently, the actual probability of travel from _i_ to _j_ equals the weighted mean of `transition_matrix[j][i]` and `1 / 31`.  Taking the weighted across all elements of the transition matrix will produce an entirely new transition matrix. Below, we’ll update our transition matrix, and recompute the _Town 12’s_ travel probability. 

**Incorporating randomized teleportation into our model**
<!-- #endregion -->

```python id="ZGSqW7-cAb1-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509610440, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2954c0be-7f90-4b16-bd53-e2e49534d306"
new_matrix = 0.85 * transition_matrix + 0.15 / 31
stop_10_probabilities = compute_stop_likelihoods(new_matrix, 10)

prob = stop_10_probabilities[12]
print(f"The probability of winding up in Town 12 is {prob:.3f}.")
```

<!-- #region id="nAuuezdmAb1_" -->
Our new output is consistent with the NetworkX result. Will that output remain consistent if we raise the number of stops from 10 to 1000? Lets find out.

**Computing the probability after 1000 stops**
<!-- #endregion -->

```python id="yIBcZm7aAb1_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509612234, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed10a0ef-8154-46b2-8285-6f5d04e92f1c"
prob = compute_stop_likelihoods(new_matrix, 1000)[12]
print(f"The probability of winding up in Town 12 is {prob:.3f}.")
```

<!-- #region id="WjCqyKcKAb2A" -->
The centrality remains consistent 0.048. Ten iterations are usually sufficient to achieve PageRank convergence. Transition / Markov matrices have mathematical properties. Markov matrices bridge graph theory with probability theory and matrix theory. However, that’s not all. Markov matrices can also be leveraged to cluster network data, using a procedure called **Markov clustering**.

## Community Detection Using Markov Clustering
<!-- #endregion -->

<!-- #region id="0QLN-21cAb2A" -->
Graph `G` represents a network of towns. Some of these fall into localized counties. Thus far, we’ve visualized the counties by mapping colors to individual county ids. What if we didn’t know the county ids? How would we identify the counties? Lets ponder this question by visualizing `G` without any sort of color mapping.

**Plotting `G` without county-based coloring**
<!-- #endregion -->

```python id="qatZ-1uyAb2A" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509789732, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1982a51b-dbc9-43ea-b892-4052fa81401e"
np.random.seed(1)
nx.draw(G)
plt.show()
```

<!-- #region id="CGyZz5coAb2A" -->
Our plotted graph has neither colors nor labels. Still, we can spot potential counties. They look like tightly connected clusters of nodes within the network. In graph theory, such clusters are formally referred to as **communities**. The process of uncovering communities in graphs is called **community detection**, or **graph clustering**. Some graph clustering algorithms depend on simulations of traffic-flow.

Suppose we drive from _Town i_ to _Town j_ and then to _Town k_. Based on our network structure, _Towns i_ and _k_ are more likely to be in the same county, We will confirm this statement shortly. However, first we’ll need to compute the probability of transition from _Town i_ to _Town k_ after two stops. This probability is called the **stochastic flow**, or **flow** for short.  We’ll need to calculate the flow between each pair of towns, and store that output in a **flow matrix**. The flow matrix is equal to `transition_matrix @ transition_matrix`. Hence, a random simulation should approximate the product of the transition matrix with itself.

**Comparing computed flow to random simulations**
<!-- #endregion -->

```python id="275UBNndAb2B"
np.random.seed(1)
flow_matrix = transition_matrix @ transition_matrix

simulated_flow_matrix = np.zeros((31, 31))
num_simulations = 10000
for town_i in range(31):
    for _ in range(num_simulations):
        town_j = np.random.choice(G[town_i])
        town_k = np.random.choice(G[town_j])
        simulated_flow_matrix[town_k][town_i] += 1

simulated_flow_matrix /= num_simulations
assert np.allclose(flow_matrix, simulated_flow_matrix, atol=1e-2)
```

<!-- #region id="_2kDeW9iAb2B" -->
We believe that the average flow between _Towns i_ and _j_ is higher if `G.nodes[i]['county_id']` equals `G.nodes[j]['country_id']`. We can confirm by separating all flows into two lists; `county_flows` and `between_county_flows`. The two lists will track intra-county flows and inter-county flows, respectively. We’ll plot a histogram for each of the lists. We’ll also compare their mean flow values. 


**Comparing intra and inter-county flow distributions**
<!-- #endregion -->

```python id="l42NCFy4Ab2B" colab={"base_uri": "https://localhost:8080/", "height": 334} executionInfo={"status": "ok", "timestamp": 1637509803936, "user_tz": -330, "elapsed": 1117, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0b5b901c-8b3e-439e-ec75-ff30f60453fa"
def compare_flow_distributions():
    county_flows = [] 
    between_county_flows = []
    for i in range(31):
        county = G.nodes[i]['county_id']
        nonzero_indices = np.nonzero(flow_matrix[:,i])[0]
        for j in nonzero_indices:
            flow = flow_matrix[j][i]
            
            if county == G.nodes[j]['county_id']:
                county_flows.append(flow)
            else:
                between_county_flows.append(flow)
    
    mean_intra_flow = np.mean(county_flows)
    mean_inter_flow = np.mean(between_county_flows)
    print(f"Mean flow within a county: {mean_intra_flow:.3f}")
    print(f"Mean flow between different counties: {mean_inter_flow:.3f}")
    
    threshold = min(county_flows)
    num_below = len([flow for flow in between_county_flows
                     if flow < threshold])
    print(f"The minimum intra-county flow is approximately {threshold:.3f}")
    print(f"{num_below} inter-county flows fall below that threshold.")
    
    plt.hist(county_flows, bins='auto',  alpha=0.5, 
             label='Intra-County Flow')
    plt.hist(between_county_flows,  bins='auto', alpha=0.5, 
             label='Inter-County Flow')
    plt.axvline(threshold, linestyle='--', color='k',
                label='Intra-County Threshold')
    plt.legend()
    plt.show()
    
compare_flow_distributions()
```

<!-- #region id="nE3dZG9HAb2B" -->
Flows below a threshold of approximately 0.04 are guaranteed to represent inter-county values. Of course, we’re only able to observe this threshold due to our advance knowledge of county identities. In a real-word scenario, the actual county ids would not be known. We’d be forced to assume that the cutoff is a low-value, like 0.01. Suppose we made that assumption with our data. How many non-zero inter-county flows are less than 0.01?

**Decreasing the separation threshold**
<!-- #endregion -->

```python id="-K_liMvaAb2C" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509803939, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ccfc9bff-d357-4c4f-b228-0adc9e61a9cb"
num_below = np.count_nonzero((flow_matrix > 0.0) & (flow_matrix < 0.01))
print(f"{num_below} inter-county flows fall below a threshold of 0.01")
```

<!-- #region id="ftwcIKYbAb2C" -->
None of the flow values fall below the stringent threshold of 0.01. We need to manipulate the flow distribution in order to exaggerate the difference between large and small values. This manipulation can be carried out with a simple process called **inflation**. Inflation is intended to influence the values of a vector while keeping its mean constant. We can inflate a vector `v` by executing `v2 = v **2` and subsequently dividing `v2` by `v2.sum()`.

**Exaggerating value differences through vector inflation**
<!-- #endregion -->

```python id="1f2uxiJKAb2C"
v = np.array([0.7, 0.3])
v2 = v ** 2
v2 /= v2.sum()
assert v.mean() == round(v2.mean(), 10)
assert v2[0] > v[0]
assert v2[1] < v[1]
```

<!-- #region id="ZHIck3N0Ab2D" -->
Like vector `v`, the columns of our flow matrix are vectors whose elements sum to 1. We can inflate each column by squaring its elements, and then dividing by the subsequent column sum.

**Exaggerating flow differences through vector inflation**
<!-- #endregion -->

```python id="-2PYJC0JAb2D" colab={"base_uri": "https://localhost:8080/", "height": 334} executionInfo={"status": "ok", "timestamp": 1637509803942, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcdc20c0-ec68-4384-cba5-56f8d9f64b45"
def inflate(matrix):
    matrix = matrix ** 2
    return matrix / matrix.sum(axis=0)

flow_matrix = inflate(flow_matrix)
compare_flow_distributions()
```

<!-- #region id="qLNWcfVYAb2E" -->
Our threshold has decreased from 0.042 to 0.012. However, it still remains above 0.01. How do we further exaggerate the difference between inter-county and intra-county edges? Well, setting the flow matrix to equal `inflate(flow_matrix @ flow_matrix)` will cause the threshold to drastically decrease. 

**Inflating the product of `flow_matrix` with itself**
<!-- #endregion -->

```python id="co40BGSKAb2E" colab={"base_uri": "https://localhost:8080/", "height": 334} executionInfo={"status": "ok", "timestamp": 1637509805743, "user_tz": -330, "elapsed": 1818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="34d82a75-8d0b-415a-9c24-ba0f780cda92"
flow_matrix = inflate(flow_matrix @ flow_matrix)
compare_flow_distributions()
```

<!-- #region id="9WgZ0Xy3Ab2E" -->
This serves as the basis for a network clustering algorithm known as the **Markov Cluster Algorithm**. It is also referred to as **Markov clustering**, or **MCL** for short. MCL is executed by running `inflate(flow_matrix @ flow_matrix)` over many repeating iterations. We’ll now attempt to execute MCL. To start, we’ll run `flow_matrix = inflate(flow_matrix @ flow_matrix)` across 20 iterations.

**Inflating the product of `flow_matrix` repeatedly with itself**
<!-- #endregion -->

```python id="6csCPHMPAb2F"
for _ in range(20):
    flow_matrix = inflate(flow_matrix @ flow_matrix)
```

<!-- #region id="ysAVA-6_Ab2F" -->
Certain edges in graph `G` should now have a flow of zero. We expect these edges to connect diverging counties. 

**Selecting suspected inter-county edges**
<!-- #endregion -->

```python id="XyXhweUIAb2F" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509805745, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="364da789-8241-497e-d7c0-b3284becc9db"
suspected_inter_county = [(i, j) for (i, j) in G.edges()
                         if not (flow_matrix[i][j] or flow_matrix[j][i])]
num_suspected = len(suspected_inter_county)
print(f"We suspect {num_suspected} edges of appearing between counties.")
```

<!-- #region id="MFAJQP2qAb2F" -->
Deleting the suspected edges from our graph should sever all cross-county connections. Consequently, only clustered counties should remain if we visualize the graph after edge deletion.

**Deleting suspected inter-county edges**
<!-- #endregion -->

```python id="fZCCz7sDAb2G" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509806510, "user_tz": -330, "elapsed": 772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa8345f7-62bd-4398-99ef-da50e34490d4"
np.random.seed(1)
G_copy = G.copy()
G_copy.remove_edges_from(suspected_inter_county)
nx.draw(G_copy, with_labels=True, node_color=node_colors)
plt.show()
```

<!-- #region id="5vKSZY2SAb2G" -->
All inter-county edges have been eliminated. Unfortunately, a few key intra-county edges have also been deleted. The problem is due to a minor error in our model. The model assumes that travelers can drive to neighboring towns, but it does not allow a traveller to remain in their current location. Adding self-loops to a graph will limit the unexpected model behaviour. Lets illustrate the impact of self-loops in a simple two-node adjacency matrix.

**Improving flow by adding self-loops**
<!-- #endregion -->

```python id="UJuEDATsAb2G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509806511, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="171b1e69-2647-4f9b-f19d-bec628407d07"
def compute_flow(adjacency_matrix):
    transaction_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)
    return (transaction_matrix @ transaction_matrix)[1][0]

M1 = np.array([[0, 1], [1, 0]])
M2 = np.array([[1, 1], [1, 1]])
flow1, flow2 = [compute_flow(M) for M in [M1, M2]]
print(f"The flow from A to B without self-loops is {flow1}")
print(f"The flow from A to B with self-loops is {flow2}")
```

<!-- #region id="qZ7pbJ3xAb2G" -->
Adding self-loops to graph `G` should limit inappropriate edge deletions. With this in mind, lets now define a `run_mcl` function.

**Defining an MCL function**
<!-- #endregion -->

```python id="6zzH0yLiAb2H" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509807276, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5d85397d-db7c-473e-ae86-af4a7b78c8a0"
def run_mcl(G):
    for i in G.nodes:
        G.add_edge(i, i)
    
    adjacency_matrix = nx.to_numpy_array(G)
    transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)
    flow_matrix = inflate(transition_matrix @ transition_matrix)
    
    for _ in range(20):
        flow_matrix = inflate(flow_matrix @ flow_matrix)
      
    G.remove_edges_from([(i, j) for i, j in G.edges()
                        if not (flow_matrix[i][j] or flow_matrix[j][i])])

G_copy = G.copy()
run_mcl(G_copy)
nx.draw(G_copy, with_labels=True, node_color=node_colors)
plt.show()
```

<!-- #region id="xHaZig1EAb2H" -->
Our graph has clustered perfectly into six secluded counties. In graph theory, such isolated clusters are referred to as **connected components**. In order to compute the full component of a node, it is sufficient to run `nx.shortest_path_length` on that node. The shortest path length algorithm will return only those nodes that are accessible within a clustered community. 

**Using path lengths to uncover a county cluster**
<!-- #endregion -->

```python id="6044lBcIAb2H" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509807277, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9d747a13-694d-4fa2-c14c-f3815ce2919e"
component = nx.shortest_path_length(G_copy, source=0).keys()
county_id = G.nodes[0]['county_id']
for i in component:
    assert G.nodes[i]['county_id'] == county_id
    
print(f"The following towns are found in County {county_id}:")
print(sorted(component))
```

<!-- #region id="bmW0FssoAb2H" -->
With minor modifications to the shortest path length algorithm, we can extract a graph’s connected components. This modified component algorithm is incorporated into NetworkX.

**Extracting all the clustered connected components**
<!-- #endregion -->

```python id="L5jMZZFTAb2I" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509807278, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af4491d7-b320-4d28-e55d-eaf5e496e036"
for component in nx.connected_components(G_copy):
    county_id = G.nodes[list(component)[0]]['county_id']
    print(f"\nThe following towns are found in County {county_id}:")
    print(component)
```

<!-- #region id="SOfYiV_aAb2I" -->
Our MCL implementation will not scale to very large networks. Further optimizations are required for successful scaling. These optimizations have been integrated into the external Markov Clustering library.

**Importing from the Markov Clustering library**
<!-- #endregion -->

```python id="EVx1KYzfMMkb"
!pip install -q markov_clustering
```

```python id="4V3vHhVGAb2I"
import markov_clustering
from markov_clustering import get_clusters, run_mcl
```

<!-- #region id="WIJLGayMAb2I" -->
Given an adjacency matrix `M`, we can efficiently execute Markov clustering by running `get_clusters(run_mcl(M))`.

**Clustering with the Markov Clustering library**
<!-- #endregion -->

```python id="uyYCFIjAAb2J" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509830839, "user_tz": -330, "elapsed": 564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="51619019-d5f9-42d1-da88-f30f2e76f12a"
adjacency_matrix = nx.to_numpy_array(G)
clusters = get_clusters(run_mcl(adjacency_matrix))

for cluster in clusters:
    county_id = G.nodes[cluster[0]]['county_id']
    print(f"\nThe following towns are found in County {county_id}:")
    print(cluster)
```

<!-- #region id="hgNoYNNnAb2J" -->
With Markov clustering, we can efficiently detect communities in community-structured graphs. This will prove useful when we search for groups of friends in social networks

## Uncovering Friend Groups in Social Networks

We can represent many processes as networks, including relationships between people. Within these **social networks**, nodes represent individual people. One famous social network is called _Zachery’s Karate Club_. That network can be loaded from NetworkX.

**Loading the Karate Club graph**
<!-- #endregion -->

```python id="xJCWjD8iAb2J" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509838542, "user_tz": -330, "elapsed": 634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dbd63bcf-8bf7-4796-a092-f44722690bc6"
G_karate = nx.karate_club_graph()
print(G_karate.nodes(data=True))
```

<!-- #region id="7p8DhDcwAb2J" -->
Our nodes track 34 people. Each node has a `club` attribute. That attribute is set to _Mr. Hi_ if the person joined Mr. Hi’s karate club. Otherwise, it’s set to _Officer_.  Lets go ahead and visualize the network.

**Visualizing the Karate Club graph**
<!-- #endregion -->

```python id="lgyc95EEAb2K" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509840758, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="20588cfd-183e-4d69-ecbb-3c60cf821980"
np.random.seed(2)
club_to_color = {'Mr. Hi': 'k', 'Officer': 'b'}
node_colors = [club_to_color[G_karate.nodes[i]['club']] 
               for i in G_karate]

nx.draw(G_karate, node_color=node_colors)
plt.show()
```

<!-- #region id="jQGp9ILMAb2K" -->
The Karate Club graph has a clear community structure. The colored clusters represent existing friend-groups. Lets extract these clusters using MCL.

**Clustering the Karate Club graph**
<!-- #endregion -->

```python id="HZJQ8f8iAb2K" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637509841287, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac546f02-ab93-4031-f5fa-a66a61476635"
adjacency_matrix = nx.to_numpy_array(G_karate)
clusters = get_clusters(run_mcl(adjacency_matrix))
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}:\n{cluster}\n")
```

<!-- #region id="lHrt9mE7Ab2L" -->
Two clusters have been outputted as expected. We’ll now re-plot the graph, while coloring each node by cluster id.

**Coloring the plotted graph by cluster**
<!-- #endregion -->

```python id="bXTDVrHEAb2L" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509844588, "user_tz": -330, "elapsed": 925, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f20172d-5c18-435d-d8c8-38182e4463fa"
np.random.seed(2)
cluster_0, cluster_1 = clusters
node_colors = ['k' if i in cluster_0 else 'b'
               for i in G_karate.nodes]

nx.draw(G_karate, node_color=node_colors)
plt.show()
```

<!-- #region id="2Nq-laPkAb2L" -->
Our clusters are nearly identical to the two splintered clubs. MCL has capably extracted the friend groups in the social network. The code below will illustrate how to color friend-groups in an automated manner.

**Coloring social graph clusters automatically**
<!-- #endregion -->

```python id="txAAoQGmAb2L" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637509845500, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="236db93f-b035-4915-f4ed-2568d0d152e5"
np.random.seed(2)
for cluster_id, node_indices in enumerate(clusters):
    for i in node_indices:
        G_karate.nodes[i]['cluster_id'] = cluster_id
        
node_colors = [G_karate.nodes[n]['cluster_id'] for n in G_karate.nodes]
nx.draw(G_karate, node_color=node_colors, cmap=plt.cm.tab20)
plt.show()
```

<!-- #region id="o4V8i1wTMXAU" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_9DTpdNAMXAU" executionInfo={"status": "ok", "timestamp": 1637509973252, "user_tz": -330, "elapsed": 3594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="60099675-4217-4380-f859-6eb72328d172"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="UnPSgsu6MXAV" -->
---
<!-- #endregion -->
