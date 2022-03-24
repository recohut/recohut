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

<!-- #region id="wK2-BSgmQ2Ai" -->
# Grocery Recommendation using Graph Network
> Building word2vec based Graph network using Instacart dataset and finding similar as well as neighbourhood items, and building a dash app

- toc: true
- badges: true
- comments: true
- categories: [Dash, App, NetworkX, Word2Vec, Graph, Retail, Visualization]
- image:
<!-- #endregion -->

<!-- #region id="bTWqNCOePwQ_" -->
## Setup
<!-- #endregion -->

```python id="ucPNt39z81kg"
!pip install -q dash dash-renderer dash-html-components dash-core-components
!pip install -q jupyter-dash
```

```python id="fVjCEEO786RX"
import re
import random
import pandas as pd
import numpy as np

import plotly.offline as py
import plotly.graph_objects as go

import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

from gensim.models import Word2Vec

from tqdm.notebook import tqdm

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State, ALL
```

```python colab={"base_uri": "https://localhost:8080/"} id="fl0gfJeX-euv" outputId="7e21aa01-98ca-4010-f9c2-b2121f773e77"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

<!-- #region id="_p_wYeySP1d1" -->
## Loading data
<!-- #endregion -->

<!-- #region id="-7wt1IwpGWlm" -->
<!-- #endregion -->

```python id="7-2nOHbi9DL8"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c instacart-market-basket-analysis
!unzip /content/instacart-market-basket-analysis.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="m8opruUi9Yoo" outputId="e68def05-38ae-4eef-bcd8-c6c64705fa92"
!sudo apt-get install tree
!tree . -L 1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8UHM52Al_RQW" outputId="015ca52d-062b-4e5d-e1b5-4ce20bb1d64b"
!unzip -qqo /content/order_products__train.csv.zip
train_df = pd.read_csv('order_products__train.csv')
train_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="PiFy5-DL_RL0" outputId="88f94165-b5fc-4d86-90cf-5711cafaff86"
!unzip -qqo /content/products.csv.zip
products_df = pd.read_csv('products.csv')
products_df.head()
```

<!-- #region id="Px03Qre9P5sO" -->
## Preprocessing
<!-- #endregion -->

```python id="Jk70YSgM9wDV"
def return_dfs(train_df, products_df, train_percent=0.1, products_cutoff=0,
               orders_q1=5, orders_q2=9):
    ''' Function that returns two dataframes for 2 segments of users based basket size
    Args:   train_file - the training dataframe
            products_file - the products dataframe
            train_percent - percentage of the train file sampled for this (smaller % makes viz possible)
            products_cutoff - only products appearing MORE often than this are included
            orders_q1 - first cutoff point for number of items in a basket
            orders_q2 - second cutoff point for number of items in a basket
    
    '''
    orders = train_df[['order_id', 'product_id']].copy()
    products = products_df.copy()
    
    # Get a wide range of orders
    order_ids = orders.order_id.unique()
    # Select a sample of the orders
    order_ids = random.sample(set(order_ids), int(len(order_ids)*train_percent))
    # Reduce the size of the initial orders data
    orders = orders[orders['order_id'].isin(order_ids)]
    
    
    # Take a look at the distribution of product counts
    counts = orders.groupby('product_id').count()
    counts.rename(columns = {'order_id':'count'}, inplace = True)
    counts.reset_index(inplace = True)
    # Remove the products occuring less often that products_cutoff
    product_ids = counts.product_id[counts['count'] > products_cutoff]
    
    # Filter for baskets of a certain size
    counts = orders.groupby('order_id').count()
    counts.rename(columns = {'product_id':'count'}, inplace = True)
    counts.reset_index(inplace = True)
    # Only keep baskets below orders_q1 size and between orders_q1 and orders_q2 size
    order_ids_Q1 = counts.order_id[counts['count'] <= orders_q1]
    order_ids_Q2  = counts.order_id[(counts['count'] <= orders_q2) & (counts['count'] > orders_q1)]
    
    # Create two dataframes for the orders
    orders_small = orders[orders['order_id'].isin(order_ids_Q1)]
    orders_small = orders_small[orders_small['product_id'].isin(product_ids)]
    orders_small = orders_small.merge(products.loc[:, ['product_id', 'product_name']], how = 'left')
    # To simplify what the orders look like, I've replaced 'bag of organic bananas' with just 'bananas'
    orders_small['product_name'] = orders_small['product_name'].replace({'Bag of Organic Bananas': 'Banana'})
    orders_small['product_name'] = orders_small['product_name'].str.replace('Organic ', '')

    orders_large = orders[orders['order_id'].isin(order_ids_Q2)]
    orders_large = orders_large[orders_large['product_id'].isin(product_ids)]
    orders_large = orders_large.merge(products.loc[:, ['product_id', 'product_name']], how = 'left')

    orders_large['product_name'] = orders_large['product_name'].replace({'Bag of Organic Bananas': 'Banana'})
    orders_large['product_name'] = orders_large['product_name'].str.replace('Organic ', '')
    
    return orders_small, orders_large, order_ids_Q1, order_ids_Q2
```

```python id="IY2I8xqY-PiJ"
orders_small, orders_large, order_ids_Q1, order_ids_Q2 = return_dfs(train_df, products_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="4QgS2UpsAXi6" outputId="f794a85b-aa76-407e-cf20-18b4f46a361c"
orders_small.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="zZfD3l2YAZWc" outputId="05615a47-b34b-4073-df56-7e1c79fdb6b2"
orders_large.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="DLAMF-ALAe3W" outputId="3874060f-31ab-4c14-c405-897255ad121f"
order_ids_Q1
```

<!-- #region id="QIZLIUYHAq9r" -->
## Processing the Data for NetworkX
Here we need to create tuples comprising the paired items in the data so that we can build the graph. This code creates two sets of data, one for the "small" baskets and one for the "large" (although they're still quite small) baskets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 115, "referenced_widgets": ["a495a0cf7928479291950a2f7f530734", "c3ce9ef5d8bd4478949f7d21b2df356d", "356f6a0c0bad461babd889941b728924", "8ebfaa8911d24f1eb8ceb66666d12d99", "5a57039da7644e16b5c097cb2c95e0dd", "974e635aab5e4ec5a8fc80bbfe7a4383", "1d48322154fd460d8c9b87ac897df9bf", "5c4596f1ab044a6da4ec47e08ac60bc4", "e949cb02ed1a4971b04545f7436db8b8", "fc75c12c7e184a4fbec57a9569691dcf", "23afc4c497154f02be04e93f85cd7842", "66b4e0f46b3b43c1b63b1e5038783979", "664304d4f06a457d9d713f3c625d1d83", "3e68e2ff48d84d4d9bc752ab023d85b6", "2a52ba2c014843698de8b1bd6e766d4b", "1876c679da2143f7874894593742baaa"]} id="e6a5M5lnAhqu" outputId="c43c8838-b2dd-4f1d-f66c-84bb80e87dac"
paired_products_small = []

# Create the pairwise product combinations
for order_id in tqdm(order_ids_Q1):
    tmp_df = orders_small[orders_small['order_id'] == order_id]
    paired_products_small.extend(list(itertools.combinations(tmp_df.iloc[:, 2], 2)))
    
paired_products_large = []

# Create the pairwise product combinations
for order_id in tqdm(order_ids_Q2):
    tmp_df = orders_large[orders_large['order_id'] == order_id]
    paired_products_large.extend(list(itertools.combinations(tmp_df.iloc[:, 2], 2)))
    
counts_small = collections.Counter(paired_products_small)

counts_large = collections.Counter(paired_products_large)

food_df_small = pd.DataFrame(counts_small.most_common(1000),
                      columns = ['products', 'counts'])


food_df_large = pd.DataFrame(counts_large.most_common(4000),
                      columns = ['products', 'counts'])
```

```python id="uEFzEPO4BGJy"
# Turn one of the dataframes into a dictionary for processing into a graph
d = food_df_small.set_index('products').T.to_dict('records')
# d = food_df_large.set_index('products').T.to_dict('records')
```

```python colab={"base_uri": "https://localhost:8080/"} id="YAPzMOhiBtKb" outputId="4f125ad9-400b-4e22-dcbd-5ca381bdc238"
dict(list(d[0].items())[0:10])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Dpt-jUMBBisR" outputId="eccf5447-701c-4e4a-a548-785559c2a61c"
# Create and populate the graph object
G = nx.Graph()

for key, val in d[0].items():
    G.add_edge(key[0], key[1], weight = val)

# Take a look at how many nodes there are in the graph; too many and it's uncomfortable to visualise
nodes = list(G.nodes)
len(nodes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ImVVMWVNCQ_H" outputId="c1816646-260c-4897-f6bc-7e16bf7a2206"
# Prune the plot so we only have items that are matched with at least two others
for node in nodes:
    try:
        if G.degree[node] <= 1:
            G.remove_node(node)
    except:
        print(f'error with node {node}')

nodes = list(G.nodes)
len(nodes)
```

```python id="cPTbLMiuCnQe"
with open('large_graph.pickle', 'wb') as handle:
    pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region id="a9kSDVEuCTVH" -->
## Build the Word2Vec model
This section of the code focuses on building the Word2Vec-like embeddings for the nodes in the network using the Deep Walk procedure.
<!-- #endregion -->

```python id="WWw8cEGsCZQV"
# Read the pickle in 
with open('large_graph.pickle', 'rb') as f:
    G_large = pickle.load(f)
```

```python id="fUvy3GNrHa5E"
def load_graph(segment):
    ''' Function that creates the graph of the graph based on the min number of edges
    Args: segment: indicates which segment: 0, 1, 2  to choose -> int
    Returns: graph and pos objects
    '''
    ### Load the data up
    segments = ['small_graph.pickle', 'med_graph.pickle', 'large_graph.pickle']

    with open(segments[segment], 'rb') as f:
        G = pickle.load(f)

    pos = nx.spring_layout(G)

    return pos, G
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["2908997c4929430f918dd0c63608b106", "a72e4ad4bfc347d3afee27b1f19ea1fa", "53e606b87aa14214adb83d0a7e152ac0", "a5898e01abb242baad7ff68221106c26", "a254c644e16d4725bf91cc0a2288d1e1", "dee2b7a161564a90989ac4ce59e6e501", "bb26e1e4c01f48dc93f1a8efcf14f0e6", "761feb5caa81411ca7b4b4f9cd45a194"]} id="O06nFr4ZChJ0" outputId="e754847b-9388-4d9e-9ad1-612a9b1cff3b"
# Build a dictionary containing the weights of the edges; doing it this way saves a LOT of time in doing the probabilistic
# random walks in the next steps
weights = {}
for node in tqdm(G_large.nodes()):
    w_ = []
    for nodes in list(G_large.edges(str(node))):
        w_.append(G_large.get_edge_data(nodes[0], nodes[1])['weight'])
    weights[node]=w_    
```

```python id="-zjlAG4CC0LU"
def random_walk(graph, node, weighted=False, n_steps = 5):
    ''' Function that takes a random walk along a graph'''
    local_path = [str(node),]
    target_node = node
    
    # Take n_steps random walk away from the node (can return to the node)
    for _ in range(n_steps):
        neighbours = list(nx.all_neighbors(graph, target_node))
        # See the difference between doing this with and without edge weight - it takes many, many times longer
        if weighted:
            # sample in a weighted manner
            target_node = random.choices(neighbours, weights[target_node])[0]
        else:
            target_node = random.choice(neighbours)
        local_path.append(str(target_node))
        
    return local_path
```

<!-- #region id="i-Cm6L9IDQFU" -->
Now we do the random walk and then we create the node embeddings


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["86a1e7530696423e9ccba1dacc5c435d", "4a7ddd8095874012841f9e7f12447b0e", "a073ac68471a431e98a14fed6f121636", "82b43f61286e4edb9fab53cf1c625de8", "10afc6afd6d34cb8b282d25af6bfb261", "b1cff006a21c4323a0e4cead86772f00", "9bca0d35e0ef4b07bbc001d787c17a39", "7a5d3adfe202447a8bc49a2eaa962008"]} id="Cz_hDXgIDOYd" outputId="269f9f0a-e48e-47f8-8037-676ad2580063"
walk_paths_weighted = []

i = 0
for node in tqdm(G_large.nodes()):
    for _ in range(10):
        walk_paths_weighted.append(random_walk(G_large, node, weighted=True))
```

```python colab={"base_uri": "https://localhost:8080/"} id="keBb_QWjDTiK" outputId="c7bfee96-752b-4b8d-ed4f-3deb9e205e07"
# Instantiate the embedder
embedder_weighted = Word2Vec(window = 4, sg=1, negative=10, alpha=0.03, min_alpha=0.0001, seed=42)
# Build the vocab
embedder_weighted.build_vocab(walk_paths_weighted, progress_per=2)
# Train teh embedder to build the word embeddings- this takes a little bit of time
embedder_weighted.train(walk_paths_weighted, total_examples=embedder_weighted.corpus_count, epochs=20, report_delay=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="LoSzE80JDbjQ" outputId="fb4f547c-1f42-4c41-a7e3-c58959d66624"
some_random_words = [list(embedder_weighted.wv.vocab.keys())[x] for x in np.random.choice(len(embedder_weighted.wv.vocab),10)]
some_random_words
```

```python colab={"base_uri": "https://localhost:8080/"} id="GoYTqbEmDYnT" outputId="58347235-533c-4ce7-9e98-aa1921aa55bc"
x_ = embedder_weighted.wv.most_similar(some_random_words[0], topn=10)
[i[0] for i in x_]
```

```python colab={"base_uri": "https://localhost:8080/"} id="pW7e8wTdFhu0" outputId="1f8152b9-779b-4280-e2ed-e99a874b23f0"
x_ = embedder_weighted.wv.most_similar(some_random_words[1], topn=10)
[i[0] for i in x_]
```

```python id="QCDKucnYFn3Y"
## Save and/or load the embedding objects

# with open('embedder_weighted.pickle', 'wb') as f:
#     pickle.dump(embedder_weighted, f)

# with open('embedder_weighted.pickle', 'rb') as f:
#     embedder_weighted = pickle.load(f)
```

<!-- #region id="70wpnnpsQQMd" -->
## Finding similar items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sr-kgFnNFy97" outputId="42b424c2-1865-4a60-e02e-a0195575f5be"
nodes = [node for node in G.nodes()]
pos = nx.spring_layout(G)
len(nodes)
```

```python id="Ql6pYoFXF7pD"
def similar_embeddings(source_node, topn):
    ''' Function that returns the top ncounts most similar items using embeddings'''
    most_similar = embedder.wv.most_similar(source_node, topn=topn)
    return [i[0] for i in most_similar]
```

```python id="QMo4v_xqH1sf"
def find_ingredient(nodes, ingredient="Pear"):
    ''' Function that returns the closet match to an ingredient in the graph
    Args: ingredient: the ingredient you want to find -> str
          nodes: a list of the nodes in the graph -> list
    Returns: a list of the closest ingredients found
    '''
    ingredients = []

    for node in nodes:
        # This does a string-like search for the ingredient/item in each node
        # So ingredient="Pear" can return "Pear Jam", "Potato and Pear Soup" etc.
        if ingredient in node:
            ingredients.append(node)

    return ingredients
```

```python colab={"base_uri": "https://localhost:8080/"} id="NtUUmE6IH3V_" outputId="be3930b1-5d4b-4f17-9bdf-753aa3713aa7"
find_ingredient(nodes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zha_f54kH_v4" outputId="bf4ae9da-af2b-4ab5-dc33-fa10acb8a7d3"
find_ingredient(nodes, ingredient='Onion')
```

<!-- #region id="wT5w70jrQhcl" -->
## Finding neighbour items
<!-- #endregion -->

```python id="HtSwrjRMIgBw"
# Traverse the graph by selecting the most weighted item
def get_neighbours(G, item, topn=10):
    ''' Function that returns the neighbours of a node
    Args: G - the netwowrkx Graph object
          item: the start node for searching -> str
          topn: number of neighbours to return -> int
    Returns: a list of grocery items occuring in a basket together
    '''

    # items = list(G.neighbors(item))
    weights = {}
    # Get all the neighbours of a node and sort them by their edge weight
    for nodes in list(G.edges(str(item))):
        weights[nodes[1]] = G.get_edge_data(nodes[0], nodes[1])['weight']
    weights_sorted = {k: v for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)}
    # Filter so we just have the topn items
    items = list(weights_sorted.keys())[0:topn]

    return items
```

<!-- #region id="DifcqCnhFyYi" -->
## Using Plotly to make the graph interactive
<!-- #endregion -->

```python id="Iirc0sYKG10x"
def create_graph_display(G, pos):
    ''' Function for displaying the graph; most of this code is taken
        from the Plotly site
    Args:   G - networkx graph object
            pos - positions of nodes
    '''
    nodes = [node for node in G.nodes()]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext="10",
        text="",
        textfont=dict(
            family="sans serif",
            size=11
        ),
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=False,
            color=[],
            size=8,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    # Update the text displayed on mouse over
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{nodes[node]}: {str(len(adjacencies[1]))} connections')

    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_text

    return node_trace, edge_trace
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="U0WGBsllF6L5" outputId="a5c6d605-7c29-46c9-fb37-6c853c017de6"
node_trace, edge_trace = create_graph_display(G, pos)

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Graph of shopping cart items',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="some text",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


fig.show()
```

<!-- #region id="lv8kwW6FKsGv" -->
## Building the Dash App
<!-- #endregion -->

<!-- #region id="WtYDHUpUKu7E" -->
### Define the app
<!-- #endregion -->

```python id="gqQ05tatKyyb"
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='Groceries on a graph'
list_dict = [{}]
```

<!-- #region id="2ijCXElvK8zZ" -->
### Load an initial Graph
<!-- #endregion -->

```python id="ec1kwyaqK21M"
with open('large_graph.pickle', 'rb') as f:
    G_init = pickle.load(f)
nodes = list(G_init.nodes)
```

<!-- #region id="i5fHFu5CK_zf" -->
### Create a global button tracker
<!-- #endregion -->

```python id="HYMOSW1ULCDI"
BUTTON_CLICKED = None
button_style = {'margin-right': '5px',
               'margin-left': '5px',
               'margin-top': '5px',
               'margin-bottom': '5px'}
```

<!-- #region id="zbxPYineLGuH" -->
### Define Layout
<!-- #endregion -->

```python id="Fr3rSoYKLH9T"
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '0px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#1a1a1a',
    'color': 'white',
    'padding': '6px'
}
```

<!-- #region id="SY0407OtLQYT" -->
> Note: Layout comprises two tabs - one for viewing of the graph and the other for making a shopping list
<!-- #endregion -->

```python id="AkTAMMDoLqpU"
app.layout = html.Div([
    html.Div(className='row', children=[
        html.H1(children='Grocery Graph Network')
        ], style={'textAlign': 'center',
                  'backgroundColor': '#1a1a1a',
                  'color': 'white'}),
    html.Div([
    ## Show the tabs
    dcc.Tabs(id='tabs-example', value='tab-1',
             children=[
        dcc.Tab(label='Explore Network Graph', value='tab-1',
                style=tab_style,
                selected_style=tab_selected_style),
         dcc.Tab(label='Shopping List Builder', value='tab-2',
                 style=tab_style,
                 selected_style=tab_selected_style)
        ], style=tabs_styles
        )
    ]
    ),
    html.Div(id='tabs-output')
])
```

<!-- #region id="HG4H_IZrLtZ8" -->
### Callback for selecting/changing tabs
<!-- #endregion -->

```python id="58sbECR0L2Hy"
@app.callback(Output('tabs-output', 'children'),
              Input('tabs-example', 'value'))
def render_content(value):
    if value == 'tab-1':
        return display_network_graph()
    elif value == 'tab-2':
        return display_shopping_list()
```

<!-- #region id="X5MwKYV9METD" -->
### Function for displaying the network graph

<!-- #endregion -->

```python id="zGdZG9hSMHIT"
def display_network_graph():
    ''' Function that displays the network tab'''

    # Setup a dropdown menu for the inputs to the graph
    dd_segment = dcc.Dropdown(
        id='dd_segment',
        className='dropdown',
        options=[{'label': 'Small', 'value': 0},
                 {'label': 'Medium', 'value': 1},
                 {'label': 'Large', 'value': 2}],
        value=2
    )
    # Create a div for the input settings, which includes the dropdown declared above
    input_settings = html.Div([
        html.Div(className='row', children=[
                    html.Div(className='col',
                        children=[
                            html.H4("Select Segment"),
                            html.P("Select a segment, named according to basket size"),
                            dd_segment
                        ],
                        style={'width': '30%', 'display': 'inline-block'}
                    )
            ]),
         # Display the main graph, with loading icon whilst loading
         html.Div(className='row', children=[
                    html.Div(children=[
                        dcc.Loading(id='loading-icon',
                                    children=[
                                        dcc.Graph(id='graph-graphic')
                                    ])
                    ])
                  ])
    ])

    return input_settings
```

<!-- #region id="nAUj6pO2MR07" -->
### Function for displaying the items on the shopping recommender/list tab
<!-- #endregion -->

```python id="cMHGrTwNMQOw"
def display_shopping_list():
    return html.Div([
        html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.P("Description: Search for items to start building your shopping list. \
                Once you've selected an item, similar items will be recommended to you")
            ])
        ]),
        html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                html.Div(className='row', children=[
                    # Search box in here
                    html.H3("Search items"),
                    dcc.Input(
                        id='item_search',
                        type='search',
                        placeholder='Search Shopping Items',
                        debounce=True,
                        value='Pears'
                    ),
                    html.P(""),
                    # Radio items to choose how
                    html.P("Method for making recommendations: "),
                    dcc.RadioItems(id='sim-radio',
                                   options=[{'label': 'Similar', 'value': 'similar'},
                                            {'label': 'Neighbours', 'value': 'neighbours'}],
                                   value='similar',
                                   labelStyle={'display': 'inline-block'}),

                    html.P(""),
                    html.P(id='explainer',
                           children=["The items below are the closest that match your search"],
                           style={'font-weight': 'bold'}),
                    html.P("")
                ]
                ),
                 # Container that loads the items closest to what you searched for or
                 # recommended items
                 html.Div(className='row', id='button-container', children=[]

            )],
            ),
            html.Div(className='six columns', children=[
                html.Div(className='row', children=[
                    html.Div(className='six columns', children=[html.H3("Your Shopping List")])
                ]),
                # Dyanamic shopping list is built here
                html.Div(className='row', id='shopping-list-container', children=[]
                    )
                ],

             )

        ])
    ])
```

<!-- #region id="qdHO4M99Mbcy" -->
### Callback for displaying similar/neighbor items (tab 2)
<!-- #endregion -->

```python id="5gb97cDeMism"
@app.callback(
    Output('button-container', 'children'),
    Output('shopping-list-container', 'children'),
    Output('explainer', 'children'),
    [Input('item_search', 'value'),
    Input({'type': 'button', 'index': ALL}, 'n_clicks'),
    Input('sim-radio', 'value')],
    [State('button-container', 'children'),
    State('shopping-list-container', 'children'),
    State('explainer', 'children')]
)
def display_search_buttons(item, vals, sim_val, buttons, shopping_list_items, explainer_text):
    ''' Function that runs all of the updates on for the shopping list'''
    ctx = dash.callback_context
    # Make the text in the same format as the times
    item = item.title()

    # If something has been triggered and it's the page loading or it's an item searched for
    # then load the items as per what was searched for
    if ctx.triggered is not None and \
            ctx.triggered[0]['prop_id'] == '.' or \
            ctx.triggered[0]['prop_id'] == 'item_search.value':
        buttons = []
        # Search for shopping items based on the item typed in the search bar
        # and create buttons
        shopping_items = find_ingredient(nodes, item)
        # Sort the items by length to make the display look nicer
        shopping_items.sort(key=len)
        counter = 0
        for i, it in enumerate(shopping_items):
            new_button = html.Button(
                f'{it}',
                id={'type': 'button',
                    'index': it
                },
                n_clicks=0,
                style=button_style
            )
            counter += 1
            buttons.append(new_button)
            # Stop too many items from being added
            if counter > 20:
                break
            explainer_text = r"""The items below are the closest that match your search"""

    # Check a button was clicked & that it's at least the first click (no auto clicks when the page loads)
    # & it's not the search input being searched in and it's not the radio button being checked
    elif ctx.triggered and ctx.triggered[0]['value'] != 0 and \
            ctx.triggered[0]['value'] is not None and \
            ctx.triggered[0]['prop_id'] != 'item_search.value' and \
            ctx.triggered[0]['prop_id'] != 'sim-radio.value':

        # Get the name of the grocery item
        button_clicked = re.findall(r':"(.*?)"', ctx.triggered[0]['prop_id'])[0]
        # track the button clicked for the next elif
        global BUTTON_CLICKED
        BUTTON_CLICKED = button_clicked
        # Add it to the shopping list
        new_item = html.P(
            f'{button_clicked}'
        )
        shopping_list_items.append(new_item)

        # Erase the list of ingredients and present similar ingredients by searching the graph
        buttons, explainer_text = recommend_groceries(button_clicked, sim_val)

    # Check if someone does something and if that something is changing the value on the
    # similarity measure radio button and
    elif ctx.triggered is not None and \
            BUTTON_CLICKED is not None and \
            ctx.triggered[0]['prop_id'] == 'sim-radio.value':
        buttons, explainer_text = recommend_groceries(BUTTON_CLICKED, sim_val)

    return buttons, shopping_list_items, explainer_text
```

<!-- #region id="2m5kNGehMt7T" -->
### Function that returns a list of recommended groceries
<!-- #endregion -->

```python id="3R_ePlVtMsCH"
def recommend_groceries(button_clicked, sim_val):
    ''' Function that returns a list of recommended groceries
    Args:  button_clicked - the button (item) clicked by the user
            sim_val - the type of similarity the user wants for recommendations
                    either 'similar' or 'neighbours'
    Returns: a list of recommended items for the user
    '''
    buttons = []
    # Get recommendations based on the similarity method chosen by the user
    if sim_val == "similar":
        recommendations = similar_embeddings(button_clicked, 10)
    else:
        recommendations = get_neighbours(G_init, button_clicked)
    # Update the explainer so the user knows what's going on
    explainer_text = r"""These items are recommended based on the last item added to your basket"""

    # Stop a system-hanging number of recommendations being added
    if len(recommendations) > 20:
        recommendations = recommendations[:20]
    # Sort the recommendations based on how many words each comprises; this makes the display
    # look nicer
    recommendations.sort(key=len)
    # Add the recommended items to the buttons for adding to the shopping list
    for i, it in enumerate(recommendations):
        new_button = html.Button(
            f'{it}',
            id={'type': 'button',
                'index': it
                },
            n_clicks=0,
            style=button_style
        )
        buttons.append(new_button)
    return buttons, explainer_text
```

<!-- #region id="hJVC9bZ0Mw1E" -->
### Callback for updating the graph network

> Note: Graph display will update when the user changes the segment size
<!-- #endregion -->

```python id="enwvE-sGIobl"
@app.callback(
    Output('graph-graphic', 'figure'),
    [Input('dd_segment', 'value')]
)
def update_graph(segment):
    ''' Function to load a pre-computed graph network based on the segment selected by
        the user in the dropdown
    Args:   segment - the segment selected by the user in the dropdown
    '''
    # Load the graph data and create the nodes and elements needed for display
    pos, G = load_graph(segment=segment)

    node_trace, edge_trace = create_graph_display(G, pos)
    # Display the graph
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} id="eFPsZbX4KARM" outputId="297dd5f2-13d5-4b19-8d77-062c78cb8656"
app.run_server(mode='external')
```

<!-- #region id="uYilo8-GNwKC" -->
### Analyzing callback map

This is retrieved from the dash layout
<!-- #endregion -->

<!-- #region id="jxdxHzufN4k0" -->
<!-- #endregion -->

<!-- #region id="geSxcD6AO-Xk" -->
### Tab 1 visual
<!-- #endregion -->

<!-- #region id="CTvMQCszPERo" -->
<!-- #endregion -->

<!-- #region id="dQg9jcIOPoBx" -->
### Tab 2 visual
<!-- #endregion -->

<!-- #region id="_S3pls0HN365" -->
<!-- #endregion -->
