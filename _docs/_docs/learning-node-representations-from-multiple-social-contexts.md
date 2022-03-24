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

<!-- #region id="QxZ7RARQFPCc" -->
# SPLITTER: Learning Node Representations from Multiple Contexts
<!-- #endregion -->

<!-- #region id="0ygKEzgBFB4X" -->
<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0097c9b8-cf21-4e70-a7e6-47c1608e05d9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T155845Z&X-Amz-Expires=86400&X-Amz-Signature=51c89f9f8e7484cb49ef55d55b955d2e96ebea5f0011e28afffe718c43604d51&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22' width=50%></p></center>
<!-- #endregion -->

<!-- #region id="VF3ME_7kAYSw" -->
## CLI Run
<!-- #endregion -->

```python id="Kio4HciT-LYr"
!git clone https://github.com/benedekrozemberczki/Splitter.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="J7GITBzn_XcE" executionInfo={"status": "ok", "timestamp": 1633966545412, "user_tz": -330, "elapsed": 490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="586e67c6-0a36-4ddc-a20c-46d3e56dbe9c"
%%writefile requirements.txt
networkx==1.11
tqdm==4.28.1
numpy==1.15.4
pandas==0.23.4
texttable==1.5.0
scipy==1.1.0
argparse==1.1.0
torch==1.1.0
gensim==3.6.0
```

```python id="sh8xmRO4_tVP"
!pip install -r requirements.txt
```

```python id="7E1iM-o6_vxk"
!pip install -q -U numpy networkx
```

```python colab={"base_uri": "https://localhost:8080/"} id="d_5BuifZ_R72" executionInfo={"status": "ok", "timestamp": 1633970608440, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5daf7ca5-22a3-4036-b83d-3cb93cf237aa"
%cd Splitter
```

```python colab={"base_uri": "https://localhost:8080/"} id="C8jsSul2_S4z" executionInfo={"status": "ok", "timestamp": 1633970303424, "user_tz": -330, "elapsed": 2920179, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6dce6d03-ef35-4b55-d850-629a393d67c1"
!python src/main.py
```

<!-- #region id="buflXVGiAziM" -->
## API Exploration
<!-- #endregion -->

```python id="1YLyLPuEPpy3"
!pip install -q -U tqdm
```

<!-- #region id="D6pFeyggDcr2" -->
### Walker
<!-- #endregion -->

```python id="7GzZAdF5DcqP"
"""DeepWalker class."""

import random
import numpy as np
from tqdm.notebook import tqdm
import networkx as nx
from gensim.models import Word2Vec

class DeepWalker(object):
    """
    DeepWalk node embedding learner object.
    A barebones implementation of "DeepWalk: Online Learning of Social Representations".
    Paper: https://arxiv.org/abs/1403.6652
    Video: https://www.youtube.com/watch?v=aZNtHJwfIVg
    """
    def __init__(self, graph, args):
        """
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        while len(walk) < self.args.walk_length:
            nebs = [n for n in nx.neighbors(self.graph, walk[-1])]
            if len(nebs) == 0:
                break
            walk.append(random.choice(nebs))
        return walk

    def create_features(self):
        """
        Creating random walks from each node.
        """
        self.paths = []
        for node in tqdm(self.graph.nodes()):
            for _ in range(self.args.number_of_walks):
                walk = self.do_walk(node)
                self.paths.append(walk)

    def learn_base_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.paths = [[str(node) for node in walk] for walk in self.paths]

        model = Word2Vec(self.paths,
                         size=self.args.dimensions,
                         window=self.args.window_size,
                         min_count=1,
                         sg=1,
                         workers=self.args.workers,
                         iter=1)

        self.embedding = np.array([list(model[str(n)]) for n in self.graph.nodes()])
        return self.embedding
```

<!-- #region id="1fxNmdBwDcoo" -->
### Ego Splitter
<!-- #endregion -->

```python id="5n9hWT8KDcmo"
"""Ego-Splitter class"""

# import community
import community.community_louvain as community
import networkx as nx
from tqdm.notebook import tqdm


class EgoNetSplitter(object):
    """An implementation of `"Ego-Splitting" see:
    https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    From the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters".
    The tool first creates the egonets of nodes.
    A persona-graph is created which is clustered by the Louvain method.
    The resulting overlapping cluster memberships are stored as a dictionary.
    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
    """
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        Args:
            node: Node ID for egonet (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        print("Creating egonets.")
        for node in tqdm(self.graph.nodes()):
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {p: n for n in self.graph.nodes() for p in self.personalities[n]}

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.
        Args:
            edge: Edge being mapped to the new identifiers.
        """
        return (self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]])

    def _create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("Creating the persona graph.")
        self.persona_graph_edges = [self._get_new_edge_ids(e) for e in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        print("Clustering the persona graph.")
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self, graph):
        """
        Fitting an Ego-Splitter clustering model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions
```

<!-- #region id="seS2ZumdDcja" -->
### Splitter
<!-- #endregion -->

```python id="pdKSnuoADcfX"
"""Splitter Class."""

import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm.notebook import trange
# from walkers import DeepWalker
# from ego_splitting import EgoNetSplitter

class Splitter(torch.nn.Module):
    """
    An implementation of "Splitter: Learning Node Representations
    that Capture Multiple Social Contexts" (WWW 2019).
    Paper: http://epasto.org/papers/www2019splitter.pdf
    """
    def __init__(self, args, base_node_count, node_count):
        """
        Splitter set up.
        :param args: Arguments object.
        :param base_node_count: Number of nodes in the source graph.
        :param node_count: Number of nodes in the persona graph.
        """
        super(Splitter, self).__init__()
        self.args = args
        self.base_node_count = base_node_count
        self.node_count = node_count

    def create_weights(self):
        """
        Creating weights for embedding.
        """
        self.base_node_embedding = torch.nn.Embedding(self.base_node_count,
                                                      self.args.dimensions,
                                                      padding_idx=0)

        self.node_embedding = torch.nn.Embedding(self.node_count,
                                                 self.args.dimensions,
                                                 padding_idx=0)

        self.node_noise_embedding = torch.nn.Embedding(self.node_count,
                                                       self.args.dimensions,
                                                       padding_idx=0)

    def initialize_weights(self, base_node_embedding, mapping):
        """
        Using the base embedding and the persona mapping for initializing the embeddings.
        :param base_node_embedding: Node embedding of the source graph.
        :param mapping: Mapping of personas to nodes.
        """
        persona_embedding = np.array([base_node_embedding[n] for _, n in mapping.items()])
        self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
        self.node_noise_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
        self.base_node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(base_node_embedding),
                                                                  requires_grad=False)

    def calculate_main_loss(self, sources, contexts, targets):
        """
        Calculating the main embedding loss.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :return main_loss: Loss value.
        """
        node_f = self.node_embedding(sources)
        node_f = torch.nn.functional.normalize(node_f, p=2, dim=1)
        feature_f = self.node_noise_embedding(contexts)
        feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1)
        scores = torch.sum(node_f*feature_f, dim=1)
        scores = torch.sigmoid(scores)
        main_loss = targets*torch.log(scores)+(1-targets)*torch.log(1-scores)
        main_loss = -torch.mean(main_loss)
        return main_loss

    def calculate_regularization(self, pure_sources, personas):
        """
        Calculating the regularization loss.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return regularization_loss: Loss value.
        """
        source_f = self.node_embedding(pure_sources)
        original_f = self.base_node_embedding(personas)
        scores = torch.clamp(torch.sum(source_f*original_f, dim=1), -15, 15)
        scores = torch.sigmoid(scores)
        regularization_loss = -torch.mean(torch.log(scores))
        return regularization_loss

    def forward(self, sources, contexts, targets, personas, pure_sources):
        """
        Doing a forward pass.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return loss: Loss value.
        """
        main_loss = self.calculate_main_loss(sources, contexts, targets)
        regularization_loss = self.calculate_regularization(pure_sources, personas)
        loss = main_loss + self.args.lambd*regularization_loss
        return loss

class SplitterTrainer(object):
    """
    Class for training a Splitter.
    """
    def __init__(self, graph, args):
        """
        :param graph: NetworkX graph object.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_noises(self):
        """
        Creating node noise distribution for negative sampling.
        """
        self.downsampled_degrees = {}
        for n in self.egonet_splitter.persona_graph.nodes():
            self.downsampled_degrees[n] = int(1+self.egonet_splitter.persona_graph.degree(n)**0.75)
        self.noises = [k for k, v in self.downsampled_degrees.items() for i in range(v)]

    def base_model_fit(self):
        """
        Fitting DeepWalk on base model.
        """
        self.base_walker = DeepWalker(self.graph, self.args)
        print("\nDoing base random walks.\n")
        self.base_walker.create_features()
        print("\nLearning the base model.\n")
        self.base_node_embedding = self.base_walker.learn_base_embedding()
        print("\nDeleting the base walker.\n")
        del self.base_walker

    def create_split(self):
        """
        Creating an EgoNetSplitter.
        """
        self.egonet_splitter = EgoNetSplitter()
        self.egonet_splitter.fit(self.graph)
        self.persona_walker = DeepWalker(self.egonet_splitter.persona_graph, self.args)
        print("\nDoing persona random walks.\n")
        self.persona_walker.create_features()
        self.create_noises()

    def setup_model(self):
        """
        Creating a model and doing a transfer to GPU.
        """
        base_node_count = self.graph.number_of_nodes()
        persona_node_count = self.egonet_splitter.persona_graph.number_of_nodes()
        self.model = Splitter(self.args, base_node_count, persona_node_count)
        self.model.create_weights()
        self.model.initialize_weights(self.base_node_embedding,
                                      self.egonet_splitter.personality_map)
        self.model = self.model.to(self.device)

    def transfer_batch(self, source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes):
        """
        Transfering the batch to GPU.
        """
        self.sources = torch.LongTensor(source_nodes).to(self.device)
        self.contexts = torch.LongTensor(context_nodes).to(self.device)
        self.targets = torch.FloatTensor(targets).to(self.device)
        self.personas = torch.LongTensor(persona_nodes).to(self.device)
        self.pure_sources = torch.LongTensor(pure_source_nodes).to(self.device)

    def optimize(self):
        """
        Doing a weight update.
        """
        loss = self.model(self.sources, self.contexts,
                          self.targets, self.personas, self.pure_sources)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def process_walk(self, walk):
        """
        Process random walk (source, context) pairs.
        Sample negative instances and create persona node list.
        :param walk: Random walk sequence.
        """
        left_nodes = [walk[i] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        right_nodes = [walk[i+j] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        node_pair_count = len(left_nodes)
        source_nodes = left_nodes + right_nodes
        context_nodes = right_nodes + left_nodes
        persona_nodes = np.array([self.egonet_splitter.personality_map[source_node] for source_node in source_nodes])
        pure_source_nodes = np.array(source_nodes)
        source_nodes = np.array((self.args.negative_samples+1)*source_nodes)
        noises = np.random.choice(self.noises, node_pair_count*2*self.args.negative_samples)
        context_nodes = np.concatenate((np.array(context_nodes), noises))
        positives = [1.0 for node in range(node_pair_count*2)]
        negatives = [0.0 for node in range(node_pair_count*self.args.negative_samples*2)]
        targets = np.array(positives + negatives)
        self.transfer_batch(source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes)

    def update_average_loss(self, loss_score):
        """
        Updating the average loss and the description of the time remains bar.
        :param loss_score: Loss on the sample.
        """
        self.cummulative_loss = self.cummulative_loss + loss_score
        self.steps = self.steps + 1
        average_loss = self.cummulative_loss/self.steps
        self.walk_steps.set_description("Splitter (Loss=%g)" % round(average_loss, 4))

    def reset_average_loss(self, step):
        """
        Doing a reset on the average loss.
        :param step: Current number of walks processed.
        """
        if step % 100 == 0:
            self.cummulative_loss = 0
            self.steps = 0

    def fit(self):
        """
        Fitting a model.
        """
        self.base_model_fit()
        self.create_split()
        self.setup_model()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        print("\nLearning the joint model.\n")
        random.shuffle(self.persona_walker.paths)
        self.walk_steps = trange(len(self.persona_walker.paths), desc="Loss")
        for step in self.walk_steps:
            self.reset_average_loss(step)
            walk = self.persona_walker.paths[step]
            self.process_walk(walk)
            loss_score = self.optimize()
            self.update_average_loss(loss_score)

    def save_embedding(self):
        """
        Saving the node embedding.
        """
        print("\n\nSaving the model.\n")
        nodes = [node for node in self.egonet_splitter.persona_graph.nodes()]
        nodes.sort()
        nodes = torch.LongTensor(nodes).to(self.device)
        embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        embedding = [np.array(range(embedding.shape[0])).reshape(-1, 1), embedding]
        embedding = np.concatenate(embedding, axis=1)
        embedding = pd.DataFrame(embedding, columns=embedding_header)
        embedding.to_csv(self.args.embedding_output_path, index=None)

    def save_persona_graph_mapping(self):
        """
        Saving the persona map.
        """
        with open(self.args.persona_output_path, "w") as f:
            json.dump(self.egonet_splitter.personality_map, f)                     
```

<!-- #region id="FPYSDKdTEExT" -->
### Utils
<!-- #endregion -->

```python id="JQiv-uZ0EEsU"
"""Data reading and printing utils."""

import pandas as pd
import networkx as nx
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph
```

<!-- #region id="-8ppLeZUOrev" -->
### Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NqX_PCosOskk" executionInfo={"status": "ok", "timestamp": 1633970763918, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b925fd0-c4e4-4ee4-e9ec-91a1405666f3"
!head ./input/cora_edges.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="m_5s3ykwOwGm" executionInfo={"status": "ok", "timestamp": 1633970763920, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b86b91ac-7a3c-451f-c605-f9fb1c83da3c"
!head ./input/cora_target.csv
```

<!-- #region id="6TJtEjJdDngP" -->
### Params
<!-- #endregion -->

```python id="RzA3V-7vDcci"
"""Parameter parsing."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it trains on the coras dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run Splitter.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/cora_edges.csv",
	                help="Edge list csv.")

    parser.add_argument("--embedding-output-path",
                        nargs="?",
                        default="./output/cora_embedding.csv",
	                help="Embedding output path.")

    parser.add_argument("--persona-output-path",
                        nargs="?",
                        default="./output/cora_personas.json",
	                help="Persona output path.")

    parser.add_argument("--number-of-walks",
                        type=int,
                        default=10,
	                help="Number of random walks per source node. Default is 10.")

    parser.add_argument("--window-size",
                        type=int,
                        default=5,
	                help="Skip-gram window size. Default is 5.")

    parser.add_argument("--negative-samples",
                        type=int,
                        default=5,
	                help="Negative sample number. Default is 5.")

    parser.add_argument("--walk-length",
                        type=int,
                        default=40,
	                help="Truncated random walk length. Default is 40.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for PyTorch. Default is 42.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
	                help="Learning rate. Default is 0.025.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.1,
	                help="Regularization parameter. Default is 0.1.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
	                help="Embedding dimensions. Default is 128.")

    parser.add_argument('--workers',
                        type=int,
                        default=4,
	                help='Number of parallel workers. Default is 4.')

    return parser.parse_args(args={})
```

<!-- #region id="oW9kBXppDWPN" -->
### Main
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 945, "referenced_widgets": ["88eb015e26694d2c851e441d1f1bee50", "30fe03db016e432fbc8a332cb50210e3", "8ac890dfa873414f9bcc36d320b71eb7", "2664dbc2b11d4d5c920e4eae1c9e7b11", "16bacab8fd6c41bcaf6b49086834bd11", "b9bd884368c84ea9ae71069698c68f61", "126b55bae38d4a5791974495c9411b40", "4c91bffbae13491cb981f4c3526619ac", "046f8b70cea54486b54b815e21a394cb", "224fc7f353ae452282d2c4646d897546", "b612687527b64294a41b98c5f94e897f", "72370da8470340afbe5df829d07c6cd5", "eb3fc1bab35f420f9c3a882491732de9", "f5aced473ae74b67964b35ce2615ec3f", "e352a9d727b8471aa805b2b3ebb34760"]} id="qh8LUk1yCnxJ" outputId="580f4aa6-7cf7-4b81-8ca2-90f52c88e996"
"""Running the Splitter."""

import torch
# from param_parser import parameter_parser
# from splitter import SplitterTrainer
# from utils import tab_printer, graph_reader

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    trainer = SplitterTrainer(graph, args)
    trainer.fit()
    trainer.save_embedding()
    trainer.save_persona_graph_mapping()

if __name__ == "__main__":
    main()
```
