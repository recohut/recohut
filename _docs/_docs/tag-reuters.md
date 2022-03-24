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

# Text Analytics using Graphs

```python
import os
project_name = "reco-tut-gml"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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

```python
%%writefile requirements.txt
networkx==2.4  
scikit-learn==0.24.0 
stellargraph==1.2.1 
spacy==3.0.3 
pandas==1.1.3 
numpy==1.19.2 
node2vec==0.3.3 
Keras==2.0.2 
tensorflow==2.4.1 
communities==2.2.0 
gensim==3.8.3 
matplotlib==3.3.4 
nltk==3.5 
langdetect==1.0.9
fasttext==0.9.2
python-louvain==0.15
click==7.1.2
smart-open==3.0.0
```

```python
!pip install -r requirements.txt
```

```python
import numpy as np
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus import reuters
import langdetect
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim.summarization import keywords

from matplotlib import pyplot as plt
from spacy import displacy
from sklearn.manifold import TSNE

import networkx as nx
from networkx.algorithms.bipartite.projection import *
from node2vec import Node2Vec
import community
from community import community_louvain

%matplotlib inline
```

```python
nltk.download('reuters')
!python -m spacy download en_core_web_md

default_edge_color = 'gray'
default_node_color = '#407cc9'
enhanced_node_color = '#f5b042'
enhanced_edge_color = '#cc2f04'
```

## Dataset overview


We will use Reuters-21578 dataset. The original dataset includes a set of 21,578 news articles that were published in the financial Reuters newswire in 1987, which were assembled and indexed in categories. The original dataset has a very skewed distribution, with some categories appearing only in the training set or in the test set. For this reason, we will use a modified version, known as ApteMod, also referred to as Reuters-21578 Distribution 1.0, that has a smaller skew distribution and consistent labels between the training and test datasets. The Reuters-21578 dataset can easily be downloaded using the nltk library (which is a very useful library for post-processing documents).

```python
corpus = pd.DataFrame([
    {"id": _id, "clean_text": reuters.raw(_id).replace("\n", ""), "label": reuters.categories(_id)}
    for _id in reuters.fileids()
]).set_index("id")

corpus.head(2)
```

```python
corpus.info()
```

```python
corpus.describe().T
```

## NLP

In this section, we will extract structured information from text by using NLP techniques and models


### Language detection

```python
def getLanguage(text: str):
    try:
        return langdetect.detect(text)
    except: 
        return np.nan
```

```python
corpus["language"] = corpus["clean_text"].apply(getLanguage)
corpus["language"].value_counts().head(10)
```

### NLP enrichment

```python
nlp = spacy.load('en_core_web_md')
corpus["parsed"] = corpus["clean_text"].apply(nlp)
corpus.loc["test/14832"]["clean_text"]
```

```python
displacy.render(corpus.loc["test/14832"]["parsed"], style='ent', jupyter=True)
```

```python
corpus[["clean_text", "label", "language", "parsed"]].to_pickle("/content/corpus.p")
```

```python
corpus.to_pickle()
```

```python
corpus[["parsed"]].to_pickle("/content/parsed.p", compression='gzip')
```

## Graph Generation
In this section, we will create two different kind of graphs out of a corpus of documents:
1. Knowledge base graphs, where the subject-verb-object relation will be encoded to build a semantic graph
2. Bipartite graphs, linking documents with the entities/keywords appearing therein


## Knowledge Graph

```python
#@markdown SVO
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    #potentialNewVerb, potentialNewObjs = getObjsFromAttrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs

def findSVOs(tokens, output="str"):
    svos = []
    # verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    verbs = [tok for tok in tokens if tok.dep_ != "AUX"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    
                    if output is "str":
                        element = (
                            sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_
                        )
                    elif output is "obj":
                        element = (sub, (v, verbNegated or objNegated), obj)
                    
                    svos.append(element)
    return svos

def getAbuserOntoVictimSVOs(tokens):
    maleAbuser = {'he', 'boyfriend', 'bf', 'father', 'dad', 'husband', 'brother', 'man'}
    femaleAbuser = {'she', 'girlfriend', 'gf', 'mother', 'mom', 'wife', 'sister', 'woman'}
    neutralAbuser = {'pastor', 'abuser', 'offender', 'ex', 'x', 'lover', 'church', 'they'}
    victim = {'me', 'sister', 'brother', 'child', 'kid', 'baby', 'friend', 'her', 'him', 'man', 'woman'}

    svos = findSVOs(tokens)
    wnl = WordNetLemmatizer()
    passed = []
    for s, v, o in svos:
        s = wnl.lemmatize(s)
        v = "!" + wnl.lemmatize(v[1:], 'v') if v[0] == "!" else wnl.lemmatize(v, 'v')
        o = "!" + wnl.lemmatize(o[1:]) if o[0] == "!" else wnl.lemmatize(o)
        if s in maleAbuser.union(femaleAbuser).union(neutralAbuser) and o in victim:
            passed.append((s, v, o))
    return passed

def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def testSVOs():
    tok = nlp("making $12 an hour? where am i going to go? i have no other financial assistance available and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}
    print(svos)

    tok = nlp("i don't have other assistance")
    svos = findSVOs(tok)
    printDeps(tok)
    assert set(svos) == {('i', '!have', 'assistance')}

    print("-----------------------------------------------")
    tok = nlp("They ate the pizza with anchovies.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('they', 'ate', 'pizza')}

    print("--------------------------------------------------")
    tok = nlp("I have no other financial assistance available and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

    print("--------------------------------------------------")
    tok = nlp("I have no other financial assistance available, and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

    print("--------------------------------------------------")
    tok = nlp("he did not kill me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!kill', 'me')}

    #print("--------------------------------------------------")
    #tok = nlp("he is an evil man that hurt my child and sister")
    #svos = findSVOs(tok)
    #printDeps(tok)
    #print(svos)
    #assert set(svos) == {('he', 'hurt', 'child'), ('he', 'hurt', 'sister'), ('man', 'hurt', 'child'), ('man', 'hurt', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("he told me i would die alone with nothing but my career someday")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'told', 'me')}

    print("--------------------------------------------------")
    tok = nlp("I wanted to kill him with a hammer.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', 'kill', 'him')}

    print("--------------------------------------------------")
    tok = nlp("because he hit me and also made me so angry i wanted to kill him with a hammer.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'hit', 'me'), ('i', 'kill', 'him')}

    print("--------------------------------------------------")
    tok = nlp("he and his brother shot me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'shot', 'me'), ('brother', 'shot', 'me')}

    print("--------------------------------------------------")
    tok = nlp("he and his brother shot me and my sister")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'shot', 'me'), ('he', 'shot', 'sister'), ('brother', 'shot', 'me'), ('brother', 'shot', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("the annoying person that was my boyfriend hit me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('person', 'was', 'boyfriend'), ('person', 'hit', 'me')}

    print("--------------------------------------------------")
    tok = nlp("the boy raced the girl who had a hat that had spots.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', 'had', 'spots')}

    print("--------------------------------------------------")
    tok = nlp("he spit on me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'spit', 'me')}

    print("--------------------------------------------------")
    tok = nlp("he didn't spit on me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!spit', 'me')}

    print("--------------------------------------------------")
    tok = nlp("the boy raced the girl who had a hat that didn't have spots.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', '!have', 'spots')}

    print("--------------------------------------------------")
    tok = nlp("he is a nice man that didn't hurt my child and sister")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'is', 'man'), ('man', '!hurt', 'child'), ('man', '!hurt', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("he didn't spit on me and my child")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!spit', 'me'), ('he', '!spit', 'child')}

    print("--------------------------------------------------")
    tok = nlp("he beat and hurt me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    # tok = nlp("he beat and hurt me")
```

```python
corpus["triplets"] = corpus["parsed"].apply(lambda x: findSVOs(x, output="obj"))
corpus.head()
```

```python
edge_list = [
    {"id": _id, "source": source.lemma_.lower(), "target": target.lemma_.lower(), "edge": edge.lemma_.lower()}
    for _id, triplets in corpus["triplets"].iteritems()
    for (source, (edge, neg), target) in triplets
]

edges = pd.DataFrame(edge_list)
edges.head()
```

```python
edges["edge"].value_counts().head(10)
```

```python
G=nx.from_pandas_edgelist(edges, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
```

```python
print(nx.info(G))
```

```python
np.log10(pd.Series({k: v for k, v in nx.degree(G)}).sort_values(ascending=False)).hist()
plt.yscale("log")
plt.show()
```

```python
e = edges[(edges["source"]!=" ") & (edges["target"]!=" ") & (edges["edge"]=="lend")]
G=nx.from_pandas_edgelist(e, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(13, 6))
pos = nx.spring_layout(G, k=1.2) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos, font_size=12)
plt.savefig("KnowledgeGraph.png", dpi=300, format="png")
plt.show()
```

## Bipartite Graph


### Keyword extraction

```python
text = corpus["clean_text"][0]
keywords(text, words=10, split=True, scores=True, pos_filter=('NN', 'JJ'), lemmatize=True)
```

```python
corpus["keywords"] = corpus["clean_text"].apply(
    lambda text: keywords(text, words=10, split=True, scores=True, pos_filter=('NN', 'JJ'), lemmatize=True)
)
corpus.head()
```

```python
def extractEntities(ents, minValue=1, typeFilters=["GPE", "ORG", "PERSON"]):
    entities = pd.DataFrame([
        {"lemma": e.lemma_, "lower": e.lemma_.lower(), "type": e.label_}
        for e in ents if hasattr(e, "label_")
    ])

    if len(entities)==0:
        return pd.DataFrame()
    
    g = entities.groupby(["type", "lower"])
x
    summary = pd.concat({
        "alias": g.apply(lambda x: x["lemma"].unique()), 
        "count": g["lower"].count()
    }, axis=1)
    
    return summary[summary["count"]>1].loc[pd.IndexSlice[typeFilters, :, :]]
```

```python
def getOrEmpty(parsed, _type):
    try:
        return list(parsed.loc[_type]["count"].sort_values(ascending=False).to_dict().items())
    except:
        return []
```

```python
def toField(ents):
    typeFilters=["GPE", "ORG", "PERSON"]
    parsed = extractEntities(ents, 1, typeFilters)
    return pd.Series({_type: getOrEmpty(parsed, _type) for _type in typeFilters})
```

```python
entities = corpus["parsed"].apply(lambda x: toField(x.ents))
merged = pd.concat([corpus, entities], axis=1) 
merged.head()
```

### Entity-entity graph projection

```python
edges = pd.DataFrame([
    {"source": _id, "target": keyword, "weight": score, "type": _type}
    for _id, row in merged.iterrows()
    for _type in ["keywords", "GPE", "ORG", "PERSON"] 
    for (keyword, score) in row[_type]
])
```

```python
G = nx.Graph()
G.add_nodes_from(edges["source"].unique(), bipartite=0)
G.add_nodes_from(edges["target"].unique(), bipartite=1)
G.add_edges_from([
    (row["source"], row["target"])
    for _, row in edges.iterrows()
])
```

```python
document_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
entity_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
nodes_with_low_degree = {n for n, d in nx.degree(G, nbunch=entity_nodes) if d<5}
```

```python
print(nx.info(G))
```

> Warning: Following cell will take 30-40 mins to run.

```python
dimensions = 10
window = 20

node2vec = Node2Vec(G, dimensions=dimensions) 
model = node2vec.fit(window=window) 
embeddings = model.wv 

pd.DataFrame(embeddings.vectors, index=embeddings.index2word)\
    .to_pickle(f"bipartiteGraphEmbeddings_{dimensions}_{window}.p")
```

```python
edges.to_pickle('bipartiteEdges.p')
```

```python
subGraph = G.subgraph(set(G.nodes) - nodes_with_low_degree)
entityGraph = overlap_weighted_projected_graph(
    subGraph, 
    {n for n, d in subGraph.nodes(data=True) if d["bipartite"] == 1}
)
print(nx.info(entityGraph))
```

```python
filteredEntityGraph = entityGraph.edge_subgraph(
    [edge for edge in entityGraph.edges if entityGraph.edges[edge]["weight"]>0.05]
)
print(nx.info(filteredEntityGraph))
```

#### Local and global properties of the graph

```python
globalKpis = [{
    "shortest_path": nx.average_shortest_path_length(_graph),
    "clustering_coefficient": nx.average_clustering(_graph),
    "global_efficiency": nx.global_efficiency(_graph)
} for components in nx.connected_components(filteredEntityGraph) 
    for _graph in [nx.subgraph(filteredEntityGraph, components)]]
    
pd.concat([
    pd.DataFrame(globalKpis), 
    pd.Series([len(c) for c in nx.connected_components(filteredEntityGraph)])
], axis=1)
```

```python
globalKpis[0]
```

```python
betweeness = nx.betweenness_centrality(filteredEntityGraph)
_betweeness = pd.Series(betweeness)
pageRanks = pd.Series(nx.pagerank(filteredEntityGraph))
degrees = pd.Series({k: v for k, v in nx.degree(filteredEntityGraph)})

kpis = pd.concat({
    "pageRank": pageRanks, 
    "degrees": degrees, 
    "betweeness": _betweeness
}, axis=1)
```

```python
def plotDistribution(serie: pd.Series, nbins: int, minValue=None, maxValue=None):
    _minValue=int(np.floor(np.log10(minValue if minValue is not None else serie.min())))
    _maxValue=int(np.ceil(np.log10(maxValue if maxValue is not None else serie.max())))
    bins = [0] + list(np.logspace(_minValue, _maxValue, nbins)) + [np.inf]
    serie.hist(bins=bins)
    plt.xscale("log")
```

```python
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.title("Page rank vs degrees")
plt.plot(kpis["pageRank"], kpis["degrees"], '.', color="tab:blue")
plt.xlabel("page rank")
plt.ylabel("degree")
plt.xscale("log")
plt.yscale("log")

plt.subplot(1,2,2)
plt.title("Page rank vs betweeness")
plt.plot(kpis["pageRank"], kpis["betweeness"], '.', color="tab:blue")
plt.xlabel("page rank")
plt.ylabel("betweeness")
plt.xscale("log")
plt.yscale("log")
plt.ylim([1E-5, 2E-2])
```

```python
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plotDistribution(degrees, 13)
plt.yscale("log")
plt.title("Degree Distribution")

plt.subplot(1,2,2)
plotDistribution(allEdgesWeights, 20)
plt.xlim([1E-2, 10])
plt.yscale("log")
plt.title("Edge Weight Distribution")
```

#### Network visualization

```python
#Create network layout for visualizations
spring_pos = nx.spring_layout(filteredEntityGraph)

default_edge_color = 'gray'
default_node_color = '#407cc9'
enhanced_node_color = '#f5b042'
enhanced_edge_color = '#cc2f04'

plt.axis("off")
nx.draw_networkx(filteredEntityGraph, pos=spring_pos, node_color=default_node_color, 
                 edge_color=default_edge_color, with_labels=False, node_size=15)
plt.show()
```

#### Community detection

```python
communities = pd.Series(community_louvain.best_partition(filteredEntityGraph))
communities.value_counts().sort_values(ascending=False).plot(kind="bar", figsize=(12, 5))
plt.xlabel("Community")
plt.ylabel("# Members")
```

```python
nodes = communities[communities==17].index
nodes
```

```python
smallGrap = nx.subgraph(filteredEntityGraph, nbunch=nodes)

plt.figure(figsize=(10,10))
pos = nx.spring_layout(smallGrap) # k regulates the distance between nodes
nx.draw(smallGrap, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.savefig("CloseUp.png", dpi=300, format="png")
plt.show()
```

```python
bipartiteCloseup = subGraph.edge_subgraph( {e for e in subGraph.edges() if len(set(e).intersection(nodes))>0})
deg = nx.degree(bipartiteCloseup)
smallGrap = nx.subgraph(bipartiteCloseup, {n for n, d in bipartiteCloseup.nodes(data=True) if d["bipartite"]==1 or deg[n]>1})

plt.figure(figsize=(10,10))
pos = nx.kamada_kawai_layout(smallGrap) # k regulates the distance between nodes
node_color = ["skyblue" if d["bipartite"]==1 else "red" for n, d in smallGrap.nodes(data=True)]
nx.draw(smallGrap, with_labels=False, node_color=node_color, #'skyblue', 
        node_size=150, edge_cmap=plt.cm.Blues, pos = pos)
plt.savefig("BipartiteCloseUp.png", dpi=300, format="png")
plt.show()
```

#### Embeddings

```python
node2vec = Node2Vec(filteredEntityGraph, dimensions=5) 
model = node2vec.fit(window=10) 
embeddings = model.wv 

tsne=TSNE(n_components=2)
embedding2d=tsne.fit_transform(embeddings.vectors)

plt.plot(embedding2d[:, 0], embedding2d[:, 1], 'o')
plt.show()
```

```python
# Node2Vec allows also to compute a similarity between entities
embeddings.most_similar(positive=["turkey"])
```

### Document-document graph projection

```python
documentGraph = overlap_weighted_projected_graph(G, {n for n, d in G.nodes(data=True) if d["bipartite"] == 0})
print(nx.info(documentGraph))
```

```python
allEdgesWeights = pd.Series({(d[0], d[1]): d[2]["weight"] for d in documentGraph.edges(data=True)})
filteredDocumentGraph = documentGraph.edge_subgraph(
    allEdgesWeights[(allEdgesWeights>0.6)].index.tolist()
)
print(nx.info(filteredDocumentGraph))
```

#### Network visualization

```python
spring_pos = nx.spring_layout(filteredDocumentGraph)

plt.axis("off")
nx.draw_networkx(filteredDocumentGraph, pos=spring_pos, node_color=default_node_color, 
                 edge_color=default_edge_color, with_labels=False, node_size=15)
plt.show()
```

```python
components = pd.Series({ith: component 
              for ith, component in enumerate(nx.connected_components(filteredDocumentGraph))})

coreDocumentGraph = nx.subgraph(
    filteredDocumentGraph,
    [node for nodes in components[components.apply(len)>8].values for node in nodes]
)

print(nx.info(coreDocumentGraph))
```

```python
spring_pos = nx.spring_layout(coreDocumentGraph)

plt.axis("off")
nx.draw_networkx(coreDocumentGraph, pos=spring_pos, node_color=default_node_color, 
                 edge_color=default_edge_color, with_labels=False, node_size=15)
plt.show()
```

#### Community Detection and Topics Clustering

```python
communities = pd.Series(community_louvain.best_partition(coreDocumentGraph))
communities = pd.Series(community_louvain.best_partition(filteredDocumentGraph))

def getTopicRatio(df):
    return Counter([label for labels in df["label"] for label in labels])

communityTopics = pd.DataFrame.from_dict({
    cid: getTopicRatio(corpus.loc[comm.index])
    for cid, comm in communities.groupby(communities)
}, orient="index")

normalizedCommunityTopics = (communityTopics.T / communityTopics.sum(axis=1)).T

topicsCorrelation = normalizedCommunityTopics.corr().fillna(0)
topicsCorrelation[topicsCorrelation<0.8]=0

topicsGraph = nx.from_pandas_adjacency(topicsCorrelation)
```

```python
plt.figure(figsize=(8,8))
pos = nx.spring_layout(topicsGraph, k=0.35) # k regulates the distance between nodes
nx.draw(topicsGraph, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.savefig("TopicsAll.png", dpi=300, format="png")
plt.show()
```

```python
filteredTopicsGraph = nx.subgraph(
    topicsGraph,
    [node for component in nx.connected_components(topicsGraph) if len(component)>3 for node in component]
)

plt.figure(figsize=(8,8))
pos = nx.kamada_kawai_layout(filteredTopicsGraph) # k regulates the distance between nodes
nx.draw(filteredTopicsGraph, with_labels=True, node_color='skyblue', node_size=1500, 
        edge_cmap=plt.cm.Blues, pos = pos)
plt.savefig("TopicsCore.png", dpi=300, format="png")
plt.show()
```

#### Embeddings

```python
node2vec = Node2Vec(coreDocumentGraph, dimensions=20) 
model = node2vec.fit(window=10) 
embeddings = model.wv 

tsne=TSNE(n_components=2)
embedding2d=tsne.fit_transform(embeddings.vectors)

plt.plot(embedding2d[:, 0], embedding2d[:, 1], 'o')
```

```python
pd.DataFrame(embeddings.vectors, index=embeddings.index2word)
```

## Shallow-Learning Topic Modelling


In the following we will create a topic model, using a shallow-learning approach. Here we will use the results and the embeddings obtained from the document-document projection of the bipartite graph.

```python
import pandas as pd
from glob import glob
from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
```

```python
corpus = pd.read_pickle("corpus.p")
corpus.head()
```

```python
topics = Counter([label for document_labels in corpus["label"] for label in document_labels]).most_common(10)
topics
```

```python
topicsList = [topic[0] for topic in topics]
topicsSet = set(topicsList)
dataset = corpus[corpus["label"].apply(lambda x: len(topicsSet.intersection(x))>0)]
```

### Create a class to "simulate" the training of the embeddings

```python
class EmbeddingsTransformer(BaseEstimator):
    
    def __init__(self, embeddings_file):
        self.embeddings_file = embeddings_file
        
    def fit(self, *args, **kwargs):
        self.embeddings = pd.read_pickle(self.embeddings_file)
        return self
        
    def transform(self, X):
        return self.embeddings.loc[X.index]
    
    def fit_transform(self, X, y):
        return self.fit().transform(X)
```

```python
files = glob("./bipartiteGraphEmbeddings*")
files
```

```python
graphEmbeddings = EmbeddingsTransformer(files[0]).fit()
```

### Train/Test split

```python
def get_labels(corpus, topicsList=topicsList):
    return corpus["label"].apply(
        lambda labels: pd.Series({label: 1 for label in labels}).reindex(topicsList).fillna(0)
    )[topicsList]
```

```python
def get_features(corpus):
    return corpus["parsed"] #graphEmbeddings.transform(corpus["parsed"])
```

```python
def get_features_and_labels(corpus):
    return get_features(corpus), get_labels(corpus)
```

```python
def train_test_split(corpus):
    graphIndex = [index for index in corpus.index if index in graphEmbeddings.embeddings.index]
    
    train_idx = [idx for idx in graphIndex if "training/" in idx]
    test_idx = [idx for idx in graphIndex if "test/" in idx]
    return corpus.loc[train_idx], corpus.loc[test_idx]
```

```python
train, test = train_test_split(dataset)
```

### Build the model and cross-validation 

```python
model = MultiOutputClassifier(RandomForestClassifier())
```

```python
pipeline = Pipeline([
    ("embeddings", graphEmbeddings),
    ("model", model)
])
```

```python
param_grid = {
    "embeddings__embeddings_file": files,
    "model__estimator__n_estimators": [50, 100], 
    "model__estimator__max_features": [0.2,0.3, "auto"], 
    #"model__estimator__max_depth": [3, 5]
}
```

```python
features, labels = get_features_and_labels(train)
```

```python
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, 
                           scoring=lambda y_true, y_pred: f1_score(y_true, y_pred,average='weighted'))
```

```python
model = grid_search.fit(features, labels)
model
```

```python
model.best_params_
```

### Evaluate performance 

```python
def get_predictions(model, features):
    return pd.DataFrame(
        model.predict(features), 
        columns=topicsList, 
        index=features.index
    )
```

```python
preds = get_predictions(model, get_features(test))
labels = get_labels(test)
```

```python
errors = 1 - (labels - preds).abs().sum().sum() / labels.abs().sum().sum()
errors
```

```python
print(classification_report(labels, preds))
```

## Graph Neural Network Topic Classifier [TODO]


In the following we will focus on building a model for topic classification based on a Graph Neural Network approach.

In particular in the following we will learn how to:

* Create a TF-IDF representation of the corpus, that will be used as node features in the Graph Neural Network model 
* Build, train a Graph Neural Network model and identify the best threshold for classifying documents 
* Test the performance of the model in a out-of-sample tests, following a truly inductive approach 

```python
import nltk 
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import stellargraph as sg
from stellargraph import StellarGraph, IndexedArray
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model
```

```python
corpus = pd.read_pickle("corpus.p")
corpus.head()
```

```python
topics = Counter([label for document_labels in corpus["label"] for label in document_labels]).most_common(10)
topics
```

```python
topicsList = [topic[0] for topic in topics]
topicsSet = set(topicsList)
dataset = corpus[corpus["label"].apply(lambda x: len(topicsSet.intersection(x))>0)]
```

```python
def get_labels(corpus, topicsList=topicsList):
    return corpus["label"].apply(
        lambda labels: pd.Series({label: 1 for label in labels}).reindex(topicsList).fillna(0)
    )[topicsList]
```

```python
labels = get_labels(dataset)
labels.head()
```

```python
def get_features(corpus):
    return corpus["parsed"]
```

```python
def get_features_and_labels(corpus):
    return get_features(corpus), get_labels(corpus)
```

```python
def train_test_split(corpus):
    train_idx = [idx for idx in corpus.index if "training/" in idx]
    test_idx = [idx for idx in corpus.index if "test/" in idx]
    return corpus.loc[train_idx], corpus.loc[test_idx]
```

```python
train, test = train_test_split(dataset)
```

```python
def my_spacy_tokenizer(pos_filter=["NOUN", "VERB", "PROPN"]):
    def tokenizer(doc):
        return [token.lemma_ for token in doc if (pos_filter is None) or (token.pos_ in pos_filter)] 
    return tokenizer
```

```python
cntVectorizer = TfidfVectorizer(
    analyzer=my_spacy_tokenizer(),
    max_df = 0.25, min_df = 2, max_features = 10000
)
```

```python
trainFeatures, _ = get_features_and_labels(train)
testFeatures, _ = get_features_and_labels(test)
```

```python
trainedTransformed = cntVectorizer.fit_transform(trainFeatures)
testTransformed = cntVectorizer.transform(testFeatures)
```

```python
features = pd.concat([
    pd.DataFrame.sparse.from_spmatrix(trainedTransformed, index=trainFeatures.index), 
    pd.DataFrame.sparse.from_spmatrix(testTransformed, index=testFeatures.index)
])
```

```python
features.shape
```

### Creating the Graph

```python
edges = pd.read_pickle("bipartiteEdges.p")
entityTypes = {entity: ith for ith, entity in enumerate(edges["type"].unique())}
entityTypes
```

```python
documentFeatures = features.loc[set(corpus.index).intersection(features.index)] #.assign(document=1, entity=0)
documentFeatures.head()
```

```python
entities = edges.groupby(["target", "type"])["source"].count().groupby(level=0).apply(
    lambda s: s.droplevel(0).reindex(entityTypes.keys()).fillna(0)
).unstack(level=1)
```

```python
entityFeatures = (entities.T / entities.sum(axis=1)).T.assign(document=0, entity=1)
```

```python
nodes = {"entity": entityFeatures, 
         "document": documentFeatures}
```

```python
stellarGraph = StellarGraph(nodes, 
                            edges[edges["source"].isin(documentFeatures.index)], 
                            edge_type_column="type")
```

```python
print(stellarGraph.info())
```

```python
splitter = EdgeSplitter(stellarGraph)
```

```python
graphTest, samplesTest, labelsTest = splitter.train_test_split(p=0.2)
```

```python
print(stellarGraph.info())
```

```python
print(graphTest.info())
```

### Creating a Topic Classification Model 


We start by splitting the data into train, validation and test

```python
targets = labels.reindex(documentFeatures.index).fillna(0)
#documentFeatures.drop(["entity", "document"], axis=1)
```

```python
targets.head()
```

```python
def train_test_split(corpus):
    graphIndex = [index for index in corpus.index]
    
    train_idx = [idx for idx in graphIndex if "training/" in idx]
    test_idx = [idx for idx in graphIndex if "test/" in idx]
    return corpus.loc[train_idx], corpus.loc[test_idx]
```

```python
sampled, hold_out = train_test_split(targets)
```

```python
allNeighbors = np.unique([n for node in sampled.index for n in stellarGraph.neighbors(node)])
```

```python
subgraph = stellarGraph.subgraph(set(sampled.index).union(allNeighbors))
```

```python
print(subgraph.info())
```

```python
train, leftOut = train_test_split(
    sampled,
    train_size=0.1,
    test_size=None,
    random_state=42,
)

validation, test = train_test_split(
    leftOut, train_size=0.2, test_size=None, random_state=100,
)
```

```python
validation = validation[validation.sum(axis=1) > 0]
test = test[test.sum(axis=1) > 0]
```

```python
print(f"Validation: {validation.shape}")
print(f"Test: {test.shape}")
```

### Training the Model


We start  by creating the model 

```python
batch_size = 50
num_samples = [10, 5]
```

```python
generator = HinSAGENodeGenerator(subgraph, batch_size, num_samples, head_node_type="document")
graphsage_model = HinSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
)
```

```python
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train.shape[1], activation="sigmoid")(x_out)
```

```python
prediction.shape
```

```python
model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.binary_crossentropy,
    metrics=["acc"],
)

```

We now train the model 

```python
train_gen = generator.flow(train.index, train, shuffle=True)
```

```python
val_gen = generator.flow(validation.index, validation)
```

```python
history = model.fit(
    train_gen, epochs=50, validation_data=val_gen, verbose=1, shuffle=False
)
```

```python
sg.utils.plot_history(history)
```

```python
history = model.fit(
    train_gen, epochs=50, validation_data=val_gen, verbose=1, shuffle=False
)
```

```python
sg.utils.plot_history(history)
```

### Threshold identification

```python
test_gen = generator.flow(test.index, test)
```

```python
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
```

```python
test_predictions = pd.DataFrame(model.predict(test_gen), index=test.index, columns=test.columns)
```

```python
test_results = pd.concat({
    "target": test, 
    "preds": test_predictions
}, axis=1)
```

```python
f1s = {}

for th in [0.01,0.05,0.1,0.2,0.3,0.4,0.5]:
    f1s[th] = f1_score(test_results["target"], 1.0*(test_results["preds"]>th), average="macro")
    
pd.Series(f1s).plot()
```

As it can be seen, with a threshold of about 0.2 we obtain the best performances. We thus use this value for producing the classification report

```python
print(classification_report(test_results["target"], 1.0*(test_results["preds"]>0.2)))
```

### Inductive Prediction


We now provide a prediction truly inductive, thus we will be using the full graph and we will also use the threshold of 0.2 we have identified above as the one providing the top f1-score.  

```python
generator = HinSAGENodeGenerator(stellarGraph, batch_size, num_samples, head_node_type="document")
```

```python
hold_out = hold_out[hold_out.sum(axis=1) > 0]
```

```python
hold_out_gen = generator.flow(hold_out.index, hold_out)
```

```python
hold_out_predictions = model.predict(hold_out_gen)
```

```python
preds = pd.DataFrame(1.0*(hold_out_predictions > 0.2), index=hold_out.index, columns=hold_out.columns)
```

```python
results = pd.concat({
    "target": hold_out, 
    "preds": preds
}, axis=1)
```

```python
print(classification_report(results["target"], results["preds"]))
```
