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

<!-- #region id="yCaUMaTaT4Ia" -->
# Food Preference Recommender on WWEIA dataset
<!-- #endregion -->

<!-- #region id="tGLOLyWwRZ1A" -->
## Background
<!-- #endregion -->

<!-- #region id="_eZxgCN3RYm-" -->
### Traditional methods

Traditional food recommenders use content-based and collaborative techniques. But effectiveness of these approaches relies on having detailed information about users’ feelings, specifically the user for whom the recommendation is made and similar users, toward many different items. Such detailed information is difficult to obtain. Additionally, approaches either use data scraped from recipe rating websites or present users with a series of recipes and ask for their rating. This data is likely not fully representative of users’ eating patterns, as most users do not input ratings for all foods they eat. They probably only review foods they feel especially strongly (positively or negatively) about. Additionally, many of these methods do not consider the frequency with which a user eats a dish. This is important information that influences how likely a user would be to eat a recommended dish.

### Innovative methods

In addition to CB- and CF-based methods, others have taken innovative approaches to gauge food preferences in their recommender systems. [Ueda et al.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.369.4380&rep=rep1&type=pdf) use a user’s food log to gauge how much users like various recipe ingredients based on their frequency of use. [Another group](https://pubmed.ncbi.nlm.nih.gov/30464375/) successively presents images to a user, using a convolutional neural network (CNN) algorithm to learn a user’s preferences. While the approach is innovative and interesting, it is unclear whether people base eating decisions on food appearance when preparing meals at home (instead of ordering in a restaurant). [Toledo et al.](https://www.semanticscholar.org/paper/A-Food-Recommender-System-Considering-Nutritional-Toledo-Alzahrani/160612b921627becdcc006e5256aacebc68612d1) integrate food preferences into their recommendation approach by devising menus containing ingredients that users have liked in the past but not eaten recently and revising based on user feedback of which ingredients they like and do not like. Other studies treat food recommendations as a query over a knowledge graph. [Chen et. al.](https://arxiv.org/abs/2101.01775) takes in a food log or list of liked foods and allergies and outputs recipes that are most similar to the input. [Park et. al.](https://www.nature.com/articles/s41598-020-79422-8) outputs sets of foods that are predicted to pair well together. However, these predictions do not take in user input.

### Difficulty in food data logging

Food logs have been used for purposes such as allergy detection and weight and disease management. One common difficulty has been consistently recording entries for a variety of reasons, such as the laboriousness involved in recording certain kinds of foods, negative emotions arising when journaling and lack of social support. Recent food logging applications such as MyFitnessPal, Cronometer and Lose It! have addressed these challenges by providing a platform that allows the user to input the food name, the quantity, the type of meal, and the time at which the user consumed the food. There can be gamification features or social support to address the barriers to journaling mentioned earlier. However, there are several shortcomings in using the food logs exported directly from these food logging apps. Food names can contain specific brand names, and the food log structure can differ from one website to another, making them difficult to streamline data processing and information retrieval.
<!-- #endregion -->

<!-- #region id="wyl_lRIxR2m3" -->
## Problem Statement

Identify user’s most frequently-eaten foods. The assumption is that foods that are eaten most frequently are the preferred ones by the user. This information regarding user’s favored foods can be used to generate healthy and realistic meal recommendations that feature ingredients that the user commonly consumes.
<!-- #endregion -->

<!-- #region id="Sbic_ndaR4ct" -->
## Approach
<!-- #endregion -->

<!-- #region id="V1FIXgJQR9dk" -->
<!-- #endregion -->

<!-- #region id="t1159C7YRotW" -->
Workflow of the food preference learning algorithm. Each entry of the food log is processed through an NLP module. Then food embeddings is obtained for each food entry in the food log and the the database. Next, for each food embedding of the food log, Cosine similarity is run against all embeddings of the foods in the WWEIA database. The food category label of the food with the highest Cosine similarity is assigned to the food log entry. This process is repeated for all foods in the food logs. The most common food categories are then calculated.

As the correct labeling of a food depends on the similarity between a food log entry and its database counterpart (or the one most closely analogous). Thus to improve food label accuracy, food log entry names were preprocessed in various ways to remove words that increased similarity to incorrect FNDDS entries or decreased similarity to the correct entry.

- Method 1 - only retain the food’s general name.
- Method 2 - similar to method 1, but retained the comma-separated phrases that contained specific details.
- Method 3 - like Method 2, retained most of the food name, but used another heuristic to judge whether the first comma-separated phrase contained a brand name. Instead of counting the number of words that belonged to the FNDDS vocabulary, the percentage of FNDDS words in the comma-separated phrase was used.
- Method 4 - generic food-related terms were removed from the food log entry name (in addition to the preprocessing done in Method 3), was introduced after noticing that for some food log entries, the most similar database food was one that was wholly unrelated but contained a generic word in common. Frequency of each word in the FNDDS vocabulary was tabulated. All of the generic words among the top 250 most common words were removed from the food log name.
- Method 5 addressed the mislabeling of foods such as ”Kind, Nuts & Spices Bar, Dark Chocolate Nuts & Sea Salt,” where the first comma-separated phrase contains not only the brand name but also the general food name. This method was identical to Method 4 except that instead of removing the whole first comma-separated phrase, only words not found in the FNDDS vocabulary were removed.
<!-- #endregion -->

<!-- #region id="enqHWXpbKtvd" -->
## Setup
<!-- #endregion -->

<!-- #region id="RraknsjZK3YO" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="F6bJJphVL5Jz" executionInfo={"status": "ok", "timestamp": 1635866761188, "user_tz": -330, "elapsed": 2747, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="83c8a672-fde0-4fa4-b304-18708e11e7bc"
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
import math
import re
from collections import Counter

import nltk
nltk.download('wordnet')
```

<!-- #region id="s8nNAVpaK4xH" -->
### Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TXmxyAivLD_4" executionInfo={"status": "ok", "timestamp": 1635866808914, "user_tz": -330, "elapsed": 1702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6bf51116-5a1f-460a-dd58-0a4b2e2dc275"
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/wweia_data.csv
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/food_words.csv
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/sample_food_logs/sample_food_log.csv
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/wweia_food_categories_addtl.csv

!wget -q --show-progress https://github.com/sparsh-ai/food-recsys/raw/main/data/wweia/data.zip
!unzip data.zip
```

<!-- #region id="dAwWO40YK5d6" -->
### Pre-trained embeddings
<!-- #endregion -->

```python id="0KUQKv7rJbMc"
# !gdown --id 1EoNIIfAhXgMaWgmrCOQkzqV2pOvKQOBN
```

```python id="VxnpEg7PLCB4"
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/model_added_sentences.kv
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/wweia_cat_nums_to_words.pickle
# !wget -q --show-progress https://github.com/aametwally/LearningFoodPreferences/raw/main/data/wweia_synonym_cats.pickle
```

```python id="XCCSmYTCLB_p"
# Load word2vec model
model = KeyedVectors.load('model_added_sentences.kv', mmap='r')

# Getting a sample of what an embedding looks like
model["wheat"]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="fkYDUdsCLB5v" executionInfo={"status": "ok", "timestamp": 1635798547316, "user_tz": -330, "elapsed": 1231, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="235fe11b-8b91-49a7-f943-b49d9f2db895"
# Load WWEIA food categories csv. Contains the food category codes and associated verbal food categories.
# The "same_category" column lists which categories are considered synonymous.
wweia_food_categories = pd.read_csv('wweia_food_categories_addtl.csv')
wweia_food_categories
```

```python id="7OkmD6B3MPKT"
# Load the vocabulary of WWEIA food descriptions
food_words = pd.read_csv('food_words.csv')
food_words = food_words.iloc[:,1]
food_words = food_words.to_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 694} id="kCfW1PPtMQ14" executionInfo={"status": "ok", "timestamp": 1635798564831, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b859441-638a-44a6-a1c2-92ae99c1e65c"
# Load wweia_data, which contains the FNDDS foods, their nutritional information, and their food category
wweia_data = pd.read_csv('wweia_data.csv')
wweia_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="UvvIHYGKMTmu" executionInfo={"status": "ok", "timestamp": 1635799472746, "user_tz": -330, "elapsed": 3403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="668783bd-0f05-4b02-85d0-4878beca0054"
# Load embeddings
wweia_embeddings = np.loadtxt('wweia_embeddings_added_sen.csv', delimiter = ",")
wweia_embeddings.shape
```

```python id="gYBTbfWcMchX"
def get_most_pop_categories(curr_log, n, col_name):

  cats = curr_log.loc[:, col_name]

  cats_counter = Counter(cats)
  counter_results = cats_counter.most_common(n)
  
  counter_list = [elem[0] for elem in counter_results]
  
  return counter_list
```

```python id="F3cZ8_sJNByP"
# initialize lemmatizer
lemmatizer = WordNetLemmatizer()
```

<!-- #region id="eolyvZLhLB85" -->
## Method 1
<!-- #endregion -->

```python id="ilffQmenND2v"
# Only keep words in phrase that are in food_words

def reduce_with_food_words(comma_phrase):
  comma_phrase_reduced = [lemmatizer.lemmatize(word.lower()) for word in comma_phrase.split() if (lemmatizer.lemmatize(word.lower()) in food_words or word.lower() in food_words)]
  return comma_phrase_reduced
```

```python id="pImiiwlQNFwf"
# Format of food names seems to be (Optional) Company, Food, Details about food
# run function to guess if First or second comma phrase has Food

def process_food_log(curr_log, wweia_synonym_cats):

  curr_log['predicted_categories_number'] = 0
  curr_log['predicted_categories_words'] = ""
  curr_log['max_cosim_score'] = 0
  curr_log['most_sim_food'] = ""
  curr_log['reciprocal_rank'] = 0.0
  curr_log['sym_reciprocal_rank'] = 0.0

  for i in range(curr_log.shape[0]):
    descrip = curr_log.loc[i, 'Food Name']
    descrip_split = descrip.split(", ")

    # Reduce food log description to words in food_words
    first_phrase_num_words = reduce_with_food_words(descrip_split[0])

    pre_embedding = []

    if len(descrip_split) > 1:

      second_phrase_num_words = reduce_with_food_words(descrip_split[1])
      
      descrip_split_0_words = descrip_split[0].split()
      descrip_split_1_words = descrip_split[1].split()
      
      begin_first = False if len(second_phrase_num_words) > len(first_phrase_num_words) else True
       
      if begin_first:
        last_phrase = descrip_split_0_words

      if begin_first == False:
        last_phrase = descrip_split_1_words

    else:
      last_phrase = reduce_with_food_words(descrip)

    word_embed = np.zeros(shape = (1, len(model["sushi"])))

    if len(last_phrase) > 0:
      # Turned edited_descrip into embedding vector by averaging the words
      word_embed = np.zeros(shape = (1, len(model["sushi"])))
      num_words = 0
      for word in last_phrase:
        word = word.lower()
        if word in model:
          num_words += 1
          word_embed += model[word]

      if num_words != 0:
        word_embed /= num_words

    # Compare to the other vectors
    similarities = cosine_similarity(word_embed, wweia_embeddings) # (1, 7918) for each wweia food

    # Finding WWEIA foods with highest similarity with each
    # food log food. Row has sorted list of WWEIA food IDs 
    # for each food log food.
    to_keep = np.sort(similarities, axis=1)
    to_keep_args = np.argsort(similarities, axis=1)
    indices = np.flip(to_keep_args, axis = 1)
    
    sym_rank = 1000000

    # record RR here
    for index in range(indices.shape[1]):
      true_cat = curr_log.loc[i, 'wweia_food_category_code']
      if math.isnan(true_cat): continue
      if wweia_data.loc[indices[0,index], 'wweia_category_code'] == true_cat:
        rank = index
        rr = 1 / (rank+1)
        if sym_rank > rank:
          sym_rank = rank
          sym_rr = 1 / (rank+1)

        break
      else:
        if wweia_data.loc[indices[0,index], 'wweia_category_code'] in wweia_synonym_cats[true_cat]:
          if sym_rank > index:
            sym_rank = index
            sym_rr = 1 / (index+1)
    
    most_sim_food_index = indices[0,0]
    
    most_sim_food_row = wweia_data.iloc[most_sim_food_index,:]
    highest_cat_num = most_sim_food_row['wweia_category_code']
    
    highest_cat_words = wweia_food_categories.loc[wweia_food_categories['wweia_food_category_code'] == highest_cat_num, 'wweia_food_category_description']

    curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
    curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]
    curr_log.loc[i, 'max_cosim_score'] = np.array2string(to_keep[0,-5:])
    curr_log.loc[i, 'most_sim_food'] = most_sim_food_row['description']
    curr_log.loc[i, 'reciprocal_rank'] = rr
    curr_log.loc[i, 'sym_reciprocal_rank'] = sym_rr
    
  return curr_log
```

```python id="Eilriwf6NLbL"
def get_metrics(curr_log, wweia_synonym_cats):

  # Will hold foods already seen so foods aren't double-counted
  # when calculating accuracy
  seen_foods = set()
  total_valid_foods = 0
  num_corr = 0
  num_sym = 0

  for i in range(curr_log.shape[0]):
    
    true_cat = curr_log.loc[i, 'wweia_food_category_code']
    false_cat = curr_log.loc[i, 'predicted_categories_number']

    prev_set_size = len(seen_foods)
    seen_foods.add(curr_log.loc[i, 'Food Name'])

    if math.isnan(true_cat): continue # food does not have valid category, i.e. antacid

    if len(seen_foods) > prev_set_size: # not a repeated food

      if true_cat == false_cat:
        num_corr += 1
        num_sym += 1

      else:
        if false_cat in wweia_synonym_cats[true_cat]:
          num_sym += 1
        
      total_valid_foods += 1

  acc = num_corr / total_valid_foods
  sym_acc = num_sym / total_valid_foods

  return acc, sym_acc
```

```python id="b607ORqENOGN"
def get_food_prefs(food_log_categories, wweia_cat_nums_to_words):

  wweia_food_cats_list = wweia_food_categories['wweia_food_category_code'].tolist()

  # Make dict to hold the number of foods that belong to a
  # specific WWEIA category. Key is the broad food category.
  cats_dict = {}
  cats_dict["Protein"] = defaultdict(int)
  cats_dict["Vegetable"] = defaultdict(int)
  cats_dict["Grain"] = defaultdict(int)
  cats_dict["Fruit"] = defaultdict(int)
  cats_dict["Dairy"] = defaultdict(int)

  # Dict with lists containing cat codes for each broad food category
  category_mapping = { "Protein" : wweia_food_cats_list[14:34], "Vegetable" : wweia_food_cats_list[91:104],
  "Grain" : wweia_food_cats_list[35:81], "Fruit" : wweia_food_cats_list[82:90], "Dairy" : wweia_food_cats_list[0:13] }

  broad_categories_list = ["Protein", "Vegetable", "Grain", "Fruit", "Dairy"]

  for category_code in food_log_categories:

    if math.isnan(category_code): continue # food does not have valid category, i.e. antacid

    # Loop through each category code and add 1 to its dict entry
    # if it's the right one
    for broad_cat in broad_categories_list:
      if category_code in category_mapping[broad_cat]:
        cats_dict[broad_cat][category_code] += 1
        continue

  max_code_dict = {}
  
  for broad_cat in broad_categories_list:

    if len(list(cats_dict[broad_cat].keys())) > 0:
      max_code_dict[broad_cat] = max(cats_dict[broad_cat], key=cats_dict[broad_cat].get)

    else:
      max_code_dict[broad_cat] = "None of this category was eaten"

  return max_code_dict
```

```python id="wTTe4LVdNPs1"
def make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats):
  
  acc = 0
  sym_acc = 0

  for key in true_food_prefs:
    if true_food_prefs[key] == pred_food_prefs[key]:
      acc += 1
      sym_acc += 1
    else:
      if true_food_prefs[key] not in wweia_synonym_cats: continue
      elif pred_food_prefs[key] in wweia_synonym_cats[true_food_prefs[key]]:
        sym_acc += 1

  acc = acc / len(list(true_food_prefs.keys()))
  sym_acc = sym_acc / len(list(true_food_prefs.keys()))

  return acc, sym_acc
```

```python id="J1vpvYEwNQ5L"
def get_pref_metrics(curr_log, n):
  
  true_pop_cats = get_most_pop_categories(curr_log, n, 'wweia_food_category_code')
  pref_pop_cats = get_most_pop_categories(curr_log, n, 'predicted_categories_number')

  true_pop_cats_sym = set()
  pref_pop_cats_sym = set()

  for elem in true_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      true_pop_cats_sym.add(sym_elem)

  for elem in pref_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      pref_pop_cats_sym.add(sym_elem)

  cats_common = set(true_pop_cats).intersection(set(pref_pop_cats))
  percent_common = len(cats_common) / n

  cats_common_sym = set(true_pop_cats_sym).intersection(set(pref_pop_cats_sym))
  percent_common_sym = min(1, len(cats_common_sym) / n)

  return percent_common, percent_common_sym
```

```python colab={"base_uri": "https://localhost:8080/"} id="ky3LFYaTNSIm" executionInfo={"status": "ok", "timestamp": 1635799541557, "user_tz": -330, "elapsed": 635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f4ca0640-c4bb-451b-ed84-28faca457171"
# Main method

avg_acc = 0.0
avg_sym_acc = 0.0
avg_pref_acc = 0.0
avg_pref_sym_acc = 0.0
avg_percent_common = 0.0
avg_percent_common_sym = 0.0
avg_mean_rr = 0.0
avg_sym_mean_rr = 0.0

food_log_names = []
accs = []
sym_accs = []
pref_accs = []
pref_sym_accs = []
percent_common_list = []
percent_common_sym_list = []
mean_rrs = []
sym_mean_rrs = []

sample_food_logs = ['sample_food_log.csv']

for food_log_name in sample_food_logs:
  
  curr_log = pd.read_csv(food_log_name)

  with open('wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
  
  curr_log = process_food_log(curr_log, wweia_synonym_cats)

  food_log_names.append(food_log_name)

  # # Write to file saved in Google Drive folder
  # file_name = 'lfp_method4/' + food_log_name
  # curr_log.to_csv(file_name)

  # Compare to true food categories
  acc, sym_acc = get_metrics(curr_log, wweia_synonym_cats)
  avg_acc += acc
  avg_sym_acc += sym_acc
  accs.append(acc)
  sym_accs.append(sym_acc)
  print("Accuracy", acc, "Average Synonymous Accuracy:", sym_acc)

  # Count most popular categories and get food preferences
  with open('wweia_cat_nums_to_words.pickle', 'rb') as handle:
    wweia_cat_nums_to_words = pickle.load(handle)

  percent_common, percent_common_sym = get_pref_metrics(curr_log, 10)
  print("Percent Top N Categories Identified:", percent_common, "Percent Top N Synonymous Categories Identified:", percent_common_sym)
  avg_percent_common += percent_common
  avg_percent_common_sym += percent_common_sym
  percent_common_list.append(percent_common)
  percent_common_sym_list.append(percent_common_sym)

  true_food_prefs = get_food_prefs(curr_log.loc[:,'wweia_food_category_code'].to_list(), wweia_cat_nums_to_words)
  pred_food_prefs = get_food_prefs(curr_log.loc[:,'predicted_categories_number'].to_list(), wweia_cat_nums_to_words)

  # print(true_food_prefs)
  # print(pred_food_prefs)

  pref_acc, pref_sym_acc = make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats)
  avg_pref_acc += pref_acc
  avg_pref_sym_acc += pref_sym_acc
  pref_accs.append(pref_acc)
  pref_sym_accs.append(pref_sym_acc)
  print("Preference Accuracy:", pref_acc, "Preference Synonymous Accuracy:", pref_sym_acc)

  mean_rr = np.mean(curr_log.loc[:,'reciprocal_rank'])
  sym_mean_rr = np.mean(curr_log.loc[:,'sym_reciprocal_rank'])
  avg_mean_rr += mean_rr
  avg_sym_mean_rr += sym_mean_rr
  mean_rrs.append(mean_rr)
  sym_mean_rrs.append(sym_mean_rr)
  print("Mean Reciprocal Rank:", mean_rr, "Synonymous Mean Reciprocal Rank:", sym_mean_rr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="8dR1AQeTP4i0" executionInfo={"status": "ok", "timestamp": 1635799542264, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69d86b40-6156-4a5c-8d6e-304751678aa5"
# Make table to show accuracies
df = pd.DataFrame(list(zip(food_log_names, accs, sym_accs, pref_accs, pref_sym_accs, percent_common_list, percent_common_sym_list, mean_rrs, sym_mean_rrs)),
               columns =['Food Log', 'Accuracies', "Synomynous Accuracies", "Preference Accuracies", "Synomynous Preference Accuracies", "Percent of Top N Categories Achieved", "Percent of Top N Categories Achieved Sym", "Mean RR", "Sym Mean RR"])
df
```

<!-- #region id="Tz48OyJnP_vY" -->
## Method 2
<!-- #endregion -->

```python id="kB0fL4NnQi8O"
# Only keep words in phrase that are in food_words

def reduce_with_food_words(comma_phrase):
  comma_phrase_reduced = [lemmatizer.lemmatize(word.lower()) for word in comma_phrase.split() if (lemmatizer.lemmatize(word.lower()) in food_words or word.lower() in food_words)]
  return comma_phrase_reduced
# Format of food names seems to be (Optional) Company, Food, Details about food
# run function to guess if First or second comma phrase has Food


def process_food_log(curr_log, wweia_synonym_cats):

  curr_log['predicted_categories_number'] = 0
  curr_log['predicted_categories_words'] = ""
  curr_log['max_cosim_score'] = 0
  curr_log['most_sim_food'] = ""
  curr_log['reciprocal_rank'] = 0.0
  curr_log['sym_reciprocal_rank'] = 0.0

  for i in range(curr_log.shape[0]):
    descrip = curr_log.loc[i, 'Food Name']
    descrip_split = descrip.split(", ")

    # Reduce food log description to words in food_words
    first_phrase_num_words = reduce_with_food_words(descrip_split[0])

    pre_embedding = []

    if len(descrip_split) > 1:

      second_phrase_num_words = reduce_with_food_words(descrip_split[1])
      
      descrip_split_0_words = descrip_split[0].split()
      descrip_split_1_words = descrip_split[1].split()
      
      begin_first = False if len(second_phrase_num_words) > len(first_phrase_num_words) else True
       
      if begin_first:
        last_phrase = descrip

      if begin_first == False:
        three_phrases = descrip.partition(", ")
        last_phrase = three_phrases[2]

    else:
      last_phrase = descrip

    last_phrase_comma = last_phrase.split(", ")
    for j in range(len(last_phrase_comma)):
      split_comma = last_phrase_comma[j].split(" ")
      for elem in split_comma:
        elem = elem.lower()
        pre_embedding.append(elem)

    word_embed = np.zeros(shape = (1, len(model["sushi"])))

    if len(pre_embedding) > 0:
      # Turned edited_descrip into embedding vector by averaging the words
      word_embed = np.zeros(shape = (1, len(model["sushi"])))
      num_words = 0
      for word in pre_embedding:
        word = word.lower()
        if word in model:
          num_words += 1
          word_embed += model[word]

      if num_words != 0:
        word_embed /= num_words

    # Compare to the other vectors
    similarities = cosine_similarity(word_embed, wweia_embeddings) # (1, 7918) for each wweia food

    # Finding WWEIA foods with highest similarity with each
    # food log food. Row has sorted list of WWEIA food IDs 
    # for each food log food.
    to_keep = np.sort(similarities, axis=1)
    to_keep_args = np.argsort(similarities, axis=1)
    indices = np.flip(to_keep_args, axis = 1)
    
    sym_rank = 1000000

    # record RR here
    for index in range(indices.shape[1]):
      true_cat = curr_log.loc[i, 'wweia_food_category_code']
      if math.isnan(true_cat): continue
      if wweia_data.loc[indices[0,index], 'wweia_category_code'] == true_cat:
        rank = index
        rr = 1 / (rank+1)
        if sym_rank > rank:
          sym_rank = rank
          sym_rr = 1 / (rank+1)

        break
      else:
        if wweia_data.loc[indices[0,index], 'wweia_category_code'] in wweia_synonym_cats[true_cat]:
          if sym_rank > index:
            sym_rank = index
            sym_rr = 1 / (index+1)
    
    most_sim_food_index = indices[0,0]
    
    most_sim_food_row = wweia_data.iloc[most_sim_food_index,:]
    highest_cat_num = most_sim_food_row['wweia_category_code']
    
    highest_cat_words = wweia_food_categories.loc[wweia_food_categories['wweia_food_category_code'] == highest_cat_num, 'wweia_food_category_description']

    curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
    curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]
    curr_log.loc[i, 'max_cosim_score'] = np.array2string(to_keep[0,-5:])
    curr_log.loc[i, 'most_sim_food'] = most_sim_food_row['description']
    curr_log.loc[i, 'reciprocal_rank'] = rr
    curr_log.loc[i, 'sym_reciprocal_rank'] = sym_rr
    
  return curr_log


def get_metrics(curr_log, wweia_synonym_cats):

  # Will hold foods already seen so foods aren't double-counted
  # when calculating accuracy
  seen_foods = set()
  total_valid_foods = 0
  num_corr = 0
  num_sym = 0

  for i in range(curr_log.shape[0]):
    
    true_cat = curr_log.loc[i, 'wweia_food_category_code']
    false_cat = curr_log.loc[i, 'predicted_categories_number']

    prev_set_size = len(seen_foods)
    seen_foods.add(curr_log.loc[i, 'Food Name'])

    if math.isnan(true_cat): continue # food does not have valid category, i.e. antacid

    if len(seen_foods) > prev_set_size: # not a repeated food

      if true_cat == false_cat:
        num_corr += 1
        num_sym += 1

      else:
        if false_cat in wweia_synonym_cats[true_cat]:
          num_sym += 1
        
      total_valid_foods += 1

  acc = num_corr / total_valid_foods
  sym_acc = num_sym / total_valid_foods

  return acc, sym_acc


def get_food_prefs(food_log_categories, wweia_cat_nums_to_words):

  wweia_food_cats_list = wweia_food_categories['wweia_food_category_code'].tolist()

  # Make dict to hold the number of foods that belong to a
  # specific WWEIA category. Key is the broad food category.
  cats_dict = {}
  cats_dict["Protein"] = defaultdict(int)
  cats_dict["Vegetable"] = defaultdict(int)
  cats_dict["Grain"] = defaultdict(int)
  cats_dict["Fruit"] = defaultdict(int)
  cats_dict["Dairy"] = defaultdict(int)

  # Dict with lists containing cat codes for each broad food category
  category_mapping = { "Protein" : wweia_food_cats_list[14:34], "Vegetable" : wweia_food_cats_list[91:104],
  "Grain" : wweia_food_cats_list[35:81], "Fruit" : wweia_food_cats_list[82:90], "Dairy" : wweia_food_cats_list[0:13] }

  broad_categories_list = ["Protein", "Vegetable", "Grain", "Fruit", "Dairy"]

  for category_code in food_log_categories:

    if math.isnan(category_code): continue # food does not have valid category, i.e. antacid

    # Loop through each category code and add 1 to its dict entry
    # if it's the right one
    for broad_cat in broad_categories_list:
      if category_code in category_mapping[broad_cat]:
        cats_dict[broad_cat][category_code] += 1
        continue

  max_code_dict = {}
  
  for broad_cat in broad_categories_list:

    if len(list(cats_dict[broad_cat].keys())) > 0:
      max_code_dict[broad_cat] = max(cats_dict[broad_cat], key=cats_dict[broad_cat].get)

    else:
      max_code_dict[broad_cat] = "None of this category was eaten"

  return max_code_dict


def make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats):
  
  acc = 0
  sym_acc = 0

  for key in true_food_prefs:
    if true_food_prefs[key] == pred_food_prefs[key]:
      acc += 1
      sym_acc += 1
    else:
      if true_food_prefs[key] not in wweia_synonym_cats: continue
      elif pred_food_prefs[key] in wweia_synonym_cats[true_food_prefs[key]]:
        sym_acc += 1

  acc = acc / len(list(true_food_prefs.keys()))
  sym_acc = sym_acc / len(list(true_food_prefs.keys()))

  return acc, sym_acc


def get_pref_metrics(curr_log, n):
  
  true_pop_cats = get_most_pop_categories(curr_log, n, 'wweia_food_category_code')
  pref_pop_cats = get_most_pop_categories(curr_log, n, 'predicted_categories_number')

  true_pop_cats_sym = set()
  pref_pop_cats_sym = set()

  for elem in true_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      true_pop_cats_sym.add(sym_elem)

  for elem in pref_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      pref_pop_cats_sym.add(sym_elem)

  cats_common = set(true_pop_cats).intersection(set(pref_pop_cats))
  percent_common = len(cats_common) / n

  cats_common_sym = set(true_pop_cats_sym).intersection(set(pref_pop_cats_sym))
  percent_common_sym = min(1, len(cats_common_sym) / n)

  return percent_common, percent_common_sym
```

```python id="Rr4pTYmlQ1_2"
# Main method

avg_acc = 0.0
avg_sym_acc = 0.0
avg_pref_acc = 0.0
avg_pref_sym_acc = 0.0
avg_percent_common = 0.0
avg_percent_common_sym = 0.0
avg_mean_rr = 0.0
avg_sym_mean_rr = 0.0

food_log_names = []
accs = []
sym_accs = []
pref_accs = []
pref_sym_accs = []
percent_common_list = []
percent_common_sym_list = []
mean_rrs = []
sym_mean_rrs = []

sample_food_logs = ['sample_food_log.csv']
```

```python colab={"base_uri": "https://localhost:8080/"} id="1YyUtyFSQnRX" executionInfo={"status": "ok", "timestamp": 1635799813537, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="23d0476d-99cc-4a9d-b52c-61cbdcc33688"
for food_log_name in sample_food_logs:
  
  curr_log = pd.read_csv(food_log_name)

  with open('wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
  
  curr_log = process_food_log(curr_log, wweia_synonym_cats)

  food_log_names.append(food_log_name)

  # # Write to file saved in Google Drive folder
  # file_name = 'lfp_method4/' + food_log_name
  # curr_log.to_csv(file_name)

  # Compare to true food categories
  acc, sym_acc = get_metrics(curr_log, wweia_synonym_cats)
  avg_acc += acc
  avg_sym_acc += sym_acc
  accs.append(acc)
  sym_accs.append(sym_acc)
  print("Accuracy", acc, "Average Synonymous Accuracy:", sym_acc)

  # Count most popular categories and get food preferences
  with open('wweia_cat_nums_to_words.pickle', 'rb') as handle:
    wweia_cat_nums_to_words = pickle.load(handle)

  percent_common, percent_common_sym = get_pref_metrics(curr_log, 10)
  print("Percent Top N Categories Identified:", percent_common, "Percent Top N Synonymous Categories Identified:", percent_common_sym)
  avg_percent_common += percent_common
  avg_percent_common_sym += percent_common_sym
  percent_common_list.append(percent_common)
  percent_common_sym_list.append(percent_common_sym)

  true_food_prefs = get_food_prefs(curr_log.loc[:,'wweia_food_category_code'].to_list(), wweia_cat_nums_to_words)
  pred_food_prefs = get_food_prefs(curr_log.loc[:,'predicted_categories_number'].to_list(), wweia_cat_nums_to_words)

  # print(true_food_prefs)
  # print(pred_food_prefs)

  pref_acc, pref_sym_acc = make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats)
  avg_pref_acc += pref_acc
  avg_pref_sym_acc += pref_sym_acc
  pref_accs.append(pref_acc)
  pref_sym_accs.append(pref_sym_acc)
  print("Preference Accuracy:", pref_acc, "Preference Synonymous Accuracy:", pref_sym_acc)

  mean_rr = np.mean(curr_log.loc[:,'reciprocal_rank'])
  sym_mean_rr = np.mean(curr_log.loc[:,'sym_reciprocal_rank'])
  avg_mean_rr += mean_rr
  avg_sym_mean_rr += sym_mean_rr
  mean_rrs.append(mean_rr)
  sym_mean_rrs.append(sym_mean_rr)
  print("Mean Reciprocal Rank:", mean_rr, "Synonymous Mean Reciprocal Rank:", sym_mean_rr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="SJSr4hx1QkDk" executionInfo={"status": "ok", "timestamp": 1635799818022, "user_tz": -330, "elapsed": 579, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="da16a8db-6d98-4896-a563-7df51d157f78"
# Make table to show accuracies
df = pd.DataFrame(list(zip(food_log_names, accs, sym_accs, pref_accs, pref_sym_accs, percent_common_list, percent_common_sym_list, mean_rrs, sym_mean_rrs)),
               columns =['Food Log', 'Accuracies', "Synomynous Accuracies", "Preference Accuracies", "Synomynous Preference Accuracies", "Percent of Top N Categories Achieved", "Percent of Top N Categories Achieved Sym", "Mean RR", "Sym Mean RR"])
df
```

<!-- #region id="zMsf6ojeRFjr" -->
## Method 3
<!-- #endregion -->

```python id="-jJS1iO6RWQO"
# Only keep words in phrase that are in food_words

def reduce_with_food_words(comma_phrase):
  comma_phrase_reduced = [lemmatizer.lemmatize(word.lower()) for word in comma_phrase.split() if (lemmatizer.lemmatize(word.lower()) in food_words or word.lower() in food_words)]
  return comma_phrase_reduced


# Format of food names seems to be (Optional) Company, Food, Details about food
# run function to guess if First or second comma phrase has Food

def process_food_log(curr_log, wweia_synonym_cats):

  curr_log['predicted_categories_number'] = 0
  curr_log['predicted_categories_words'] = ""
  curr_log['max_cosim_score'] = 0
  curr_log['most_sim_food'] = ""
  curr_log['reciprocal_rank'] = 0.0
  curr_log['sym_reciprocal_rank'] = 0.0

  for i in range(curr_log.shape[0]):
    descrip = curr_log.loc[i, 'Food Name']
    descrip_split = descrip.split(", ")

    # Reduce food log description to words in food_words
    first_phrase_num_words = reduce_with_food_words(descrip_split[0])

    pre_embedding = []

    if len(descrip_split) > 1:

      second_phrase_num_words = reduce_with_food_words(descrip_split[1])
      
      descrip_split_0_words = descrip_split[0].split()
      descrip_split_1_words = descrip_split[1].split()

      begin_first = False if len(second_phrase_num_words)/len(descrip_split_1_words) > len(first_phrase_num_words)/len(descrip_split_0_words) else True
      
      if begin_first:
        last_phrase = descrip

      if begin_first == False:
        three_phrases = descrip.partition(", ")
        last_phrase = three_phrases[2]

    else:
      last_phrase = descrip

    last_phrase_comma = last_phrase.split(", ")
    for j in range(len(last_phrase_comma)):
      split_comma = last_phrase_comma[j].split(" ")
      for elem in split_comma:
        elem = elem.lower()
        pre_embedding.append(elem)

    word_embed = np.zeros(shape = (1, len(model["sushi"])))

    if len(pre_embedding) > 0:
      # Turned edited_descrip into embedding vector by averaging the words
      word_embed = np.zeros(shape = (1, len(model["sushi"])))
      num_words = 0
      for word in pre_embedding:
        word = word.lower()
        if word in model:
          num_words += 1
          word_embed += model[word]

      if num_words != 0:
        word_embed /= num_words

    # Compare to the other vectors
    similarities = cosine_similarity(word_embed, wweia_embeddings) # (1, 7918) for each wweia food

    # Finding WWEIA foods with highest similarity with each
    # food log food. Row has sorted list of WWEIA food IDs 
    # for each food log food.
    to_keep = np.sort(similarities, axis=1)
    to_keep_args = np.argsort(similarities, axis=1)
    indices = np.flip(to_keep_args, axis = 1)
    
    sym_rank = 1000000

    # record RR here
    for index in range(indices.shape[1]):
      true_cat = curr_log.loc[i, 'wweia_food_category_code']
      if math.isnan(true_cat): continue
      if wweia_data.loc[indices[0,index], 'wweia_category_code'] == true_cat:
        rank = index
        rr = 1 / (rank+1)
        if sym_rank > rank:
          sym_rank = rank
          sym_rr = 1 / (rank+1)

        break
      else:
        if wweia_data.loc[indices[0,index], 'wweia_category_code'] in wweia_synonym_cats[true_cat]:
          if sym_rank > index:
            sym_rank = index
            sym_rr = 1 / (index+1)
    
    most_sim_food_index = indices[0,0]
    
    most_sim_food_row = wweia_data.iloc[most_sim_food_index,:]
    highest_cat_num = most_sim_food_row['wweia_category_code']
    
    highest_cat_words = wweia_food_categories.loc[wweia_food_categories['wweia_food_category_code'] == highest_cat_num, 'wweia_food_category_description']

    curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
    curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]
    curr_log.loc[i, 'max_cosim_score'] = np.array2string(to_keep[0,-5:])
    curr_log.loc[i, 'most_sim_food'] = most_sim_food_row['description']
    curr_log.loc[i, 'reciprocal_rank'] = rr
    curr_log.loc[i, 'sym_reciprocal_rank'] = sym_rr
    
  return curr_log


def get_metrics(curr_log, wweia_synonym_cats):

  # Will hold foods already seen so foods aren't double-counted
  # when calculating accuracy
  seen_foods = set()
  total_valid_foods = 0
  num_corr = 0
  num_sym = 0

  for i in range(curr_log.shape[0]):
    
    true_cat = curr_log.loc[i, 'wweia_food_category_code']
    false_cat = curr_log.loc[i, 'predicted_categories_number']

    prev_set_size = len(seen_foods)
    seen_foods.add(curr_log.loc[i, 'Food Name'])

    if math.isnan(true_cat): continue # food does not have valid category, i.e. antacid

    if len(seen_foods) > prev_set_size: # not a repeated food

      if true_cat == false_cat:
        num_corr += 1
        num_sym += 1

      else:
        if false_cat in wweia_synonym_cats[true_cat]:
          num_sym += 1
        
      total_valid_foods += 1

  acc = num_corr / total_valid_foods
  sym_acc = num_sym / total_valid_foods

  return acc, sym_acc


def get_food_prefs(food_log_categories, wweia_cat_nums_to_words):

  wweia_food_cats_list = wweia_food_categories['wweia_food_category_code'].tolist()

  # Make dict to hold the number of foods that belong to a
  # specific WWEIA category. Key is the broad food category.
  cats_dict = {}
  cats_dict["Protein"] = defaultdict(int)
  cats_dict["Vegetable"] = defaultdict(int)
  cats_dict["Grain"] = defaultdict(int)
  cats_dict["Fruit"] = defaultdict(int)
  cats_dict["Dairy"] = defaultdict(int)

  # Dict with lists containing cat codes for each broad food category
  category_mapping = { "Protein" : wweia_food_cats_list[14:34], "Vegetable" : wweia_food_cats_list[91:104],
  "Grain" : wweia_food_cats_list[35:81], "Fruit" : wweia_food_cats_list[82:90], "Dairy" : wweia_food_cats_list[0:13] }

  broad_categories_list = ["Protein", "Vegetable", "Grain", "Fruit", "Dairy"]

  for category_code in food_log_categories:

    if math.isnan(category_code): continue # food does not have valid category, i.e. antacid

    # Loop through each category code and add 1 to its dict entry
    # if it's the right one
    for broad_cat in broad_categories_list:
      if category_code in category_mapping[broad_cat]:
        cats_dict[broad_cat][category_code] += 1
        continue

  max_code_dict = {}
  
  for broad_cat in broad_categories_list:

    if len(list(cats_dict[broad_cat].keys())) > 0:
      max_code_dict[broad_cat] = max(cats_dict[broad_cat], key=cats_dict[broad_cat].get)

    else:
      max_code_dict[broad_cat] = "None of this category was eaten"

  return max_code_dict


def make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats):
  
  acc = 0
  sym_acc = 0

  for key in true_food_prefs:
    if true_food_prefs[key] == pred_food_prefs[key]:
      acc += 1
      sym_acc += 1
    else:
      if true_food_prefs[key] not in wweia_synonym_cats: continue
      elif pred_food_prefs[key] in wweia_synonym_cats[true_food_prefs[key]]:
        sym_acc += 1

  acc = acc / len(list(true_food_prefs.keys()))
  sym_acc = sym_acc / len(list(true_food_prefs.keys()))

  return acc, sym_acc


def get_pref_metrics(curr_log, n):
  
  true_pop_cats = get_most_pop_categories(curr_log, n, 'wweia_food_category_code')
  pref_pop_cats = get_most_pop_categories(curr_log, n, 'predicted_categories_number')

  true_pop_cats_sym = set()
  pref_pop_cats_sym = set()

  for elem in true_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      true_pop_cats_sym.add(sym_elem)

  for elem in pref_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      pref_pop_cats_sym.add(sym_elem)

  cats_common = set(true_pop_cats).intersection(set(pref_pop_cats))
  percent_common = len(cats_common) / n

  cats_common_sym = set(true_pop_cats_sym).intersection(set(pref_pop_cats_sym))
  percent_common_sym = min(1, len(cats_common_sym) / n)

  return percent_common, percent_common_sym
```

```python colab={"base_uri": "https://localhost:8080/"} id="PZo53R2SRew1" executionInfo={"status": "ok", "timestamp": 1635799981875, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82cd75c1-05b6-4111-e32a-11661dee3876"
# Main method

avg_acc = 0.0
avg_sym_acc = 0.0
avg_pref_acc = 0.0
avg_pref_sym_acc = 0.0
avg_percent_common = 0.0
avg_percent_common_sym = 0.0
avg_mean_rr = 0.0
avg_sym_mean_rr = 0.0

food_log_names = []
accs = []
sym_accs = []
pref_accs = []
pref_sym_accs = []
percent_common_list = []
percent_common_sym_list = []
mean_rrs = []
sym_mean_rrs = []

sample_food_logs = ['sample_food_log.csv']

for food_log_name in sample_food_logs:
  
  curr_log = pd.read_csv(food_log_name)

  with open('wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
  
  curr_log = process_food_log(curr_log, wweia_synonym_cats)

  food_log_names.append(food_log_name)

  # # Write to file saved in Google Drive folder
  # file_name = 'lfp_method4/' + food_log_name
  # curr_log.to_csv(file_name)

  # Compare to true food categories
  acc, sym_acc = get_metrics(curr_log, wweia_synonym_cats)
  avg_acc += acc
  avg_sym_acc += sym_acc
  accs.append(acc)
  sym_accs.append(sym_acc)
  print("Accuracy", acc, "Average Synonymous Accuracy:", sym_acc)

  # Count most popular categories and get food preferences
  with open('wweia_cat_nums_to_words.pickle', 'rb') as handle:
    wweia_cat_nums_to_words = pickle.load(handle)

  percent_common, percent_common_sym = get_pref_metrics(curr_log, 10)
  print("Percent Top N Categories Identified:", percent_common, "Percent Top N Synonymous Categories Identified:", percent_common_sym)
  avg_percent_common += percent_common
  avg_percent_common_sym += percent_common_sym
  percent_common_list.append(percent_common)
  percent_common_sym_list.append(percent_common_sym)

  true_food_prefs = get_food_prefs(curr_log.loc[:,'wweia_food_category_code'].to_list(), wweia_cat_nums_to_words)
  pred_food_prefs = get_food_prefs(curr_log.loc[:,'predicted_categories_number'].to_list(), wweia_cat_nums_to_words)

  # print(true_food_prefs)
  # print(pred_food_prefs)

  pref_acc, pref_sym_acc = make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats)
  avg_pref_acc += pref_acc
  avg_pref_sym_acc += pref_sym_acc
  pref_accs.append(pref_acc)
  pref_sym_accs.append(pref_sym_acc)
  print("Preference Accuracy:", pref_acc, "Preference Synonymous Accuracy:", pref_sym_acc)

  mean_rr = np.mean(curr_log.loc[:,'reciprocal_rank'])
  sym_mean_rr = np.mean(curr_log.loc[:,'sym_reciprocal_rank'])
  avg_mean_rr += mean_rr
  avg_sym_mean_rr += sym_mean_rr
  mean_rrs.append(mean_rr)
  sym_mean_rrs.append(sym_mean_rr)
  print("Mean Reciprocal Rank:", mean_rr, "Synonymous Mean Reciprocal Rank:", sym_mean_rr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="YOQz1O4jRspo" executionInfo={"status": "ok", "timestamp": 1635799982604, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5d5e24d1-ee36-4c9e-96a2-cfa1e3117947"
# Make table to show accuracies
df = pd.DataFrame(list(zip(food_log_names, accs, sym_accs, pref_accs, pref_sym_accs, percent_common_list, percent_common_sym_list, mean_rrs, sym_mean_rrs)),
               columns =['Food Log', 'Accuracies', "Synomynous Accuracies", "Preference Accuracies", "Synomynous Preference Accuracies", "Percent of Top N Categories Achieved", "Percent of Top N Categories Achieved Sym", "Mean RR", "Sym Mean RR"])
df
```

<!-- #region id="NKnPIuCTRpd-" -->
## Method 4
<!-- #endregion -->

```python id="S_aSYWXYRwFn"
unhelpful_words = ["with", "or", "cooked", "to", "and", "as", "ns", "cooking", "in", "made", "from", " ", "canned", "frozen", "type", "eaten", "of", "on", "fresh", "nfs", "ready", "strained", "style", "than", "prepared", "method", "stewed", "drained", "homemade", "home"]


# Only keep words in phrase that are in food_words

def reduce_with_food_words(comma_phrase):
  comma_phrase_reduced = [lemmatizer.lemmatize(word.lower()) for word in comma_phrase.split() if (lemmatizer.lemmatize(word.lower()) in food_words or word.lower() in food_words) and (lemmatizer.lemmatize(word.lower()) not in unhelpful_words or word.lower() not in unhelpful_words)]
  return comma_phrase_reduced


# Format of food names seems to be (Optional) Company, Food, Details about food
# run function to guess if first or second comma phrase has Food

def process_food_log(curr_log, wweia_synonym_cats):

  curr_log['predicted_categories_number'] = 0
  curr_log['predicted_categories_words'] = ""
  curr_log['max_cosim_score'] = 0
  curr_log['most_sim_food'] = ""
  curr_log['reciprocal_rank'] = 0.0
  curr_log['sym_reciprocal_rank'] = 0.0

  for i in range(curr_log.shape[0]):
    descrip = curr_log.loc[i, 'Food Name']
    descrip_split = descrip.split(", ")

    # Reduce food log description to words in food_words
    first_phrase_num_words = reduce_with_food_words(descrip_split[0])

    pre_embedding = []

    if len(descrip_split) > 1:

      second_phrase_num_words = reduce_with_food_words(descrip_split[1])
      
      descrip_split_0_words = descrip_split[0].split()
      descrip_split_1_words = descrip_split[1].split()

      begin_first = False if len(second_phrase_num_words)/len(descrip_split_1_words) > len(first_phrase_num_words)/len(descrip_split_0_words) else True
      
      if begin_first:
        last_phrase = descrip

      if begin_first == False:
        three_phrases = descrip.partition(", ")
        last_phrase = three_phrases[2]

    else:
      last_phrase = descrip

    last_phrase_comma = last_phrase.split(", ")
    for j in range(len(last_phrase_comma)):
      split_comma = last_phrase_comma[j].split(" ")
      for elem in split_comma:
        elem = elem.lower()
        if elem not in unhelpful_words:
          pre_embedding.append(elem)

    word_embed = np.zeros(shape = (1, len(model["sushi"])))

    if len(pre_embedding) > 0:
      # Turned edited_descrip into embedding vector by averaging the words
      word_embed = np.zeros(shape = (1, len(model["sushi"])))
      num_words = 0
      for word in pre_embedding:
        word = word.lower()
        if word in model:
          num_words += 1
          word_embed += model[word]

      if num_words != 0:
        word_embed /= num_words

    # Compare to the other vectors
    similarities = cosine_similarity(word_embed, wweia_embeddings) # (1, 7918) for each wweia food

    # Finding WWEIA foods with highest similarity with each
    # food log food. Row has sorted list of WWEIA food IDs 
    # for each food log food.
    to_keep = np.sort(similarities, axis=1)
    to_keep_args = np.argsort(similarities, axis=1)
    indices = np.flip(to_keep_args, axis = 1)
    
    sym_rank = 1000000

    # record RR here
    for index in range(indices.shape[1]):
      true_cat = curr_log.loc[i, 'wweia_food_category_code']
      if math.isnan(true_cat): continue
      
      if wweia_data.loc[indices[0,index], 'wweia_category_code'] == true_cat:
        rank = index
        rr = 1 / (rank+1)
        if sym_rank > rank:
          sym_rank = rank
          sym_rr = 1 / (rank+1)

        break
      else:
        
        if wweia_data.loc[indices[0,index], 'wweia_category_code'] in wweia_synonym_cats[true_cat]:
          if sym_rank > index:
            sym_rank = index
            sym_rr = 1 / (index+1)
    
    most_sim_food_index = indices[0,0]
    
    most_sim_food_row = wweia_data.iloc[most_sim_food_index,:]
    highest_cat_num = most_sim_food_row['wweia_category_code']
    
    highest_cat_words = wweia_food_categories.loc[wweia_food_categories['wweia_food_category_code'] == highest_cat_num, 'wweia_food_category_description']

    curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
    curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]
    curr_log.loc[i, 'max_cosim_score'] = np.array2string(to_keep[0,-5:])
    curr_log.loc[i, 'most_sim_food'] = most_sim_food_row['description']
    curr_log.loc[i, 'reciprocal_rank'] = rr
    curr_log.loc[i, 'sym_reciprocal_rank'] = sym_rr
    
  return curr_log


def get_metrics(curr_log, wweia_synonym_cats):

  # Will hold foods already seen so foods aren't double-counted
  # when calculating accuracy
  seen_foods = set()
  total_valid_foods = 0
  num_corr = 0
  num_sym = 0

  for i in range(curr_log.shape[0]):
    
    true_cat = curr_log.loc[i, 'wweia_food_category_code']
    false_cat = curr_log.loc[i, 'predicted_categories_number']

    prev_set_size = len(seen_foods)
    seen_foods.add(curr_log.loc[i, 'Food Name'])

    if math.isnan(true_cat): continue # food does not have valid category, i.e. antacid

    if len(seen_foods) > prev_set_size: # not a repeated food

      if true_cat == false_cat:
        num_corr += 1
        num_sym += 1

      else:
        if false_cat in wweia_synonym_cats[true_cat]:
          num_sym += 1
        
      total_valid_foods += 1

  acc = num_corr / total_valid_foods
  sym_acc = num_sym / total_valid_foods

  return acc, sym_acc


def get_food_prefs(food_log_categories, wweia_cat_nums_to_words):

  wweia_food_cats_list = wweia_food_categories['wweia_food_category_code'].tolist()

  # Make dict to hold the number of foods that belong to a
  # specific WWEIA category. Key is the broad food category.
  cats_dict = {}
  cats_dict["Protein"] = defaultdict(int)
  cats_dict["Vegetable"] = defaultdict(int)
  cats_dict["Grain"] = defaultdict(int)
  cats_dict["Fruit"] = defaultdict(int)
  cats_dict["Dairy"] = defaultdict(int)

  # Dict with lists containing cat codes for each broad food category
  category_mapping = { "Protein" : wweia_food_cats_list[14:34], "Vegetable" : wweia_food_cats_list[91:104],
  "Grain" : wweia_food_cats_list[35:81], "Fruit" : wweia_food_cats_list[82:90], "Dairy" : wweia_food_cats_list[0:13] }

  broad_categories_list = ["Protein", "Vegetable", "Grain", "Fruit", "Dairy"]

  for category_code in food_log_categories:

    if math.isnan(category_code): continue # food does not have valid category, i.e. antacid

    # Loop through each category code and add 1 to its dict entry
    # if it's the right one
    for broad_cat in broad_categories_list:
      if category_code in category_mapping[broad_cat]:
        cats_dict[broad_cat][category_code] += 1
        continue

  max_code_dict = {}
  
  for broad_cat in broad_categories_list:

    if len(list(cats_dict[broad_cat].keys())) > 0:
      max_code_dict[broad_cat] = max(cats_dict[broad_cat], key=cats_dict[broad_cat].get)

    else:
      max_code_dict[broad_cat] = "None of this category was eaten"

  return max_code_dict


def make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats):
  
  acc = 0
  sym_acc = 0

  for key in true_food_prefs:
    if true_food_prefs[key] == pred_food_prefs[key]:
      acc += 1
      sym_acc += 1
    else:
      if true_food_prefs[key] not in wweia_synonym_cats: continue
      elif pred_food_prefs[key] in wweia_synonym_cats[true_food_prefs[key]]:
        sym_acc += 1

  acc = acc / len(list(true_food_prefs.keys()))
  sym_acc = sym_acc / len(list(true_food_prefs.keys()))

  return acc, sym_acc

  
def get_pref_metrics(curr_log, n):
  
  true_pop_cats = get_most_pop_categories(curr_log, n, 'wweia_food_category_code')
  pref_pop_cats = get_most_pop_categories(curr_log, n, 'predicted_categories_number')

  true_pop_cats_sym = set()
  pref_pop_cats_sym = set()

  for elem in true_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      true_pop_cats_sym.add(sym_elem)

  for elem in pref_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      pref_pop_cats_sym.add(sym_elem)

  cats_common = set(true_pop_cats).intersection(set(pref_pop_cats))
  percent_common = len(cats_common) / n

  cats_common_sym = set(true_pop_cats_sym).intersection(set(pref_pop_cats_sym))
  percent_common_sym = min(1, len(cats_common_sym) / n)

  return percent_common, percent_common_sym
```

```python colab={"base_uri": "https://localhost:8080/"} id="CEZya0tjR_Jo" executionInfo={"status": "ok", "timestamp": 1635800090618, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="56daabf3-7f9b-4055-a7a0-f1d97be0432e"
# Main method

avg_acc = 0.0
avg_sym_acc = 0.0
avg_pref_acc = 0.0
avg_pref_sym_acc = 0.0
avg_percent_common = 0.0
avg_percent_common_sym = 0.0
avg_mean_rr = 0.0
avg_sym_mean_rr = 0.0

food_log_names = []
accs = []
sym_accs = []
pref_accs = []
pref_sym_accs = []
percent_common_list = []
percent_common_sym_list = []
mean_rrs = []
sym_mean_rrs = []

sample_food_logs = ['sample_food_log.csv']

for food_log_name in sample_food_logs:
  
  curr_log = pd.read_csv(food_log_name)

  with open('wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
  
  curr_log = process_food_log(curr_log, wweia_synonym_cats)

  food_log_names.append(food_log_name)

  # # Write to file saved in Google Drive folder
  # file_name = 'lfp_method4/' + food_log_name
  # curr_log.to_csv(file_name)

  # Compare to true food categories
  acc, sym_acc = get_metrics(curr_log, wweia_synonym_cats)
  avg_acc += acc
  avg_sym_acc += sym_acc
  accs.append(acc)
  sym_accs.append(sym_acc)
  print("Accuracy", acc, "Average Synonymous Accuracy:", sym_acc)

  # Count most popular categories and get food preferences
  with open('wweia_cat_nums_to_words.pickle', 'rb') as handle:
    wweia_cat_nums_to_words = pickle.load(handle)

  percent_common, percent_common_sym = get_pref_metrics(curr_log, 10)
  print("Percent Top N Categories Identified:", percent_common, "Percent Top N Synonymous Categories Identified:", percent_common_sym)
  avg_percent_common += percent_common
  avg_percent_common_sym += percent_common_sym
  percent_common_list.append(percent_common)
  percent_common_sym_list.append(percent_common_sym)

  true_food_prefs = get_food_prefs(curr_log.loc[:,'wweia_food_category_code'].to_list(), wweia_cat_nums_to_words)
  pred_food_prefs = get_food_prefs(curr_log.loc[:,'predicted_categories_number'].to_list(), wweia_cat_nums_to_words)

  # print(true_food_prefs)
  # print(pred_food_prefs)

  pref_acc, pref_sym_acc = make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats)
  avg_pref_acc += pref_acc
  avg_pref_sym_acc += pref_sym_acc
  pref_accs.append(pref_acc)
  pref_sym_accs.append(pref_sym_acc)
  print("Preference Accuracy:", pref_acc, "Preference Synonymous Accuracy:", pref_sym_acc)

  mean_rr = np.mean(curr_log.loc[:,'reciprocal_rank'])
  sym_mean_rr = np.mean(curr_log.loc[:,'sym_reciprocal_rank'])
  avg_mean_rr += mean_rr
  avg_sym_mean_rr += sym_mean_rr
  mean_rrs.append(mean_rr)
  sym_mean_rrs.append(sym_mean_rr)
  print("Mean Reciprocal Rank:", mean_rr, "Synonymous Mean Reciprocal Rank:", sym_mean_rr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="1KlGI7kAR0wC" executionInfo={"status": "ok", "timestamp": 1635800095046, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="431b5d94-322d-43c9-f6cd-a3e7125786aa"
# Make table to show accuracies
df = pd.DataFrame(list(zip(food_log_names, accs, sym_accs, pref_accs, pref_sym_accs, percent_common_list, percent_common_sym_list, mean_rrs, sym_mean_rrs)),
               columns =['Food Log', 'Accuracies', "Synomynous Accuracies", "Preference Accuracies", "Synomynous Preference Accuracies", "Percent of Top N Categories Achieved", "Percent of Top N Categories Achieved Sym", "Mean RR", "Sym Mean RR"])
df
```

<!-- #region id="ObVGbd-lSJQd" -->
## Method 5
<!-- #endregion -->

```python id="RtjTqbxESQ_m"
unhelpful_words = ["with", "or", "cooked", "to", "and", "as", "ns", "cooking", "in", "made", "from", " ", "canned", "frozen", "type", "eaten", "of", "on", "fresh", "nfs", "ready", "strained", "style", "than", "prepared", "method", "stewed", "drained", "homemade", "home"]


# Only keep words in phrase that are in food_words

def reduce_with_food_words(comma_phrase):
  comma_phrase_reduced = [lemmatizer.lemmatize(word.lower()) for word in comma_phrase.split() if (lemmatizer.lemmatize(word.lower()) in food_words or word.lower() in food_words) and (lemmatizer.lemmatize(word.lower()) not in unhelpful_words or word.lower() not in unhelpful_words)]
  return comma_phrase_reduced


# Format of food names seems to be (Optional) Company, Food, Details about food
# run function to guess if First or second comma phrase has Food

def process_food_log(curr_log, possible_cats_dict):

  curr_log['predicted_categories_number'] = 0
  curr_log['predicted_categories_words'] = ""
  curr_log['max_cosim_score'] = 0
  curr_log['most_sim_food'] = ""
  curr_log['reciprocal_rank'] = 0.0
  curr_log['sym_reciprocal_rank'] = 0.0

  for i in range(curr_log.shape[0]):
    descrip = curr_log.loc[i, 'Food Name']

    pre_embedding = []

    # Reduce food log description to words in food_words
    last_phrase = reduce_with_food_words(descrip)
    for elem in last_phrase:
      pre_embedding.append(elem)

    word_embed = np.zeros(shape = (1, len(model["sushi"])))

    if len(pre_embedding) > 0:
      # Turned edited_descrip into embedding vector by averaging the words
      word_embed = np.zeros(shape = (1, len(model["sushi"])))
      num_words = 0
      for word in pre_embedding:
        word = word.lower()
        if word in model:
          num_words += 1
          word_embed += model[word]

      if num_words != 0:
        word_embed /= num_words

    # Compare to the other vectors
    similarities = cosine_similarity(word_embed, wweia_embeddings) # (1, 7918) for each wweia food

    # Finding WWEIA foods with highest similarity with each
    # food log food. Row has sorted list of WWEIA food IDs 
    # for each food log food.
    to_keep = np.sort(similarities, axis=1)
    to_keep_args = np.argsort(similarities, axis=1)
    indices = np.flip(to_keep_args, axis = 1)
    
    sym_rank = 1000000

    # record RR here
    for index in range(indices.shape[1]):
      true_cat = curr_log.loc[i, 'wweia_food_category_code']
      if math.isnan(true_cat): continue
      if wweia_data.loc[indices[0,index], 'wweia_category_code'] == true_cat:
        rank = index
        rr = 1 / (rank+1)
        if sym_rank > rank:
          sym_rank = rank
          sym_rr = 1 / (rank+1)

        break
      else:
        if wweia_data.loc[indices[0,index], 'wweia_category_code'] in wweia_synonym_cats[true_cat]:
          if sym_rank > index:
            sym_rank = index
            sym_rr = 1 / (index+1)
    
    most_sim_food_index = indices[0,0]
    
    most_sim_food_row = wweia_data.iloc[most_sim_food_index,:]
    highest_cat_num = most_sim_food_row['wweia_category_code']
    
    highest_cat_words = wweia_food_categories.loc[wweia_food_categories['wweia_food_category_code'] == highest_cat_num, 'wweia_food_category_description']

    curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
    curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]
    curr_log.loc[i, 'max_cosim_score'] = np.array2string(to_keep[0,-5:])
    curr_log.loc[i, 'most_sim_food'] = most_sim_food_row['description']
    curr_log.loc[i, 'reciprocal_rank'] = rr
    curr_log.loc[i, 'sym_reciprocal_rank'] = sym_rr
    
  return curr_log


def get_metrics(curr_log, wweia_synonym_cats):

  # Will hold foods already seen so foods aren't double-counted
  # when calculating accuracy
  seen_foods = set()
  total_valid_foods = 0
  num_corr = 0
  num_sym = 0

  for i in range(curr_log.shape[0]):
    
    true_cat = curr_log.loc[i, 'wweia_food_category_code']
    false_cat = curr_log.loc[i, 'predicted_categories_number']

    prev_set_size = len(seen_foods)
    seen_foods.add(curr_log.loc[i, 'Food Name'])

    if math.isnan(true_cat): continue # food does not have valid category, i.e. antacid

    if len(seen_foods) > prev_set_size: # not a repeated food

      if true_cat == false_cat:
        num_corr += 1
        num_sym += 1

      else:
        if false_cat in wweia_synonym_cats[true_cat]:
          num_sym += 1
        
      total_valid_foods += 1

  acc = num_corr / total_valid_foods
  sym_acc = num_sym / total_valid_foods

  return acc, sym_acc


def get_food_prefs(food_log_categories, wweia_cat_nums_to_words):

  wweia_food_cats_list = wweia_food_categories['wweia_food_category_code'].tolist()

  # Make dict to hold the number of foods that belong to a
  # specific WWEIA category. Key is the broad food category.
  cats_dict = {}
  cats_dict["Protein"] = defaultdict(int)
  cats_dict["Vegetable"] = defaultdict(int)
  cats_dict["Grain"] = defaultdict(int)
  cats_dict["Fruit"] = defaultdict(int)
  cats_dict["Dairy"] = defaultdict(int)

  # Dict with lists containing cat codes for each broad food category
  category_mapping = { "Protein" : wweia_food_cats_list[14:34], "Vegetable" : wweia_food_cats_list[91:104],
  "Grain" : wweia_food_cats_list[35:81], "Fruit" : wweia_food_cats_list[82:90], "Dairy" : wweia_food_cats_list[0:13] }

  broad_categories_list = ["Protein", "Vegetable", "Grain", "Fruit", "Dairy"]

  for category_code in food_log_categories:

    if math.isnan(category_code): continue # food does not have valid category, i.e. antacid

    # Loop through each category code and add 1 to its dict entry
    # if it's the right one
    for broad_cat in broad_categories_list:
      if category_code in category_mapping[broad_cat]:
        cats_dict[broad_cat][category_code] += 1
        continue

  max_code_dict = {}
  
  for broad_cat in broad_categories_list:

    if len(list(cats_dict[broad_cat].keys())) > 0:
      max_code_dict[broad_cat] = max(cats_dict[broad_cat], key=cats_dict[broad_cat].get)

    else:
      max_code_dict[broad_cat] = "None of this category was eaten"

  return max_code_dict


def make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats):
  
  acc = 0
  sym_acc = 0

  for key in true_food_prefs:
    if true_food_prefs[key] == pred_food_prefs[key]:
      acc += 1
      sym_acc += 1
    else:
      if true_food_prefs[key] not in wweia_synonym_cats: continue
      elif pred_food_prefs[key] in wweia_synonym_cats[true_food_prefs[key]]:
        sym_acc += 1

  acc = acc / len(list(true_food_prefs.keys()))
  sym_acc = sym_acc / len(list(true_food_prefs.keys()))

  return acc, sym_acc


def get_pref_metrics(curr_log, n):
  
  true_pop_cats = get_most_pop_categories(curr_log, n, 'wweia_food_category_code')
  pref_pop_cats = get_most_pop_categories(curr_log, n, 'predicted_categories_number')

  true_pop_cats_sym = set()
  pref_pop_cats_sym = set()

  for elem in true_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      true_pop_cats_sym.add(sym_elem)

  for elem in pref_pop_cats:
    for sym_elem in wweia_synonym_cats[elem]:
      pref_pop_cats_sym.add(sym_elem)

  cats_common = set(true_pop_cats).intersection(set(pref_pop_cats))
  percent_common = len(cats_common) / n

  cats_common_sym = set(true_pop_cats_sym).intersection(set(pref_pop_cats_sym))
  percent_common_sym = min(1, len(cats_common_sym) / n)

  return percent_common, percent_common_sym
```

```python colab={"base_uri": "https://localhost:8080/"} id="yK-xB322SWyi" executionInfo={"status": "ok", "timestamp": 1635800223142, "user_tz": -330, "elapsed": 620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="394c4b67-c704-4122-9588-cd2ac59d6aa5"
# Main method

avg_acc = 0.0
avg_sym_acc = 0.0
avg_pref_acc = 0.0
avg_pref_sym_acc = 0.0
avg_percent_common = 0.0
avg_percent_common_sym = 0.0
avg_mean_rr = 0.0
avg_sym_mean_rr = 0.0

food_log_names = []
accs = []
sym_accs = []
pref_accs = []
pref_sym_accs = []
percent_common_list = []
percent_common_sym_list = []
mean_rrs = []
sym_mean_rrs = []

sample_food_logs = ['sample_food_log.csv']

for food_log_name in sample_food_logs:
  
  curr_log = pd.read_csv(food_log_name)

  with open('wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
  
  curr_log = process_food_log(curr_log, wweia_synonym_cats)

  food_log_names.append(food_log_name)

  # # Write to file saved in Google Drive folder
  # file_name = 'lfp_method4/' + food_log_name
  # curr_log.to_csv(file_name)

  # Compare to true food categories
  acc, sym_acc = get_metrics(curr_log, wweia_synonym_cats)
  avg_acc += acc
  avg_sym_acc += sym_acc
  accs.append(acc)
  sym_accs.append(sym_acc)
  print("Accuracy", acc, "Average Synonymous Accuracy:", sym_acc)

  # Count most popular categories and get food preferences
  with open('wweia_cat_nums_to_words.pickle', 'rb') as handle:
    wweia_cat_nums_to_words = pickle.load(handle)

  percent_common, percent_common_sym = get_pref_metrics(curr_log, 10)
  print("Percent Top N Categories Identified:", percent_common, "Percent Top N Synonymous Categories Identified:", percent_common_sym)
  avg_percent_common += percent_common
  avg_percent_common_sym += percent_common_sym
  percent_common_list.append(percent_common)
  percent_common_sym_list.append(percent_common_sym)

  true_food_prefs = get_food_prefs(curr_log.loc[:,'wweia_food_category_code'].to_list(), wweia_cat_nums_to_words)
  pred_food_prefs = get_food_prefs(curr_log.loc[:,'predicted_categories_number'].to_list(), wweia_cat_nums_to_words)

  # print(true_food_prefs)
  # print(pred_food_prefs)

  pref_acc, pref_sym_acc = make_food_prefs_table(true_food_prefs, pred_food_prefs, wweia_synonym_cats)
  avg_pref_acc += pref_acc
  avg_pref_sym_acc += pref_sym_acc
  pref_accs.append(pref_acc)
  pref_sym_accs.append(pref_sym_acc)
  print("Preference Accuracy:", pref_acc, "Preference Synonymous Accuracy:", pref_sym_acc)

  mean_rr = np.mean(curr_log.loc[:,'reciprocal_rank'])
  sym_mean_rr = np.mean(curr_log.loc[:,'sym_reciprocal_rank'])
  avg_mean_rr += mean_rr
  avg_sym_mean_rr += sym_mean_rr
  mean_rrs.append(mean_rr)
  sym_mean_rrs.append(sym_mean_rr)
  print("Mean Reciprocal Rank:", mean_rr, "Synonymous Mean Reciprocal Rank:", sym_mean_rr)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="I6e2-Oz-SSjX" executionInfo={"status": "ok", "timestamp": 1635800226976, "user_tz": -330, "elapsed": 632, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f8a773f4-8871-4f70-ef72-83dcccc0c558"
# Make table to show accuracies
df = pd.DataFrame(list(zip(food_log_names, accs, sym_accs, pref_accs, pref_sym_accs, percent_common_list, percent_common_sym_list, mean_rrs, sym_mean_rrs)),
               columns =['Food Log', 'Accuracies', "Synomynous Accuracies", "Preference Accuracies", "Synomynous Preference Accuracies", "Percent of Top N Categories Achieved", "Percent of Top N Categories Achieved Sym", "Mean RR", "Sym Mean RR"])
df
```
