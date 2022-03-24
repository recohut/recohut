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

<!-- #region id="W8Z8LeiP64Tg" -->
# Text Analysis

Rapid text analysis can save lives. Let’s consider a real-world incident when US soldiers stormed a terrorist compound. In the compound, they discovered a computer containing terabytes of archived data. The data included documents, text messages, and emails pertaining to terrorist activities. The documents were too numerous to be read by any single human being. Fortunately, the soldiers were equipped with special software that could perform very fast text analysis. The software allowed the soldiers to process all of the text data without even having to leave the compound. The onsite analysis immediately revealed an active terrorist plot in a nearby neighborhood. The soldiers instantly responded to the plot and prevented a terrorist attack.

This swift defensive response would not have been possible without *natural language processing* (NLP) techniques. NLP is a branch of data science that focuses on speedy text analysis. Typically, NLP is applied to very large text datasets. NLP use cases are numerous and diverse and include the following:

- Corporate monitoring of social media posts to measure the public’s sentiment toward a company’s brand
- Analyzing transcribed call center conversations to monitor common customer complaints
- Matching people on dating sites based on written descriptions of shared interests
- Processing written doctors’ notes to ensure proper patient diagnosis

These use cases depend on fast analysis. Delayed signal extraction could be costly. Unfortunately, the direct handling of text is an inherently slow process. Most computational techniques are optimized for numbers, not text. Consequently, NLP methods depend on a conversion from pure text to a numeric representation. Once all words and sentences have been replaced with numbers, the data can be analyzed very rapidly.
<!-- #endregion -->

<!-- #region id="x9qStg1xNMm-" -->
## Measuring Text Similarity
<!-- #endregion -->

```python id="p-WdtuW1Lzl5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476276435, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5b17ac38-d9a6-4b74-fe66-65a3287ed072"
text1 = 'She sells seashells by the seashore.'
text2 = '"Seashells! The seashells are on sale! By the seashore."'
text3 = 'She sells 3 seashells to John, who lives by the lake.'
words_lists = [text.split() for text in [text1, text2, text3]]
words1, words2, words3 = words_lists

for i, words in enumerate(words_lists, 1):
    print(f"Words in text {i}")
    print(f"{words}\n")
```

```python id="D5zaIYD8Lzl-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476302067, "user_tz": -330, "elapsed": 434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="34db9041-3c77-4aab-9262-b0f594d2bfe7"
def simplify_text(text):
    for punctuation in ['.', ',', '!', '?', '"']:
        text = text.replace(punctuation, '')

    return text.lower()

for i, words in enumerate(words_lists, 1):
    for j, word in enumerate(words):
        words[j] = simplify_text(word)

    print(f"Words in text {i}")
    print(f"{words}\n")
```

```python id="pP_gvmpeLzmB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476308884, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="044a5da1-3fc5-4751-c44a-153bbb80c22e"
words_sets = [set(words) for words in words_lists]
for i, unique_words in enumerate(words_sets, 1):
    print(f"Unique Words in text {i}")
    print(f"{unique_words}\n")
```

```python id="RMb-zTvLLzmC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476316778, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a7e7ac55-5ed5-44f8-e88d-4c40ebb0c4fa"
words_sets
```

```python id="Q1wBnkveLzmD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476336656, "user_tz": -330, "elapsed": 415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8261b0bb-d14e-484c-d1c0-5cce113c3208"
words_set1 = words_sets[0]
for i, words_set in enumerate(words_sets[1:], 2):
    shared_words = words_set1 & words_set
    print(f"Texts 1 and {i} share these {len(shared_words)} words:")
    print(f"{shared_words}\n")
```

```python id="TH-xFq9hLzmF" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476344975, "user_tz": -330, "elapsed": 487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="850c01e1-f375-47f9-ac84-0a04430d2d9f"
for i, words_set in enumerate(words_sets[1:], 2):
    diverging_words = words_set1 ^ words_set
    print(f"Texts 1 and {i} don't share these {len(diverging_words)} words:")
    print(f"{diverging_words}\n")
```

```python id="sQOkzE2GLzmH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476350146, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f052b4a-74aa-488b-f3a7-84f76a79bb20"
for i, words_set in enumerate(words_sets[1:], 2):
    total_words = words_set1 | words_set
    print(f"Together, texts 1 and {i} contain {len(total_words)} "
          f"unique words. These words are:\n {total_words}\n")
```

```python id="2LTGvcvPLzmJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476424729, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c5bf0cc-1b17-4dbe-d36d-1b01fc597d50"
for i, words_set in enumerate(words_sets[1:], 2):
    shared_words = words_set1 & words_set
    diverging_words = words_set1 ^ words_set
    total_words = words_set1 | words_set
    assert len(total_words) == len(shared_words) + len(diverging_words)
    percent_shared = 100 * len(shared_words) / len(total_words)
    percent_diverging = 100 * len(diverging_words) / len(total_words)

    print(f"Together, texts 1 and {i} contain {len(total_words)} "
          f"unique words. \n{percent_shared:.2f}% of these words are "
          f"shared. \n{percent_diverging:.2f}% of these words diverge.\n")
```

```python id="t_8CBfA5LzmL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476428377, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="290cbb69-5097-45aa-9cf2-f612ec3a13f8"
def jaccard_similarity(text_a, text_b):
    word_set_a, word_set_b = [set(simplify_text(text).split())
                              for text in [text_a, text_b]]
    num_shared = len(word_set_a & word_set_b)
    num_total = len(word_set_a | word_set_b)
    return num_shared / num_total

for text in [text2, text3]:
    similarity = jaccard_similarity(text1, text)
    print(f"The Jaccard similarity between '{text1}' and '{text}' "
          f"equals {similarity:.4f}." "\n")
```

```python id="2-ChygDcLzmN"
def jaccard_similarity_efficient(text_a, text_b):
    word_set_a, word_set_b = [set(simplify_text(text).split())
                              for text in [text_a, text_b]]
    num_shared = len(word_set_a & word_set_b)
    num_total = len(word_set_a) + len(word_set_b) - num_shared
    return num_shared / num_total

for text in [text2, text3]:
    similarity = jaccard_similarity_efficient(text1, text)
    assert similarity == jaccard_similarity(text1, text)
```

```python id="rJ_Ks8QoLzmO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476441013, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b7d72dfa-0eaf-486e-ebb8-5d8dadf10a7e"
words_set1, words_set2, words_set3 = words_sets
total_words = words_set1 | words_set2 | words_set3
vocabulary = {word : i for i, word in enumerate(total_words)}
value_to_word = {value: word for word, value in vocabulary.items()}
print(f"Our vocabulary contains {len(vocabulary)} words. "
      f"This vocabulary is:\n{vocabulary}")
```

```python id="yoIbE6g7LzmQ" colab={"base_uri": "https://localhost:8080/", "height": 306} executionInfo={"status": "ok", "timestamp": 1637476451835, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f92f0d07-a1dd-4cb2-dd6d-cd295ebed120"
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

vectors = []
for i, words_set in enumerate(words_sets, 1):
    vector = np.array([0] * len(vocabulary))
    for word in words_set:
        vector[vocabulary[word]] = 1
    vectors.append(vector)

sns.heatmap(vectors, annot=True,  cmap='YlGnBu',
            xticklabels=vocabulary.keys(),
yticklabels=['Text 1', 'Text 2', 'Text 3'])
plt.yticks(rotation=0)
plt.show()
```

```python id="f8A9L5JcLzmR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476473400, "user_tz": -330, "elapsed": 652, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0fc82652-d063-4de8-81e3-2feef584feed"
vector1, vector2 = vectors[:2]
for i in range(len(vocabulary)):
    if vector1[i] * vector2[i]:
        shared_word = value_to_word[i]
        print(f"'{shared_word}' is present in both texts 1 and 2")
```

```python id="NymtsVMaLzmT"
def tanimoto_similarity(vector_a, vector_b):
    num_shared = vector_a @ vector_b
    num_total = vector_a @ vector_a + vector_b @ vector_b - num_shared
    return num_shared / num_total

for i, text in enumerate([text2, text3], 1):
    similarity = tanimoto_similarity(vector1, vectors[i])
    assert similarity == jaccard_similarity(text1, text)
```

```python id="npwlALhLLzmU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476479503, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5896a0d4-071b-4cd2-ebd7-23b7860f3f50"
non_binary_vector1 = np.array([5, 3])
non_binary_vector2 = np.array([5, 2])
similarity = tanimoto_similarity(non_binary_vector1, non_binary_vector2)
print(f"The similarity of 2 non-binary vectors is {similarity}")
```

<!-- #region id="BdiQv9V2LzmU" -->
## Vectorizing Texts Using Word Counts
<!-- #endregion -->

```python id="ja558cq2LzmV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476545929, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba69730e-e1ac-4397-b731-a255d556bde3"
similarity = tanimoto_similarity(np.array([61, 2]), np.array([1, 71]))
print(f"The similarity between texts is approximately {similarity:.3f}")
```

```python id="6tT6afkQLzmV"
assert tanimoto_similarity(np.array([1, 1]), np.array([1, 1])) == 1
```

```python id="NKaVShYYLzmW" colab={"base_uri": "https://localhost:8080/", "height": 306} executionInfo={"status": "ok", "timestamp": 1637476547679, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7bdccbba-51a4-418e-e3e9-56b397a8a4aa"
tf_vectors = []
for i, words_list in enumerate(words_lists, 1):
    tf_vector = np.array([0] * len(vocabulary))
    for word in words_list:
        word_index = vocabulary[word]
        # Update the count of each word using its vocabulary index.
        tf_vector[word_index] += 1

    tf_vectors.append(tf_vector)


sns.heatmap(tf_vectors,  cmap='YlGnBu', annot=True,
            xticklabels=vocabulary.keys(),
yticklabels=['Text 1', 'Text 2', 'Text 3'])
plt.yticks(rotation=0)
plt.show()
```

```python id="2sN6RcKuLzmX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476555239, "user_tz": -330, "elapsed": 879, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9330424-cf82-41a3-ce66-1d6ba8344d7e"
tf_vector1 = tf_vectors[0]
binary_vector1 = vectors[0]

for i, tf_vector in enumerate(tf_vectors[1:], 2):
    similarity = tanimoto_similarity(tf_vector1, tf_vector)
    old_similarity = tanimoto_similarity(binary_vector1, vectors[i - 1])
    print(f"The recomputed Tanimoto similarity between texts 1 and {i} is"
          f" {similarity:.4f}.")
    print(f"Previously, that similarity equaled {old_similarity:.4f} " "\n")
```

```python id="aom-kVSuLzmX"
query_vector = np.array([1, 1])
title_a_vector = np.array([3, 3])
title_b_vector = np.array([1, 0])
```

```python id="rXJXyT_DLzmX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476563355, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17cebcbc-234a-4547-9ce2-6a230c9f5551"
titles = ["A: Pepperoni Pizza! Pepperoni Pizza! Pepperoni Pizza!",
          "B: Pepperoni"]
title_vectors = [title_a_vector, title_b_vector]
similarities = [tanimoto_similarity(query_vector, title_vector)
                for title_vector in title_vectors]

for index in sorted(range(len(titles)), key=lambda i: similarities[i],
                    reverse=True):
    title = titles[index]
    similarity = similarities[index]
    print(f"'{title}' has a query similarity of {similarity:.4f}")
```

```python id="JiQbls63LzmY"
assert np.array_equal(query_vector, title_a_vector / 3)
assert tanimoto_similarity(query_vector,
                           title_a_vector / 3) == 1
```

```python id="v_VWz5nXLzmY" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476581364, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="09f76b1e-f39c-4eb6-ee50-000d9bf0a41f"
plt.plot([0, query_vector[0]], [0, query_vector[1]], c='k',
         linewidth=3, label='Query Vector')
plt.plot([0, title_a_vector[0]], [0, title_a_vector[1]], c='b',
          linestyle='--', label='Title A Vector')
plt.plot([0, title_b_vector[0]], [0, title_b_vector[1]], c='g',
         linewidth=2, linestyle='-.', label='Title B Vector')
plt.xlabel('Pepperoni')
plt.ylabel('Pizza')
plt.legend()
plt.show()
```

```python id="JOQZUi1ZLzmZ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476587223, "user_tz": -330, "elapsed": 825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aa13ef9b-bbe7-449a-b7a6-2efdc7500d6b"
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

vector_names = ['Query Vector', 'Title A Vector', 'Title B Vector']
tf_search_vectors = [query_vector, title_a_vector, title_b_vector]
origin = np.array([0, 0])
for name, tf_vector in zip(vector_names, tf_search_vectors):
    magnitude = euclidean(tf_vector, origin)
    assert magnitude == norm(tf_vector)
    assert magnitude == (tf_vector @ tf_vector) ** 0.5
    print(f"{name}'s magnitude is approximately {magnitude:.4f}")

magnitude_ratio = norm(title_a_vector) / norm(query_vector)
print(f"\nVector A is {magnitude_ratio:.0f}x as long as Query Vector")
```

```python id="AuLK2J6oLzmZ" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476595185, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9790e075-f1a3-41c6-a2e5-24d7139b72ce"
unit_query_vector = query_vector / norm(query_vector)
unit_title_a_vector = title_a_vector / norm(title_a_vector)
np.array_equal(unit_query_vector, unit_title_a_vector)
unit_title_b_vector = title_b_vector

plt.plot([0, unit_query_vector[0]], [0, unit_query_vector[1]], c='k',
         linewidth=3, label='Normalized Query Vector')
plt.plot([0, unit_title_a_vector[0]], [0, unit_title_a_vector[1]], c='b',
          linestyle='--', label='Normalized Title A Vector')
plt.plot([0, unit_title_b_vector[0]], [0, unit_title_b_vector[1]], c='g',
         linewidth=2, linestyle='-.', label='Title B Vector')

plt.axis('equal')
plt.legend()
plt.show()
```

```python id="MXoAEtjQLzma" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476670328, "user_tz": -330, "elapsed": 841, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35ee2c65-8bd6-4e5a-be39-9285e463ade0"
unit_title_vectors = [unit_title_a_vector, unit_title_b_vector]
similarities = [tanimoto_similarity(unit_query_vector, unit_title_vector)
                for unit_title_vector in unit_title_vectors]

for index in sorted(range(len(titles)), key=lambda i: similarities[i],
                    reverse=True):
    title = titles[index]
    similarity = similarities[index]
    print(f"'{title}' has a normalized query similarity of {similarity:.4f}")
```

```python id="TNdScxoCLzma"
def normalized_tanimoto(u1, u2):
    dot_product = u1 @ u2
    return dot_product / (2 - dot_product)

for unit_title_vector in unit_title_vectors[1:]:
    similarity = normalized_tanimoto(unit_query_vector, unit_title_vector)
    assert similarity == tanimoto_similarity(unit_query_vector,
                                             unit_title_vector)
```

```python id="ZoGZmpufLzmb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476673945, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="551a5259-635a-428f-a7c1-daccd189a772"
unit_vector_names = ['Normalized Title A vector', 'Title B Vector']
u1 = unit_query_vector

for unit_vector_name, u2 in zip(unit_vector_names, unit_title_vectors):
    similarity = normalized_tanimoto(u1, u2)
    cosine_similarity  = 2 * similarity / (1 + similarity)
    assert cosine_similarity == u1 @ u2
    angle = np.arccos(cosine_similarity)
    euclidean_distance = (2 - 2 * cosine_similarity) ** 0.5
    assert round(euclidean_distance, 10) == round(euclidean(u1, u2), 10)
    measurements = {'Tanimoto similarity': similarity,
                    'cosine similarity': cosine_similarity,
                    'Euclidean distance': euclidean_distance,
                    'angle': np.degrees(angle)}

    print("We are comparing Normalized Query Vector and "
           f"{unit_vector_name}")
    for measurement_type, value in measurements.items():
        output = f"The {measurement_type} between vectors is {value:.4f}"
        if measurement_type == 'angle':
            output += ' degrees\n'

        print(output)
```

<!-- #region id="gH6GMCGvLzmb" -->
## Matrix Multiplication for Efficient Similarity Calculation
<!-- #endregion -->

```python id="uD_nBmUSLzmb" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1637476687180, "user_tz": -330, "elapsed": 714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e4aed14-c657-46eb-8d1e-cfc31cfb0490"
num_texts = len(tf_vectors)
similarities = np.array([[0.0] * num_texts for _ in range(num_texts)])
similarities = np.zeros((num_texts, num_texts))
unit_vectors = np.array([vector / norm(vector) for vector in tf_vectors])
for i, vector_a in enumerate(unit_vectors):
    for j, vector_b in enumerate(unit_vectors):
        similarities[i][j] = normalized_tanimoto(vector_a, vector_b)

labels = ['Text 1', 'Text 2', 'Text 3']
sns.heatmap(similarities,  cmap='YlGnBu', annot=True,
            xticklabels=labels, yticklabels=labels)
plt.yticks(rotation=0)
plt.show()
```

```python id="GfOOzHXSLzmc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476689853, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ca6b9699-6730-447d-addf-8a2b007aba78"
import pandas as pd

matrices = [unit_vectors, pd.DataFrame(unit_vectors)]
matrix_types = ['Pandas DataFrame', '2D NumPy array']

for matrix_type, matrix in zip(matrix_types, matrices):
    row_count, column_count = matrix.shape
    print(f"Our {matrix_type} contains "
          f"{row_count} rows and {column_count} columns")
    assert (column_count, row_count) == matrix.T.shape
```

```python id="GifvwAAjLzmc"
double_similarites = 2 * similarities
np.array_equal(double_similarites, similarities + similarities)
zero_matrix = similarities - similarities
negative_1_matrix = similarities - similarities - 1

for i in range(similarities.shape[0]):
    for j in range(similarities.shape[1]):
        assert double_similarites[i][j] == 2 * similarities[i][j]
        assert zero_matrix[i][j] == 0
        assert negative_1_matrix[i][j] == -1
```

```python id="m3cXxr9CLzmd"
squared_similarities = similarities * similarities
assert np.array_equal(squared_similarities, similarities ** 2)
ones_matrix = similarities / similarities

for i in range(similarities.shape[0]):
    for j in range(similarities.shape[1]):
        assert squared_similarities[i][j] == similarities[i][j] ** 2
        assert ones_matrix[i][j] == 1
```

```python id="2sFEspLtLzmd"
cosine_similarities  = 2 * similarities / (1 + similarities)
for i in range(similarities.shape[0]):
    for j in range(similarities.shape[1]):
        cosine_sim = unit_vectors[i] @ unit_vectors[j]
        assert round(cosine_similarities[i][j],
                     15) == round(cosine_sim, 15)
```

```python id="WrGOwujjLzmd" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476698894, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d8c511d5-60d5-4ce5-ecf4-9ad341b2fed5"
for name, matrix in [('Similarities', similarities),
                     ('Unit Vectors', unit_vectors)]:
    print(f"Accessing rows and columns in the {name} Matrix.")
    row, column = matrix[1], matrix[:,1]
    print(f"Row at index 0 is:\n{row}")
    print(f"\nColumn at index 0 is:\n{column}\n")
```

```python id="vGzju6VkLzmh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476714397, "user_tz": -330, "elapsed": 869, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cdf0c35a-89dd-4467-cdb5-1fc935a3a777"
row = similarities[0]
column = unit_vectors[:,0]
dot_product = row @ column
print(f"The dot product between the row and column is: {dot_product:.4f}")
```

```python id="nDnqxMu6Lzmi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476718430, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fe19ce46-ade8-4c0c-efac-fc45b16647e8"
num_rows = similarities.shape[0]
num_columns = unit_vectors.shape[1]
for i in range(num_rows):
    for j in range(num_columns):
        row = similarities[i]
        column = unit_vectors[:,j]
        dot_product = row @ column
        print(f"The dot product between row {i} column {j} is: "
              f"{dot_product:.4f}")
```

```python id="QIS2-EDMLzmj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476723524, "user_tz": -330, "elapsed": 763, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2b417755-fe7c-415b-b854-946282490428"
dot_products = np.zeros((num_rows, num_columns))
for i in range(num_rows):
    for j in range(num_columns):
        dot_products[i][j] = similarities[i] @ unit_vectors[:,j]

print(dot_products)
```

```python id="N864AC2cLzmk"
matrix_product = similarities @ unit_vectors
assert np.allclose(matrix_product, dot_products)
```

```python id="6uc8323iLzml" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476726889, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5b11528-cb53-4885-d52f-e6fd3f9f5f44"
try:
    matrix_product = unit_vectors @ similarities
except:
    print("We can't compute the matrix product")
```

```python id="KwTnRuBMLzml"
matrix_product = np.matmul(similarities, unit_vectors)
assert np.array_equal(matrix_product,
                      similarities @ unit_vectors)
```

```python id="WeFZPq7TLzmm" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476733084, "user_tz": -330, "elapsed": 1600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="67894dca-9eae-4692-dabd-48bb5d3e51f6"
import time

numpy_run_times = []
for_loop_run_times = []

matrix_sizes = range(1, 101)
for size in matrix_sizes:
    matrix = np.ones((size, size))

    start_time = time.time()
    matrix @ matrix
    numpy_run_times.append(time.time() - start_time)

    start_time = time.time()
    for i in range(size):
        for j in range(size):
            matrix[i] @ matrix[:,j]

    for_loop_run_times.append(time.time() - start_time)

plt.plot(matrix_sizes, numpy_run_times,
         label='NumPy Matrix Product', linestyle='--')
plt.plot(matrix_sizes, for_loop_run_times,
         label='For-Loop Matrix Product', color='k')
plt.xlabel('Row / Column Size')
plt.ylabel('Running Time (Seconds)')
plt.legend()
plt.show()
```

```python id="Vdc68o4rLzmn"
cosine_matrix = unit_vectors @ unit_vectors.T
assert np.allclose(cosine_matrix, cosine_similarities)
```

```python id="rOmWLcobLzmn"
tanimoto_matrix = cosine_matrix / (2 - cosine_matrix)
assert np.allclose(tanimoto_matrix, similarities)
```

```python id="CcsE41XwLzmo"
output = normalized_tanimoto(unit_vectors, unit_vectors.T)
assert np.array_equal(output, tanimoto_matrix)
```

```python id="esX_gSTYLzmo" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476750528, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c618c68e-7379-417b-865a-2ed3e1d87941"
vocabulary_size = 50000
normalized_vector = [1 / vocabulary_size] * vocabulary_size
book_count = 30

def measure_run_time(book_count):
    book_matrix = np.array([normalized_vector] * book_count)
    start_time = time.time()
    normalized_tanimoto(book_matrix, book_matrix.T)
    return time.time() - start_time

run_time = measure_run_time(book_count)
print(f"It took {run_time:.4f} seconds to compute the similarities across a "
      f"{book_count}-book by {vocabulary_size}-word matrix")
```

```python id="huenUPCMLzmp" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476835695, "user_tz": -330, "elapsed": 57580, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a25890c8-73f8-4fad-eb12-f84af2fd5e32"
book_counts = range(30, 1000, 30)
run_times = [measure_run_time(book_count)
             for book_count in book_counts]
plt.scatter(book_counts, run_times)
plt.xlabel('Book Count')
plt.ylabel('Running Time (Seconds)')
plt.show()
```

```python id="w5vesGUMLzmq" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476835699, "user_tz": -330, "elapsed": 77, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5496c7d8-96d6-400c-d26c-6a8a391c2202"
def y(x): return (0.27 / (1000 ** 2)) * (x ** 2)
plt.scatter(book_counts, run_times)
plt.plot(book_counts, y(np.array(book_counts)), c='k')
plt.xlabel('Book Count')
plt.ylabel('Running Time (Seconds)')
plt.show()
```

```python id="MotoTv9RLzmr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476835701, "user_tz": -330, "elapsed": 73, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4bae5a76-eb62-4fb2-d29a-369535998c9a"
book_count = 300000
run_time = y(book_count) / 3600
print(f"It will take {run_time} hours to compute all-by-all similarities "
      f"from a {book_count}-book by {vocabulary_size}-word matrix")
```

<!-- #region id="N-GYqMjvLzmr" -->
## Dimension Reduction of Matrix Data
<!-- #endregion -->

```python id="zGHhy7aLLzmr" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476835702, "user_tz": -330, "elapsed": 62, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6356130e-d4cb-46f3-9269-93bd5939f957"
import numpy as np
heights = np.arange(60, 78, 0.1)
np.random.seed(0)
random_fluctuations = np.random.normal(scale=10, size=heights.size)
weights = 4 * heights - 130 + random_fluctuations
import matplotlib.pyplot as plt
measurements = np.array([heights, weights])
plt.scatter(measurements[0], measurements[1])
plt.xlabel('Height (in)')
plt.ylabel('Weight (lb)')
plt.axis('equal')
plt.show()
```

```python id="NX8q2mogLzms" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476835704, "user_tz": -330, "elapsed": 58, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="347a31df-bd93-408c-85c5-4af7793edba8"
centered_data = np.array([heights - heights.mean(),
                          weights - weights.mean()])
plt.scatter(centered_data[0], centered_data[1])
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.xlabel('Centralized Height (in)')
plt.ylabel('Centralized Weight (lb)')
plt.axis('equal')
plt.show()
```

```python id="2hXr79qiLzms" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476837673, "user_tz": -330, "elapsed": 2025, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b049069e-f92e-4da7-c786-c23ed43728a8"
from math import sin, cos
angle = np.radians(-90)
rotation_matrix = np.array([[cos(angle), -sin(angle)],
                            [sin(angle), cos(angle)]])

rotated_data = rotation_matrix @ centered_data
plt.scatter(centered_data[0], centered_data[1], label='Original Data')
plt.scatter(rotated_data[0], rotated_data[1], c='y', label='Rotated Data')
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.legend()
plt.axis('equal')
plt.show()
```

```python id="R1skVDNVLzmt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476837675, "user_tz": -330, "elapsed": 116, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed6b3254-2a8d-493c-863e-602e9d9f2512"
data_labels = ['unrotated', 'rotated']
data_list = [centered_data, rotated_data]
for data_label, data in zip(data_labels, data_list):
    y_values = data[1]
    penalty = y_values @ y_values / y_values.size
    print(f"The penalty score for the {data_label} data is {penalty:.2f}")
```

```python id="jzQj36pvLzmt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476837677, "user_tz": -330, "elapsed": 101, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="03ab4326-34c2-4887-a80c-06f41b0eb234"
for data_label, data in zip(data_labels, data_list):
    y_var = data[1].var()
    print(y_var)
    print(data[1] @ data[1] / data[1].size)
    #    assert y_var == data[1] @ data[1] / data[1].size
    print(f"The y-axis variance for the {data_label} data is {y_var:.2f}")
```

```python id="QA7oZ4XPLzmu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476837679, "user_tz": -330, "elapsed": 89, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="913b1318-94be-4829-d338-ca01c6bca1b4"
for data_label, data in zip(data_labels, data_list):
    x_var = data[0].var()
    print(f"The x-axis variance for the {data_label} data is {x_var:.2f}")
```

```python id="2d4EDvKLLzmu"
total_variance = centered_data[0].var() + centered_data[1].var()
assert total_variance == rotated_data[0].var() + rotated_data[1].var()
```

```python id="zcWwkx3jLzmu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476837682, "user_tz": -330, "elapsed": 77, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="943e4d33-9a8a-4ba3-e9f0-53b9d94c9ba3"
for data_label, data in zip(data_labels, data_list):
    percent_x_axis_var = 100 * data[0].var() / total_variance
    percent_y_axis_var = 100 * data[1].var() / total_variance
    print(f"In the {data_label} data, {percent_x_axis_var:.2f}% of the "
           "total variance is distributed across the x-axis")
    print(f"The remaining {percent_y_axis_var:.2f}% of the total "
           "variance is distributed across the y-axis\n")
```

```python id="nYYahIR2Lzmu" colab={"base_uri": "https://localhost:8080/", "height": 314} executionInfo={"status": "ok", "timestamp": 1637476837684, "user_tz": -330, "elapsed": 69, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac41ab4b-31c4-4d73-ad87-5d0a2d14ffdc"
def rotate(angle, data=centered_data):
    angle = np.radians(-angle)
    rotation_matrix = np.array([[cos(angle), -sin(angle)],
                                [sin(angle), cos(angle)]])
    return rotation_matrix @ data

angles = np.arange(1, 180, 0.1)
x_variances = [(rotate(angle)[0].var()) for angle in angles]

percent_x_variances = 100 * np.array(x_variances) / total_variance
optimal_index = np.argmax(percent_x_variances)
optimal_angle = angles[optimal_index]
plt.plot(angles, percent_x_variances)
plt.axvline(optimal_angle, c='k')
plt.xlabel('Angle (degrees)')
plt.ylabel('% x-axis coverage')
plt.show()

max_coverage = percent_x_variances[optimal_index]
max_x_var = x_variances[optimal_index]

print("The horizontal variance is maximized to approximately "
      f"{int(max_x_var)} after a {optimal_angle:.1f} degree rotation.")
print(f"That rotation distributes {max_coverage:.2f}% of the total "
       "variance onto the x-axis.")
```

```python id="YkFpXYsOLzmv" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476837686, "user_tz": -330, "elapsed": 68, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e7c72c4-163f-48fd-b13e-373ad16e422e"
best_rotated_data = rotate(optimal_angle)
plt.scatter(best_rotated_data[0], best_rotated_data[1])
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.axis('equal')
plt.show()
```

```python id="N8msRC_eLzmv" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476837688, "user_tz": -330, "elapsed": 68, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0e8fa32-dca3-4fa3-b2d8-123f6f1bdb44"
x_values = best_rotated_data[0]
sorted_x_values = sorted(x_values)
cluster_size = int(x_values.size / 3)
small_cutoff = max(sorted_x_values[:cluster_size])
large_cutoff = min(sorted_x_values[-cluster_size:])
print(f"A 1D threshold of {small_cutoff:.2f} seperates the small-sized "
           "and medium-sized customers.")
print(f"A 1D threshold of {large_cutoff:.2f} seperates the medium-sized "
                "and large-sized customers.")
```

```python id="azg9RNpXLzm0" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476837691, "user_tz": -330, "elapsed": 57, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="994faa8e-95b2-4c35-faad-1d2115e57821"
def plot_customer_segments(horizontal_2d_data):
    small, medium, large = [], [], []
    cluster_labels = ['Small', 'Medium', 'Large']
    for x_value, y_value in horizontal_2d_data.T:
        if x_value <= small_cutoff:
            small.append([x_value, y_value])
        elif small_cutoff < x_value < large_cutoff:
            medium.append([x_value, y_value])
        else:
            large.append([x_value, y_value])

    for i, cluster in enumerate([small, medium, large]):
        cluster_x_values, cluster_y_values = np.array(cluster).T
        plt.scatter(cluster_x_values, cluster_y_values,
                    color=['g', 'b', 'y'][i],
                    label=cluster_labels[i])

    plt.axhline(0, c='black')
    plt.axvline(large_cutoff, c='black', linewidth=3, linestyle='--')
    plt.axvline(small_cutoff, c='black', linewidth=3, linestyle='--')
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_customer_segments(best_rotated_data)
```

```python id="0gkVAt6jLzm1" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476837693, "user_tz": -330, "elapsed": 57, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a7723adc-9434-41fa-ee3f-2e4c0233f52a"
zero_y_values = np.zeros(x_values.size)
reproduced_data = rotate(-optimal_angle, data=[x_values, zero_y_values])
plt.plot(reproduced_data[0], reproduced_data[1], c='k',
         label='Reproduced Data')
plt.scatter(centered_data[0], centered_data[1], c='y',
            label='Original Data')
plt.axis('equal')
plt.legend()
plt.show()
```

```python id="e8cnLatpLzm2" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476838433, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2d625429-39c8-4504-a56f-fa2331d86131"
np.random.seed(1)
new_heights = np.arange(60, 78, .11)
random_fluctuations = np.random.normal(scale=10, size=new_heights.size)
new_weights =  4 * new_heights - 130 + random_fluctuations
new_centered_data = np.array([new_heights - heights.mean(),
                              new_weights - weights.mean()])
plt.scatter(new_centered_data[0], new_centered_data[1], c='y',
            label='New Customer Data')
plt.plot(reproduced_data[0], reproduced_data[1], c='k',
         label='First Principal Direction')
plt.xlabel('Centralized Height (in)')
plt.ylabel('Centralized Weight (lb)')
plt.axis('equal')
plt.legend()
plt.show()
```

```python id="Zsqt_LimLzm3" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476839499, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="424c724f-18e8-455c-db33-6a143db94a93"
new_horizontal_data = rotate(optimal_angle, data=new_centered_data)
plot_customer_segments(new_horizontal_data)
```

<!-- #region id="YUSo44t2Lzm3" -->
## Dimension Reduction Using PCA and Scikit-Learn
<!-- #endregion -->

```python id="UC4IT8AkLzm4" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476847866, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fa13172b-0561-4b05-e078-25ce08122d80"
from sklearn.decomposition import PCA
pca_object = PCA()
pca_transformed_data = pca_object.fit_transform(measurements.T)
plt.scatter(pca_transformed_data[:,0], pca_transformed_data[:,1])
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.axis('equal')
plt.show()
```

```python id="Qx_Vki2OLzm4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476849317, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a629fa6b-2272-4714-bcb9-23fb17f16712"
percent_variance_coverages = 100 * pca_object.explained_variance_ratio_
x_axis_coverage, y_axis_coverage = percent_variance_coverages
print(f"The x-axis of our PCA output covers {x_axis_coverage:.2f}% of "
       "the total variance")
```

```python id="EQQaLZz0Lzm5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476849319, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6ee90d9e-a530-436f-92c5-ad3271b591a1"
first_pc = pca_object.components_[0]
magnitude = norm(first_pc)
print(f"Vector {first_pc} points in a direction that covers "
      f"{x_axis_coverage:.2f}% of the total variance.")
print(f"The vector has a magnitude of {magnitude}")
```

```python id="nHL2HVg8Lzm5" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476849321, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="61da5318-bfad-419e-bc8c-f1a4c6bd8631"
def plot_stretched_vector(v, **kwargs):
    plt.plot([-50 * v[0], 50 * v[0]], [-50 * v[1], 50 * v[1]], **kwargs)

plt.plot(reproduced_data[0], reproduced_data[1], c='k',
         label='First Principal Direction')
plt.scatter(centered_data[0], centered_data[1], c='y')
plt.xlabel('Centralized Height (in)')
plt.ylabel('Centralized Weight (lb)')
plt.axis('equal')
plt.legend()
plt.show()
```

```python id="HbUUXC5ULzm6" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637476850995, "user_tz": -330, "elapsed": 1690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b97e1850-b8a1-4b4f-9943-5ad967bed782"
principal_components = pca_object.components_
for i, pc in enumerate(principal_components):
    plot_stretched_vector(pc, c='k',
                          label='Principal Directions' if i == 0 else None)

for i, axis_vector in enumerate([np.array([0, 1]), np.array([1, 0])]):
    plot_stretched_vector(axis_vector,  c='g', linestyle='-.',
                          label='Axes' if i == 0 else None)

plt.scatter(centered_data[0], centered_data[1], c='y')
plt.xlabel('Centralized Height (in)')
plt.ylabel('Centralized Weight (lb)')
plt.axis('equal')
plt.legend()
plt.show()
```

```python id="T835WuysLzm7"
projections = principal_components @ centered_data
assert np.allclose(pca_transformed_data.T, projections)
```

```python id="R8JfCz_bLzm8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476851000, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ccb68cc-8234-4798-ca13-4ed305d1c20c"
from sklearn.datasets import load_iris
flower_data = load_iris()
flower_measurements = flower_data['data']
num_flowers, num_measurements = flower_measurements.shape
print(f"{num_flowers} flowers have been measured.")
print(f"{num_measurements} measurements were recorded for every flower.")
print("The first flower has the following measurements (in cm): "
      f"{flower_measurements[0]}")
```

```python id="hV8ytepILzm9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476851961, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b92d08ee-1580-48b8-9f96-90be0e741fc5"
pca_object_2D = PCA(n_components=2)
transformed_data_2D = pca_object_2D.fit_transform(flower_measurements)
row_count, column_count = transformed_data_2D.shape
print(f"The matrix contains {row_count} rows, corresponding to "
      f"{row_count} recorded flowers.")
print(f"It also contains {column_count} columns, corresponding to "
      f"{column_count} dimensions.")
```

```python id="dcLkQ_h3Lzm-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476851963, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cad05585-77c8-434b-f310-faf1e460fdc5"
def print_2D_variance_coverage(pca_object):
    percent_var_coverages = 100 * pca_object.explained_variance_ratio_
    x_axis_coverage, y_axis_coverage = percent_var_coverages
    total_coverage = x_axis_coverage + y_axis_coverage
    print(f"The x-axis covers {x_axis_coverage:.2f}% "
            "of the total variance")
    print(f"The y-axis covers {y_axis_coverage:.2f}% "
           "of the total variance")
    print(f"Together, the 2 axes cover {total_coverage:.2f}% "
           "of the total variance")

print_2D_variance_coverage(pca_object_2D)
```

```python id="TR_PHy-mLzm_" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1637476851965, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="34dcf07b-2c91-4728-d560-c8a2c3f1bcc2"
plt.scatter(transformed_data_2D[:,0], transformed_data_2D[:,1])
plt.show()
```

```python id="E_qtqwVJLzm_" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1637476856800, "user_tz": -330, "elapsed": 998, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="390a2539-79f9-4070-e732-ffdc2abc13bb"
def visualize_flower_data(dim_reduced_data):
    species_names = flower_data['target_names']
    for i, species in enumerate(species_names):
        species_data = np.array([dim_reduced_data[j]
                                 for j in range(dim_reduced_data.shape[0])
                                 if flower_data['target'][j] == i]).T
        plt.scatter(species_data[0], species_data[1], label=species.title(),
                     color=['g', 'k', 'y'][i])
    plt.legend()
    plt.show()

visualize_flower_data(transformed_data_2D)
```

```python id="N7rgDOtELzm_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476856801, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb1e6ac3-b955-41c9-d95a-301e15c1a799"
def detect_setosa(flower_sample):
    centered_sample = flower_sample - pca_object_2D.mean_
    projection = pca_object_2D.components_[0] @ centered_sample
    is_setosa = projection < 2
    if projection < 2:
        print("The sample could be a Satosa")
    else:
        print("The sample is not a Satosa")

new_flower_sample = np.array([4.8, 3.7, 1.2, 0.24])
detect_setosa(new_flower_sample)
```

```python id="KZf12JgwLznA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476857481, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed0de2f9-729c-41e8-c127-febaa9d78e14"
first_axis_var = flower_measurements[:,0].var()
print(f"The variance of the first axis is: {first_axis_var:.2f}")

flower_measurements[:,0] *= 10
first_axis_var = flower_measurements[:,0].var()
print("We've converted the measurements from cm to mm.\nThat variance "
      f"now equals {first_axis_var:.2f}")
```

```python id="v16yoL5FLznA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476858649, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ffa8e2de-4ced-4c3b-dcbd-a218baeb1441"
pca_object_2D.fit_transform(flower_measurements)
print_2D_variance_coverage(pca_object_2D)
```

```python id="6FSJlcsGLznA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476860608, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="55e414ff-5b43-46da-c984-d362a5b19d4f"
for i in range(flower_measurements.shape[1]):
    flower_measurements[:,i] /= norm(flower_measurements[:,i])

transformed_data_2D = pca_object_2D.fit_transform(flower_measurements)
print_2D_variance_coverage(pca_object_2D)
```

```python id="MGtISjYnLznB" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476861705, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9f23468d-310f-44d0-c724-15489b474614"
visualize_flower_data(transformed_data_2D)
```

<!-- #region id="9iQQ9F-vLznD" -->
## Computing Principal Componets Without Rotation
<!-- #endregion -->

```python id="CZlCyx8-LznE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476870652, "user_tz": -330, "elapsed": 1551, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a7ef90f-3800-4379-b48c-1d005913ae86"
cov_matrix = centered_data @ centered_data.T / centered_data.shape[1]
print(f"Covariance matrix:\n {cov_matrix}")
for i in range(centered_data.shape[0]):
    variance = cov_matrix[i][i]
    assert round(variance, 10) == round(centered_data[i].var(), 10)
```

```python id="vCp290QzLznH" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476872278, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c942c05d-f374-4837-dc91-50b242e59bab"
def plot_vector(vector, **kwargs):
    plt.plot([0, vector[0]], [0, vector[1]], **kwargs)

plot_vector(first_pc, c='y', label='First Principal Component')
product_vector = cov_matrix @ first_pc
product_vector /= norm(product_vector)
plot_vector(product_vector, c='k', linestyle='--',
            label='Normalized Product Vector')

plt.legend()
plt.axis('equal')
plt.show()
```

```python id="2XyZVi44LznJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476873022, "user_tz": -330, "elapsed": 51, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f98895c7-3b8e-4601-e763-dd804bb49a0a"
product_vector2 = cov_matrix @ product_vector
product_vector2 /= norm(product_vector2)
cosine_similarity = product_vector @ product_vector2
angle = np.degrees(np.arccos(cosine_similarity))
print(f"The angle between vectors equals {angle:.2f} degrees")
```

```python id="bShtPAwKLznL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476873026, "user_tz": -330, "elapsed": 44, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="12abd0eb-9605-4f4f-f753-7a11aee61cb8"
new_magnitude = norm(cov_matrix @ first_pc)
print("Multiplication has streched the first principal component by "
      f"approximately {new_magnitude:.1f} units.")
```

```python id="jBre-w_9LznM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476873029, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4f2c0ab0-f094-4119-f7fe-72e5ceadf0cc"
variance = (centered_data.T @ first_pc).var()
direction1_var = projections[0].var()
print("The variance along the first principal direction is approximately"
       f" {variance:.1f}")
```

```python id="kJOb0QKALznM"
np.random.seed(0)
random_vector = np.random.random(size=2)
random_vector /= norm(random_vector)
```

```python id="cZ9kpMtCLznN" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476874284, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="851f51b9-a718-40ab-88d4-ed91d0bf73d4"
product_vector = cov_matrix @ random_vector
product_vector /= norm(product_vector)

plt.plot([0, random_vector[0]], [0, random_vector[1]],
          label='Random Vector')
plt.plot([0, product_vector[0]], [0, product_vector[1]], linestyle='--',
         c='k', label='Normalized Product Vector')

plt.legend()
plt.axis('equal')
plt.show()
```

```python id="g5uGyA_yLznN" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476875340, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ccded6e-88ee-47ee-eebb-4ced0e134622"
product_vector2 = cov_matrix @ product_vector
product_vector2 /= norm(product_vector2)

plt.plot([0, product_vector[0]], [0, product_vector[1]], linestyle='--',
         c='k', label='Normalized Product Vector')
plt.plot([0, product_vector2[0]], [0, product_vector2[1]], linestyle=':',
         c='r', label='Normalized Product Vector2')
plt.legend()
plt.axis('equal')
plt.show()
```

```python id="ZobJw1mwLznP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476876109, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4f91e14f-4118-4ec4-f6ff-2a379c9d37f4"
np.random.seed(0)
def power_iteration(matrix):
    random_vector = np.random.random(size=matrix.shape[0])
    random_vector = random_vector / norm(random_vector)
    old_rotated_vector = random_vector
    for _ in range(10):
        rotated_vector = matrix @ old_rotated_vector
        rotated_vector = rotated_vector / norm(rotated_vector)
        old_rotated_vector = rotated_vector

    eigenvector = rotated_vector
    eigenvalue = norm(matrix @ eigenvector)
    return eigenvector, eigenvalue

eigenvector, eigenvalue = power_iteration(cov_matrix)
print(f"The extracted eigenvector is {eigenvector}")
print(f"Its eigenvalue is approximately {eigenvalue: .1f}")
```

```python id="eiI5goWULznR"
outer_product = np.outer(eigenvector, eigenvector)
for i in range(eigenvector.size):
    for j in range(eigenvector.size):
        assert outer_product[i][j] == eigenvector[i] * eigenvector[j]
```

```python id="jZeUidO3LznW"
deflated_matrix = cov_matrix - eigenvalue * outer_product
```

```python id="PzlmjizMLznY" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476881496, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="560db53e-b7cf-424e-eb71-b8f9360a9c4c"
np.random.seed(0)
next_eigenvector, _ = power_iteration(deflated_matrix)
components = np.array([eigenvector, next_eigenvector])
projections = components @ centered_data
plt.scatter(projections[0], projections[1])
plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.axis('equal')
plt.show()
```

```python id="TRoOOmnTLzne"
def find_top_principal_components(centered_matrix, k=2):
    cov_matrix = centered_matrix @ centered_matrix.T
    cov_matrix /= centered_matrix[1].size
    return find_top_eigenvectors(cov_matrix, k=k)

def find_top_eigenvectors(matrix, k=2):
    matrix = matrix.copy()
    eigenvectors = []
    for _ in range(k):
        eigenvector, eigenvalue = power_iteration(matrix)
        eigenvectors.append(eigenvector)
        matrix -= eigenvalue * np.outer(eigenvector, eigenvector)

    return np.array(eigenvectors)
```

```python id="19l-YM2ZLznf"
def reduce_dimensions(data, k=2, centralize_data=True):
    data = data.T.copy()
    if centralize_data:
        for i in range(data.shape[0]):
            data[i] -= data[i].mean()

    principal_components = find_top_principal_components(data)
    return (principal_components @ data).T
```

```python id="uNl8LAfiLzng" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476885976, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a5ea2b26-0dd1-40f7-d9f2-5ecfb25b7c6d"
np.random.seed(0)
dim_reduced_data = reduce_dimensions(flower_measurements)
visualize_flower_data(dim_reduced_data)
```

```python id="v3C6fAwmLzni" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476890102, "user_tz": -330, "elapsed": 805, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="561b6f1b-0b7b-4ad1-ea42-1e6365dab216"
np.random.seed(3)
dim_reduced_data = reduce_dimensions(flower_measurements,
                                     centralize_data=False)
visualize_flower_data(dim_reduced_data)
```

```python id="GxJiNV54Lznm" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476890104, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a7bd7ff-ffac-484b-bb28-84c66eece1b5"
variances = [sum(data[:,i].var() for i in range(data.shape[1]))
             for data in [dim_reduced_data, flower_measurements]]
dim_reduced_var, total_var = variances
percent_coverege = 100 * dim_reduced_var / total_var
print(f"Our plot covers {percent_coverege:.2f}% of the total variance")
```

<!-- #region id="gyzyjCFnLzno" -->
## Efficient Dimension Reduction Using SVD and Scikit-Learn
<!-- #endregion -->

```python id="6fsIRYdfLznp" colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"status": "ok", "timestamp": 1637476912849, "user_tz": -330, "elapsed": 607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e0a0dd0b-30ab-4deb-b385-287788df27f4"
from sklearn.decomposition import TruncatedSVD
svd_object = TruncatedSVD(n_components=2)
svd_transformed_data = svd_object.fit_transform(flower_measurements)
visualize_flower_data(svd_transformed_data)
```

```python id="4UmqD_wpLznr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476913328, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="31aea0af-a41d-4eab-c801-7143fa916f74"
percent_variance_coverages = 100 * svd_object.explained_variance_ratio_
x_axis_coverage, y_axis_coverage = percent_variance_coverages
total_2d_coverage = x_axis_coverage + y_axis_coverage
print(f"Our Scikit-Learn SVD output covers {total_2d_coverage:.2f}% of "
       "the total variance")
```

<!-- #region id="ltIx4b51Lznt" -->
## NLP Analysis of Large Text Datasets
<!-- #endregion -->

```python id="6h4zOaeDLznu"
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(remove=('headers', 'footers'))
```

```python id="o3-f320mLznu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476971969, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c03827d-12de-486a-b56e-aed28105835a"
print(newsgroups.target_names)
```

```python id="NdL2yzZkLznw" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476975321, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="52955986-dbec-43aa-8d16-1be1379839fd"
print(newsgroups.data[0])
```

```python id="XDi3i5TALznx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476975836, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a8dfec0c-a717-4412-ee99-340dd9d78f2c"
origin = newsgroups.target_names[newsgroups.target[0]]
print(f"The post at index 0 first appeared in the '{origin}' group")
```

```python id="BQYICWSULzny" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476977644, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="96e23b90-282e-48dc-c42c-1f308b46c34e"
dataset_size = len(newsgroups.data)
print(f"Our dataset contains {dataset_size} newsgroup posts")
```

```python id="fwUOLHKGLzn2"
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
```

```python id="aW2m_FQdLzn4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476983497, "user_tz": -330, "elapsed": 2542, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="79a5cea1-e0ec-4ee5-8a4e-26e84b05bb81"
tf_matrix = vectorizer.fit_transform(newsgroups.data)
print(tf_matrix)
```

```python id="6RKJb6NlLzn6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476983502, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dbde3fe7-d761-48de-c487-77f96b72573d"
print(type(tf_matrix))
```

```python id="XewNSqcOLzn7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476992673, "user_tz": -330, "elapsed": 8296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d0bcd290-592c-4810-92f2-66f096ece22c"
import numpy as np
from sklearn.decomposition import TruncatedSVD

tf_np_matrix = TruncatedSVD(n_components=100).fit_transform(tf_matrix)

#tf_np_matrix = tf_matrix.toarray()
print(tf_np_matrix)
```

```python id="f-UNBuwkLzn8"
#tf_np_matrix = np.memmap('test.memmap', mode='w+', dtype=tf_matrix.dtype, shape=tf_matrix.shape)
#for i,j,v in zip(m.row, m.col, m.data):
#    mm[i,j] = v
```

```python id="CSCLTRGkLzn-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476992676, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a818fba9-da04-46df-daa4-d456a860c67e"
num_posts, vocabulary_size = tf_matrix.shape
print(f"Our collection of {num_posts} newsgroup posts contain a total of "
      f"{vocabulary_size} unique words")
```

```python id="OQ3YnwhoLzn_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476992677, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="52ed4aa4-33ba-4afe-b4ba-3707932f3c39"
import numpy as np
tf_vector = tf_np_matrix[0]
non_zero_indices = np.flatnonzero(tf_vector)
num_unique_words = non_zero_indices.size
print(f"The newsgroup in row 0 contains {num_unique_words} unique words.")
print("The actual word-counts map to the following column indices:\n")
print(non_zero_indices)
```

```python id="n2bYuOcLLzn_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476995629, "user_tz": -330, "elapsed": 2965, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="01dbedea-a0dc-4d53-b54a-31654fd5feec"
vectorizer = CountVectorizer(stop_words='english')
tf_matrix = vectorizer.fit_transform(newsgroups.data)
assert tf_matrix.shape[1] < 114751

words = vectorizer.get_feature_names()
for common_word in ['the', 'this', 'was', 'if', 'it', 'on']:
    assert common_word not in words
```

```python id="keO6xe9wLzn_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476995631, "user_tz": -330, "elapsed": 75, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="da31d25c-dbe6-4381-e803-faed1e21d3b1"
tf_vector = tf_np_matrix[0]
non_zero_indices = np.flatnonzero(tf_vector)
unique_words = [words[index] for index in non_zero_indices]
data = {'Word': unique_words,
        'Count': tf_vector[non_zero_indices]}

df = pd.DataFrame(data).sort_values('Count', ascending=False)
print(f"After stop-word deletion, {df.shape[0]} unique words remain.")
print("The 10 most frequent words are:\n")
print(df[:10].to_string(index=False))
```

```python id="ZyOIIp9wLzoA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476995632, "user_tz": -330, "elapsed": 60, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3029de25-7a7b-45fb-85fb-99fc28ae5bc8"
sub_matrix = tf_matrix[:,non_zero_indices]
print("We obtained a sub-matrix correspond to the 34 words within post 0. "
      "The first row of the sub-matrix is:")
print(sub_matrix[0])
```

```python id="qiyX8JAvLzoB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476995634, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="80235b4e-4a2b-4c69-8651-40727549ba35"
from sklearn.preprocessing import binarize
csr_post_mentions = binarize(tf_matrix[:,non_zero_indices]).sum(axis=0)
#print(f'NumPy matrix-generated counts:\n {np_post_mentions}\n')
print(f'CSR matrix-generated counts:\n {csr_post_mentions}')
```

```python id="oX0uwA_JLzoB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637476995636, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4758846a-2ebd-4aa5-9e45-a36e33c7be52"
matrix_1D = binarize(tf_matrix).sum(axis=0)
my_array = np.asarray(matrix_1D)[0]
print(my_array)
```

```python id="VZ_AwtB_LzoC" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477003739, "user_tz": -330, "elapsed": 7304, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="932b3267-f727-4be3-e432-6caed52e8fdf"
np.random.seed(0)
from sklearn.decomposition import TruncatedSVD

shrunk_matrix = TruncatedSVD(n_components=100).fit_transform(tf_matrix)
print(f"We've dimensionally-reduced a {tf_matrix.shape[1]}-column "
      f"{type(tf_matrix)} matrix.")
print(f"Our output is a {shrunk_matrix.shape[1]}-column "
      f"{type(shrunk_matrix)} matrix.")
```

```python id="Nv582NYzLzoD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477003741, "user_tz": -330, "elapsed": 76, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="325a7014-d1ad-48fe-a42b-ebebf5d5353d"
magnitude = norm(shrunk_matrix[0])
print(f"The magnitude of the first row is {magnitude:.2f}")
```

```python id="W4lTOKpELzoF" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477003743, "user_tz": -330, "elapsed": 61, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="60819999-467a-474c-91b3-4543dd9755aa"
from sklearn.preprocessing import normalize
shrunk_norm_matrix = normalize(shrunk_matrix)
magnitude = norm(shrunk_norm_matrix[0])
print(f"The magnitude of the first row is {magnitude:.2f}")
```

```python id="VNGwcebILzoH"
html_contents = "<html>Hello</html>"
```

```python id="9FhWdiJXLzoI" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1637477003747, "user_tz": -330, "elapsed": 50, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e149651b-44ab-4b5c-b589-9b119e8f99a4"
from IPython.core.display import display, HTML
def render(html_contents): display(HTML(html_contents))
render(html_contents)
```

```python id="rH-iadofLzoI" colab={"base_uri": "https://localhost:8080/", "height": 251} executionInfo={"status": "ok", "timestamp": 1637477003748, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6551cfb6-4d01-4f32-86b4-010a6a614b39"
title = "<title>Data Science is Fun</title>"
head = f"<head>{title}</head>"
header = "<h1>Data Science is Fun</h1>"
paragraphs = ''
for i in range(2):
    paragraph_string = f"Paragraph {i} " * 40
    attribute = f"id='paragraph {i}'"
    paragraphs += f"<p {attribute}>{paragraph_string}</p>"
link_text = "Data Science Bookcamp"
url = "https://www.manning.com/books/data-science-bookcamp"
hyperlink = f"<a href='{url}'>{link_text}</a>"
new_paragraph = f"<p id='paragraph 2'>Here is a link to {hyperlink}</p>"
paragraphs += new_paragraph
body = f"<body>{header}{paragraphs}</body>"
html_contents = f"<html> {title} {body}</html>"
render(html_contents)
```

```python id="gAI2SfCsLzoM" colab={"base_uri": "https://localhost:8080/", "height": 370} executionInfo={"status": "ok", "timestamp": 1637477017729, "user_tz": -330, "elapsed": 1235, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="231cfacb-aef4-48e5-a9bd-d726ddd4d12b"
libraries = ['NumPy', 'Scipy', 'Pandas', 'Scikit-Learn']
items = ''
for library in libraries:
    items += f"<li>{library}</li>"
unstructured_list = f"<ul>{items}</ul>"
header2 = '<h2>Common Data Science Libraries</h2>'
body = f"<body>{header}{paragraphs}{header2}{unstructured_list}</body>"
html_contents = f"<html> {title} {body}</html>"
render(html_contents)
```

```python id="M5u8_u9cLzoN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477021940, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d50a6a2-f53e-45ea-9bd0-148f6bfb9b37"
div1 = f"<div id='paragraphs' class='text'>{paragraphs}</div>"
div2 = f"<div id='list' class='text'>{header2}{unstructured_list}</div>"
div3 = "<div id='empty' class='empty'></div>"
body = f"<body>{header}{div1}{div2}{div3}</body>"
html_contents = f"<html> {title}{body}</html>"
print(html_contents)
```

```python id="RuiX9X33LzoO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477022747, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5e5ab38-b88e-477e-e2f0-4c06405acf4b"
from bs4 import BeautifulSoup as bs
soup = bs(html_contents)
print(soup.prettify())
```

```python id="LvFxYkwzLzoO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477023458, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="07888373-5ad3-4b31-9bd9-ac56c938df9b"
title = soup.find('title')
print(title)
```

```python id="lHmveGxxLzoO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477024072, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fcd7e9c4-a415-4b0e-f518-a0ceb8ab03f1"
print(type(title))
```

```python id="wxP6g-u6LzoP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477026781, "user_tz": -330, "elapsed": 30, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="40b58759-899c-46e2-9107-d7538b27c16c"
print(title.text)
```

```python id="alljsFlhLzoP"
assert soup.title.text == title.text
```

```python id="UXV1xGd6LzoP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477026785, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fdd1c35f-2c17-4947-fd89-c7fdab23e4bd"

body = soup.body
assert body.p.text == soup.p.text
print(soup.p.text)
```

```python id="lpfbRL4ULzoQ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477027535, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b09ab25-3190-4435-ac51-58b94268e4ec"
paragraphs = body.find_all('p')
for i, paragraph in enumerate(paragraphs):
    print(f"\nPARAGRAPH {i}:")
    print(paragraph.text)
```

```python id="G2M450_cLzoQ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477027537, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32739e2c-8184-4798-b9ee-7437a1bbffae"
print([bullet.text for bullet
       in  body.find_all('li')])
```

```python id="tt7AbNqtLzoR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477029918, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="02a20502-d93b-4058-8e77-535396d003cc"
paragraph_2 = soup.find(id='paragraph 2')
print(paragraph_2.text)
```

```python id="KfiI0eLvLzoR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477030703, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4eaf3352-98bb-4e18-9421-03e6e5be2393"
assert paragraph_2.get('id') == 'paragraph 2'
print(paragraph_2.a.get('href'))
```

```python id="xHs0a4dfLzoR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477030705, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="036a4d14-ea29-4c68-d9b5-d494446b694e"
for division in soup.find_all('div', class_='text'):
    id_ = division.get('id')
    print(f"\nDivision with id '{id_}':")
    print(division.text)
```

```python id="dHv51f_DLzoS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477030707, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7ede2b58-3071-4f5c-ea84-d877d078e857"
body.find(id='paragraph 0').decompose()
soup.find(id='paragraph 1').decompose()
print(body.find(id='paragraphs').text)
```

```python id="ZIGkPxseLzoS" colab={"base_uri": "https://localhost:8080/", "height": 242} executionInfo={"status": "ok", "timestamp": 1637477031386, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e938bd83-9e0e-4815-e57d-c0c3a7fbc882"
new_paragraph = soup.new_tag('p')
new_paragraph.string = "This paragraph is new"
soup.find(id='empty').append(new_paragraph)
render(soup.prettify())
```

```python id="JFS3DdJ3Pxw6" outputId="d6653c21-2d85-406f-8840-706888d12408"
#from urllib.request import urlopen
import requests
url = "https://www.manning.com/books/data-science-bookcamp"
#html_contents = urlopen(url).read()
r = requests.get(url)
soup = bs(r.content)
#soup = bs(html_contents)
for division in soup.find_all('div', class_='sect1 available'):
    print(division.text.replace('\n\n\n', '\n'))
```
