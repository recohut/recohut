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

<!-- #region id="yH-WcJpQKox7" -->
# Diversity Aware Book Recommender
> A tutorial on building an amazon-like book recommender and keeping diversity as an important factor

- toc: true
- badges: true
- comments: true
- categories: [diversity, book]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="es2KlLDTpa89" outputId="b6c4bf6a-7bda-4c81-b038-b74de763295d"
# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Displays all rows without truncating
pd.set_option('display.max_rows', None)

# Display all columns with/without truncating (use "set" or "reset")
pd.reset_option('display.max_colwidth')
```

```python colab={"base_uri": "https://localhost:8080/"} id="33he3MF5m0gp" outputId="73c9cb4f-4dec-41c5-aaa1-66a0c17e4118"
# Load book data from csv
books = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads_v2/books.csv", encoding="ISO-8859-1")
books.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="BOeKn-Qdn6Uf" outputId="26214f51-b2e1-491d-ea66-04a66a065195"
books.sample(20)
```

<!-- #region id="vKHkghcDoBhp" -->
There are 10,000 books in this dataset and we want “book tags” as a key feature because it has rich data about the books to help us with recommendations. That data lives in different datasets so we have to data wrangle and piece the data puzzle together.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4aOz1T4qn8TI" outputId="7b3cd90d-05ad-43eb-b3cd-6be9805a710a"
# Load tags book_tags data from csv
book_tags = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads_v2/book_tags.csv", encoding="ISO-8859-1")
tags = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads_v2/tags.csv", encoding="ISO-8859-1")
# Merge book_tags and tags 
tags_join = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
# Merge tags_join and books
books_with_tags = pd.merge(books, tags_join, left_on='book_id', right_on='goodreads_book_id', how='inner')
# Store tags into the same book id row
temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head(5)
# Merge tag_names back into books
books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')
books.sample(20)
```

<!-- #region id="elBa_oqeoWiQ" -->
We now have book tags all in one dataset.
<!-- #endregion -->

<!-- #region id="krkmIMqYoaPg" -->
We have 10,000 books in the dataset each with 100 book tags. What do these book tags contain?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 137} id="XLvRBt2ooQcv" outputId="8c8de793-3405-4627-e8a0-fa170a4e99c6"
# Explore book tags
books['tag_name'][0]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 137} id="fWWCbX8Uob5Y" outputId="83d628af-7d3c-4d29-ecaf-686147e4bf09"
books['tag_name'][1]
```

<!-- #region id="vDx1xFfKopM_" -->
We want to transform these texts into numerical values so we have data that the machine learning algorithm understands. TfidfVectorizer turns text into feature vectors.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b7DFk04mofs-" outputId="e5af344f-76c1-4167-a0ff-a10c6b4482fd"
# Transform text to feature vectors
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=50, stop_words='english')
tfidf_matrix = tf.fit_transform(books['tag_name'])
tfidf_matrix.todense()
```

<!-- #region id="GlEQ3_poqYK9" -->
TF-IDF (Term Frequency — Inverse Document Frequency) calculates how important words are in relation to the whole document. TF summarizes how often a given word appears within a document. IDF downscales words that appear frequently across documents. This allows TF-IDF to define the importance of words within a document based on the relationship and weighting factor.
<!-- #endregion -->

<!-- #region id="bAen3ZJEqa3Z" -->
Now we build the recommender. We can use cosine similarity to calculate the numeric values that denote similarities between books.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rR_ASU9Nq3Ch" outputId="ec3e9e56-3f83-49ee-b7c7-baecfbf600dc"
tfidf_matrix
```

```python colab={"base_uri": "https://localhost:8080/"} id="cQ1vcuGhqWed" outputId="46303153-91f8-44ae-f124-33830740a367"
# Use numeric values to find similarities
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim
```

<!-- #region id="pa4ipcM7qe8C" -->
Cosine similarity measures the cosine of the angle between two vectors projected in a multi-dimensional space. The smaller the angle, the higher the cosine similarity. In other words, the closer these book tags are to each other, the more similar the book.

<!-- #endregion -->

<!-- #region id="l0ByNaKBqgdX" -->
Next we write the machine learning algorithm.

<!-- #endregion -->

```python id="zeMLYtssqem5"
# Get book recommendations based on the cosine similarity score of book tags
# Build a 1-dimensional array with book titles
titles = books['title']
tag_name = books['tag_name']
indices = pd.Series(books.index, index=books['title'])
# Function that gets similarity scores
def tags_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # sim_scores = sim_scores[1:11] # How many results to display
    book_indices = [i[0] for i in sim_scores]
    title_df = pd.DataFrame({'title': titles.iloc[book_indices].tolist(),
                           'similarity': [i[1] for i in sim_scores],
                            'tag_name': tag_name.iloc[book_indices].tolist()}, 
                           index=book_indices)
    return title_df
```

<!-- #region id="ovainAgCr8m3" -->
This is the foundational code we need for a recommendation engine. This is the building block for Amazon’s $98 billion revenue-generating algorithm and others like it. Almost seems too simple. We can stop here or we can expand our code to show more data insights.
<!-- #endregion -->

```python id="ep8Z1ZA2qc-t"
# Function that gets book tags and stats
def recommend_stats(target_book_title):
    
    # Get recommended books
    rec_df = tags_recommendations(target_book_title)
    
    # Get tags of the target book
    rec_book_tags = books_with_tags[books_with_tags['title'] == target_book_title]['tag_name'].to_list()
    
    # Create dictionary of tag lists by book title
    book_tag_dict = {}
    for title in rec_df['title'].tolist():
        book_tag_dict[title] = books_with_tags[books_with_tags['title'] == title]['tag_name'].to_list()
    
    # Create dictionary of tag statistics by book title
    tags_stats = {}
    for book, tags in book_tag_dict.items():
        tags_stats[book] = {}
        tags_stats[book]['total_tags'] = len(tags)
        same_tags = set(rec_book_tags).intersection(set(tags)) # Get tags in recommended book that are also in target book
        tags_stats[book]['%_common_tags'] = (len(same_tags) / len(tags)) * 100
    
    # Convert dictionary to dataframe
    tags_stats_df = pd.DataFrame.from_dict(tags_stats, orient='index').reset_index().rename(columns={'index': 'title'})
    
    # Merge tag statistics dataframe to recommended books dataframe
    all_stats_df = pd.merge(rec_df, tags_stats_df, on='title')
    return all_stats_df
```

<!-- #region id="fYJib_JpsCd_" -->
Now we input Lord of the Rings into the recommendation engine and see the results.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GmZiE-_IsBGY" outputId="08f03d81-cffa-4c8d-b18b-4bfd9697f72b"
# Find book recommendations
lor_recs = recommend_stats('The Fellowship of the Ring (The Lord of the Rings, #1)')
lor_recs
```

<!-- #region id="ObQlZRCFsJ6p" -->
We get a list of the top 10 most similar books to Lord of the Rings based on book tags. 
<!-- #endregion -->

<!-- #region id="OhnbE9PasO1c" -->
Since we are reverse engineering through the Elon Musk customer lens and wanting the recommender to output Zero to One, let’s find where this book is positioned in relation to Lord of the Rings.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="wbxVBTbKsDw6" outputId="52852372-997b-4b94-9789-ddde29edde46"
# Find Zero to One book
lor_recs[lor_recs.title == 'Zero to One: Notes on Startups, or How to Build the Future']
```

<!-- #region id="dg2KDDPfsTaX" -->
In relation to Lord of the Rings, Zero to One is rank 8,871 (because index starts from zero, index 0 means rank 1, e.g.) out of 10,000 books based on similarities. Pretty low. According to the algorithm, these two books are on opposite ends of the spectrum and not similar at all. This book is statistically in the lowest quartile which means neither you nor Elon would be recommended this diversity of thought.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="g5gDZAM2sQT9" outputId="a58fee79-ada9-4203-fafb-aab3fea4519f"
# Calculate statistical data
lor_recs.describe()
```

<!-- #region id="Jrm2rwwatbBK" -->
Using a boxplot, we can better visualize this positioning:

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 514} id="g0mISXNNtW5G" outputId="a8644a16-23bc-4048-faaa-0195e5c6ac5c"
# Boxplot of similarity score
import matplotlib.pyplot as plt
lor_recs.boxplot(column=['similarity'])
plt.show()
# Boxplot of percentage of common tags
lor_recs.boxplot(column=['%_common_tags'])
plt.show()
```

<!-- #region id="pkzx5DTitgqc" -->
We can explore the data further and find the most common book tags using NLTK (Natural Language Toolkit). First, we clean up words such as removing hyphens, tokenize the words, and then remove all the stop words. After the text is clean, we can calculate the top 10 frequent words that appear in the Lord of the Rings book tags.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 520} id="2YszWxfythB_" outputId="7586f874-0a12-47ca-ec28-709a71638664"
# Store book tags into new dataframe
lor_tags = pd.DataFrame(books_with_tags[books_with_tags['title']=='The Fellowship of the Ring (The Lord of the Rings, #1)']['tag_name'])
# Find most frequent word used in book tags
top_N = 10
txt = lor_tags.tag_name.str.lower().str.replace(r'-', ' ').str.cat(sep=' ') # Remove hyphens
words = nltk.tokenize.word_tokenize(txt)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
print('All frequencies, including STOPWORDS:')
print('=' * 60)
lor_rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])
print(lor_rslt)
print('=' * 60)
lor_rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')
matplotlib.style.use('ggplot')
lor_rslt.plot.bar(rot=0)
plt.show()
```

<!-- #region id="AA3WOU76toUC" -->
Since we want diversity and variety, we can take the most frequent words “fantasy” and “fiction” and filter by unlike or different words in the context of book genres. These might be words like non-fiction, economics, or entrepreneurial.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="wCJgnsN9topY" outputId="6f23920e-e1c7-4331-df2e-ca5ac1d69579"
# Filter by unlike words
lor_recs_filter = lor_recs[(lor_recs['tag_name'].str.contains('non-fiction')) & (lor_recs['tag_name'].str.contains('economics')) & (lor_recs['tag_name'].str.contains('entrepreneurial'))]
lor_recs_filter
```

<!-- #region id="-gvrdA9mtsep" -->
This narrows down the list and only include books that contain “non-fiction”, “economics”, or “entrepreneurial” in the book tags. To ensure our reader is recommended a good book, we merge ‘average_rating’ back into the dataset and sort the results by the highest average book rating.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="n93olb_htrtS" outputId="835ac73b-c649-4025-c99e-bb3be42aa0b7"
# Merge recommendations with ratings
lor_recs_filter_merge = pd.merge(books[['title', 'average_rating']], lor_recs_filter, left_on='title', right_on='title', how='inner')
# Sort by highest average rating
lor_recs_filter_merge = lor_recs_filter_merge.sort_values(by=['average_rating'], ascending=False)
lor_recs_filter_merge
```

<!-- #region id="-ygO0hRwtv43" -->
What appears at the top of the list — Zero to One. We engineered our way into recommending diversity. Here’s an alternative solution. Instead of recommendations based on book tags of the individual book, we can make recommendations based on categories. The rule: each recommendation can still have similarities to the original book, but it must be a unique category.
<!-- #endregion -->

<!-- #region id="-jBLsTTFt54i" -->
Instead of 5 books with similar book tags to Lord of the Rings, the result should be 5 books with similar book tags, but with a different category to Fantasy, such as Science Fiction. Subsequent results would follow the same logic. This would “diversify” the categories of books the user sees, while maintaining a chain of relevancy. It would display: Fantasy → Science Fiction → Technology → Entrepreneurship → Biographies.
<!-- #endregion -->

<!-- #region id="HyKt6H9Ot9v_" -->
This categorization becomes a sort within a sort to ensure a quality, relevant, and diverse read. This concept could more systematically connect books like Lord of the Rings and Zero to One together, eventually scaling to different product types or industries, such as music.
<!-- #endregion -->
