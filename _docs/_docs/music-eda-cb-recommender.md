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

<!-- #region id="UhxgR9LPVKTW" -->
# Music EDA and Similarity-based Recommender
> In this notebook we will go through an in depth analysis of sound and how we can visualize, classify, understand it, and ultimately recommend similar sounding music.

- toc: true
- badges: true
- comments: true
- categories: [Music, Visualization]
- author: "<a href='https://github.com/manasmurmu/Audio-visualize-and-Music-Recommendation-System'>manasmurmu</a>"
- image:
<!-- #endregion -->

<!-- #region id="IGs2NWoWO3Go" -->
## Introduction
Why are we doing this?

Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

In this notebook we will go through an in depth analysis of sound and how we can visualize, classify and ultimately understand it.
<!-- #endregion -->

<!-- #region id="XgQVSsJGO3Gu" -->
## Purpose
1) We want to understand what is an Audio file. What features we can visualize on this kind of data.

2) EDA. ALways good, here very necessary.

3) A recommender system: given a song, give me top X songs most similar.
<!-- #endregion -->

```python id="UFA6M4uPO3Gy"
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import os

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
import IPython.display as ipd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="WFp_2FYtO_Dp" -->
## Download the data
Audio Source - http://marsyas.info/downloads/datasets.html [alt1](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

CSV Source - [source](https://git.cse.iitb.ac.in/maheshabnave/cs725/blob/e23e32392fd62ab825e7386d8ce3b7a15b503eba/data/features_30_sec.csv) [alt1](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) [alt2](https://raw.githubusercontent.com/kcirerick/deep-music/master/features_30_sec.csv)
<!-- #endregion -->

```python id="xOtL8fFiO0dQ"
!wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
!tar -xvf genres.tar.gz
```

```python id="3MbgF6yXTt8h"
!wget https://raw.githubusercontent.com/kcirerick/deep-music/master/features_30_sec.csv
```

```python id="rJihS7ZjO3G1" colab={"base_uri": "https://localhost:8080/"} outputId="55a5f242-f5b4-4394-ff0f-97cea105ce04"
df = '/content'
print(list(os.listdir(f'{df}/genres/')))
```

<!-- #region id="ahPcqKH2O3G3" -->
## Explore Audio Data
Sound: Seqeuence of vibrations in pressure strengths (y)

The sample rate (sr) number of samples of audio carried per second, measured hz or khz
<!-- #endregion -->

```python id="cbsCu-HaO3G3" colab={"base_uri": "https://localhost:8080/"} outputId="01ce4b7b-d4e5-4c2a-a6b1-389426357a49"
#Importing one audio file 
#
y, sr = librosa.load(f'{df}/genres/pop/pop.00034.wav')

print('y:', y, '\n')
```

```python id="0fK7VkeRO3G4" colab={"base_uri": "https://localhost:8080/"} outputId="d5ecd877-6fb9-4a5f-aad0-ff2119ecabdc"
print('y shape:', np.shape(y), '\n')
print('Sample Rate (KHz):', sr, '\n')
```

```python id="40sStqJHO3G4" colab={"base_uri": "https://localhost:8080/"} outputId="f65cf74a-7e5e-4eb6-bd30-cf266a81f54d"
#length of the audio
print('Check Len of the Audio:', 661794/22050)
```

```python id="TDhn4_xlO3G6" colab={"base_uri": "https://localhost:8080/"} outputId="e5330967-3f97-4970-b8eb-598f49a40890"
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
audio_file, _ = librosa.effects.trim(y)

# the result is an numpy ndarray
print('Audio File:', audio_file, '\n')
print('Audio File shape:', np.shape(audio_file))
```

<!-- #region id="yB3157O7O3G7" -->
### 2D representation: sound wave
<!-- #endregion -->

```python id="c8zZgklxO3G8" colab={"base_uri": "https://localhost:8080/", "height": 413} outputId="3ad8aa70-0fe1-48be-97a1-9891ac758a07"
plt.figure(figsize = (16, 6))
librosa.display.waveplot(y = audio_file, sr = sr, color = "#A300F9");
plt.title("Sound Waves in Rock 41", fontsize = 25);
```

<!-- #region id="n-9yYwlYO3G9" -->
### Fourier Transform
1) Function that gets a signal in the time domain as input, and outputs its decomposition into frequencies

2) Transform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is approx. the log scale of amplitudes.

<!-- #endregion -->

```python id="7NeGg3C7O3G-" colab={"base_uri": "https://localhost:8080/"} outputId="d0e758c7-6c18-4b0e-9990-bc9dd30636e1"
# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

# Short-time Fourier transform (STFT)
D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))

print('Shape of D object:', np.shape(D))
```

```python id="2G8nOzPBO3G_" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="23a079ac-bec0-4fdc-f6a3-06324c121dda"
plt.figure(figsize = (16, 6))
plt.plot(D);

#Transform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is approx. the log scale of amplitudes.
```

<!-- #region id="gLU51D5KO3HB" -->
### Spectogram

- What is a spectrogram? 

-> A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams (wiki).

- Here we convert the frequency axis to a logarithmic one.
<!-- #endregion -->

```python id="OUQKYT0YO3HC" colab={"base_uri": "https://localhost:8080/", "height": 413} outputId="77362564-909c-40eb-cd65-cf0bed3e2b36"
#Here we convert the frequency axis to a logarithmic one.

# Convert an amplitude spectrogram to Decibels-scaled spectrogram.
DB = librosa.amplitude_to_db(D, ref = np.max)

# Creating the Spectogram
plt.figure(figsize = (16, 6))
librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool')
plt.colorbar();
plt.title("pop 34", fontsize = 25);
```

<!-- #region id="N-eXFDKRO3HD" -->
#### Mel Spectrogram
- The Mel Scale, mathematically speaking, is the result of some non-linear transformation of the frequency scale. The Mel Spectrogram is a normal Spectrogram, but with a Mel Scale on the y axis.
<!-- #endregion -->

```python id="-lS3QbICO3HE" colab={"base_uri": "https://localhost:8080/", "height": 412} outputId="6ffd6628-15d9-4c14-f7c7-db0ca2bf05b5"
y, sr = librosa.load(f'{df}/genres/metal/metal.00036.wav')
y, _ = librosa.effects.trim(y)


S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize = (16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',
                        cmap = 'cool');
plt.colorbar();
plt.title("Metal Mel Spectrogram", fontsize = 23);
```

```python id="4n35RVLxO3HF" colab={"base_uri": "https://localhost:8080/", "height": 412} outputId="974936d2-9f00-4ecd-bfcf-9a81fcc1970f"
y, sr = librosa.load(f'{df}/genres/classical/classical.00036.wav')
y, _ = librosa.effects.trim(y)


S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize = (16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',
                        cmap = 'cool');
plt.colorbar();
plt.title("Classical Mel Spectrogram", fontsize = 23);
```

<!-- #region id="okwGN7Z8O3HG" -->
### Audio features 

#### Zero Crossing Rate 
- the rate at which the signal changes from positive to negative or back.
<!-- #endregion -->

```python id="5jBZIXXRO3HH" colab={"base_uri": "https://localhost:8080/"} outputId="86430d82-f0b0-4244-953b-542a9c37c6d8"
# Total zero_crossings in our 1 song
zero_crossings = librosa.zero_crossings(audio_file, pad=False)
print(sum(zero_crossings))
```

<!-- #region id="T9o3Xh4WO3HW" -->
### Harmonics and Perceptrual
- Harmonics are characteristichs that human years can't distinguish (represents the sound color)
- Perceptrual understanding shock wave represents the sound rhythm and emotion
<!-- #endregion -->

```python id="OD8iZiq_O3HZ" colab={"base_uri": "https://localhost:8080/", "height": 374} outputId="f5a5ed83-d997-4742-d174-1db8d899af29"
y_harm, y_perc = librosa.effects.hpss(audio_file)

plt.figure(figsize = (16, 6))
plt.plot(y_harm, color = '#A300F9');
plt.plot(y_perc, color = '#FFB100');
```

<!-- #region id="jhbu0jRaO3Hb" -->
### Tempo BMP (beats per minute)

#### Dynamic programming beat tracker.
<!-- #endregion -->

```python id="gX1wa4w8O3Hc" colab={"base_uri": "https://localhost:8080/"} outputId="a9d1df47-5726-4eb0-c958-e1956fd81d25"
tempo, _ = librosa.beat.beat_track(y, sr = sr)
tempo
```

<!-- #region id="Xdvy9Ni8O3Hd" -->
### Spectral Centroid
- indicates where the "centre of mass" for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.

<!-- #endregion -->

```python id="EH7yV_FrO3He" colab={"base_uri": "https://localhost:8080/"} outputId="31caf658-57b6-431f-f73b-867f2a25c048"
# Calculate the Spectral Centroids
spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]

# Shape is a vector
print('Centroids:', spectral_centroids, '\n')
print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

print('frames:', frames, '\n')
print('t:', t)

# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
```

```python id="tBFYHPifO3Hg" colab={"base_uri": "https://localhost:8080/", "height": 388} outputId="e1192b89-835e-463f-ba61-6c55deb9ec24"
#Plotting the Spectral Centroid along the waveform
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_file, sr=sr, alpha=0.4, color = '#A300F9');
plt.plot(t, normalize(spectral_centroids), color='#FFB100');
```

<!-- #region id="_JEng6FrO3Hh" -->
### Spectral Rolloff
- is a measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies
<!-- #endregion -->

```python id="RWhztV2XO3Hi" colab={"base_uri": "https://localhost:8080/", "height": 388} outputId="2e2b06db-40b6-458e-e234-e5a5b23c191c"
# Spectral RollOff Vector
spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)[0]

# The plot
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_file, sr=sr, alpha=0.4, color = '#A300F9');
plt.plot(t, normalize(spectral_rolloff), color='#FFB100');
```

<!-- #region id="DU6bDl5sO3Hj" -->
### Mel-Frequency Cepstral Coefficients
- The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice
<!-- #endregion -->

```python id="dcncckd4O3Hk" colab={"base_uri": "https://localhost:8080/", "height": 405} outputId="305d7e6f-c2dd-48ff-e3f0-0e68c0b89510"
mfccs = librosa.feature.mfcc(audio_file, sr=sr)
print('mfccs shape:', mfccs.shape)

#Displaying  the MFCCs:
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
```

<!-- #region id="i_c3x8nGO3Hk" -->
Data needs to be scaled:
<!-- #endregion -->

```python id="Zyv9WICbO3Hk" colab={"base_uri": "https://localhost:8080/", "height": 439} outputId="a3d9de85-b4d7-40b8-e50e-d89b11d824f1"
# Perform Feature Scaling
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print('Mean:', mfccs.mean(), '\n')
print('Var:', mfccs.var())

plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
```

<!-- #region id="BVmgbEL8O3Hl" -->
### Chroma Frequencies 
- Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.
<!-- #endregion -->

```python id="E2ngnm0iO3Hl" colab={"base_uri": "https://localhost:8080/", "height": 405} outputId="38748740-bac1-48b3-bfa2-01852f77468e"
# Increase or decrease hop_length to change how granular you want your data to be
hop_length = 5000

# Chromogram
chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
print('Chromogram shape:', chromagram.shape)

plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm');
```

<!-- #region id="E_oACysaO3Hl" -->
## EDA ( Exploratory Data Analysis )

 EDA is going to be performed on the features_30_sec.csv. This file contains the mean and variance for each audio file fo the features analysed above.

So, the table has a final of 1000 rows (10 genrex x 100 audio files) and 60 features (dimensionalities).
<!-- #endregion -->

```python id="wt65OKH-O3Hm" colab={"base_uri": "https://localhost:8080/", "height": 224} outputId="9049e3d6-d4dd-4e93-9873-02146f008bdc"
data = pd.read_csv('features_30_sec.csv')
data.head()
```

<!-- #region id="BqUZJDv1O3Hm" -->
### Correlation Heatmap for feature means
<!-- #endregion -->

```python id="vnZqQh7lO3Hm" colab={"base_uri": "https://localhost:8080/", "height": 792} outputId="6f91f6db-6009-4949-fde7-40e8139f41a0"
# Computing the Correlation Matrix
spike_cols = [col for col in data.columns if 'mean' in col]
corr = data[spike_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 11));

# Generate a custom diverging colormap
cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 25)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);
```

<!-- #region id="2ir5axxeO3Hn" -->
### Box Plot for Genres Distributions
<!-- #endregion -->

```python id="wItsuVcBO3Hn" colab={"base_uri": "https://localhost:8080/", "height": 597} outputId="66bc0608-57da-4712-ff4e-13ddadb80438"
x = data[["label", "tempo"]]

f, ax = plt.subplots(figsize=(16, 9));
sns.boxplot(x = "label", y = "tempo", data = x, palette = 'husl');

plt.title('BPM Boxplot for Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Genre", fontsize = 15)
plt.ylabel("BPM", fontsize = 15)
```

<!-- #region id="Uka6WyJsO3Hn" -->
### Principal Component Analysis - to visualize possible groups of genres
1) Normalization

2) PCA

3) The Scatter Plot
<!-- #endregion -->

```python id="V8XHg9vtO3Ho" colab={"base_uri": "https://localhost:8080/"} outputId="008c1dce-cd79-4ad8-b198-bc811e7d6053"
data = data.iloc[0:, 1:]
y = data['label']
X = data.loc[:, data.columns != 'label']

#### NORMALIZE X ####
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)


#### PCA 2 COMPONENTS ####
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis = 1)

pca.explained_variance_ratio_

# 44.93 variance explained
```

```python id="DAmYU0C6O3Ho" colab={"base_uri": "https://localhost:8080/", "height": 577} outputId="e571bc04-7bb5-4f47-8a95-03d9b32b1a8b"
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert.jpg")
```

<!-- #region id="e4q9C7iqO3Ho" -->
## Recomender System

"Recomender" Systems enable us for any given vector to find the best similarity, ranked in descending order, from the bast match to the least best match.

For Audio files, this will be done through cosine_similarity library.
<!-- #endregion -->

```python id="MwVOdH-YO3Hp" colab={"base_uri": "https://localhost:8080/"} outputId="1e6c8f14-2441-4a11-aeeb-4088106efa1b"
# Read data
data = pd.read_csv('features_30_sec.csv', index_col='filename')

# Extract labels
labels = data[['label']]

# Drop labels from original dataframe
data = data.drop(columns=['length','label'])
data.head()

# Scale the data
data_scaled=preprocessing.scale(data)
print('Scaled data type:', type(data_scaled))
```

<!-- #region id="5PoSaLEQO3Hp" -->
### Cosine similarity
Calculates the pairwise cosine similarity for each combination of songs in the data. This results in a 1000 x 1000 matrix (with redundancy in the information as item A similarity to item B == item B similarity to item A).
<!-- #endregion -->

```python id="FhxqMgXZO3Hp" colab={"base_uri": "https://localhost:8080/", "height": 301} outputId="8b35f010-512f-4cbf-81d3-dec733738b76"
# Cosine similarity
similarity = cosine_similarity(data_scaled)
print("Similarity shape:", similarity.shape)

# Convert into a dataframe and then set the row index and column names as labels
sim_df_labels = pd.DataFrame(similarity)
sim_df_names = sim_df_labels.set_index(labels.index)
sim_df_names.columns = labels.index

sim_df_names.head()
```

<!-- #region id="otDWfzIYO3Hq" -->
### Song similarity scoring
find_similar_songs() - is a predefined function that takes the name of the song and returns top 5 best matches for that song.
<!-- #endregion -->

```python id="sw3Uhxu5O3Hq"
def find_similar_songs(name):
    # Find songs most similar to another song
    series = sim_df_names[name].sort_values(ascending = False)
    
    # Remove cosine similarity == 1 (songs will always have the best match with themselves)
    series = series.drop(name)
    
    # Display the 5 top matches 
    print("\n*******\nSimilar songs to ", name)
    print(series.head(5))
```

<!-- #region id="VyAJb3HZO3Hq" -->
### Putting the Similarity Function into Action
<!-- #endregion -->

<!-- #region id="ZZ0L4c5wWe2a" -->
#### Rock Example
<!-- #endregion -->

```python id="_lh28ZidO3Hq" colab={"base_uri": "https://localhost:8080/", "height": 245} outputId="b73686fc-cc52-4a44-d29c-acbbbfe62832"
# pop.00019 - Britney Spears "Hit me baby one more time"
find_similar_songs('rock.00067.wav') 

ipd.Audio(f'{df}/genres/rock/rock.00067.wav')
```

<!-- #region id="Mxyjx6ZQO3Hr" -->
#### Similar song match no.1
<!-- #endregion -->

```python id="49-8UjRyO3Hr" colab={"base_uri": "https://localhost:8080/", "height": 75} outputId="f1f09501-4ab8-488f-c10a-c0d6e2550563"
ipd.Audio(f'{df}/genres/rock/rock.00068.wav')
```

<!-- #region id="7mCduKNdO3Hu" -->
#### Similar song match no.2
<!-- #endregion -->

```python id="Yxd2FvUmO3Hu" colab={"base_uri": "https://localhost:8080/", "height": 75} outputId="2457e9c7-7998-498c-ede5-66ca98009e0f"
ipd.Audio(f'{df}/genres/rock/rock.00065.wav')
```

<!-- #region id="sIYMn5GkO3Hw" -->
#### Similar song match no.3
<!-- #endregion -->

```python id="gjoDOapMO3Hx" colab={"base_uri": "https://localhost:8080/", "height": 75} outputId="8f535dd9-daa1-4bd8-9701-b3f433b7238b"
ipd.Audio(f'{df}/genres/metal/metal.00065.wav')
```

<!-- #region id="-6dpgORtO3Hz" -->
#### Similar song match no.4
<!-- #endregion -->

```python id="OWo1VtQgO3Hz" colab={"base_uri": "https://localhost:8080/", "height": 75} outputId="350afe06-1af1-481d-bb3c-3f2d7e02c69e"
ipd.Audio(f'{df}/genres/metal/metal.00044.wav')
```

<!-- #region id="s0OYhlanO3IB" -->
#### Similar song match no.5
<!-- #endregion -->

```python id="VJPDkIT9O3IC" colab={"base_uri": "https://localhost:8080/", "height": 75} outputId="219e2850-e641-4d49-b1f9-b9735b096d39"
ipd.Audio(f'{df}/genres/metal/metal.00041.wav')
```
