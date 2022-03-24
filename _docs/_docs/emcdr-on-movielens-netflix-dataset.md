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

<!-- #region id="DZN09pc7HDHo" -->
# EMCDR on MovieLens-Netflix dataset
<!-- #endregion -->

<!-- #region id="RuCrMB92HXLw" -->
## Theory
<!-- #endregion -->

<!-- #region id="rBmmp4EtHP_q" -->
**Illustrative diagram of the EMCDR framework:**
<!-- #endregion -->

<!-- #region id="nchNbAcex5yo" -->
<p><center><img src='_images/T459379_1.png'></center></p>
<!-- #endregion -->

<!-- #region id="dtqwsk_jHYpL" -->
### Algorithm
<!-- #endregion -->

<!-- #region id="00qhParQx_y0" -->
<p><center><img src='_images/T459379_2.png'></center></p>
<!-- #endregion -->

<!-- #region id="xXp3pC_CHM9d" -->
## CLI Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dAXFXrropyb1" executionInfo={"status": "ok", "timestamp": 1635689079196, "user_tz": -330, "elapsed": 1361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="695750cc-aec9-4ace-9557-415ae9c6a0b8"
!git clone https://github.com/MaJining92/EMCDR.git
%cd EMCDR
```

```python colab={"base_uri": "https://localhost:8080/"} id="oAXftEADp5ju" executionInfo={"status": "ok", "timestamp": 1635689079199, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c4313550-5dbb-47b9-b180-7e24d5878b68"
%tensorflow_version 1.x
```

```python id="LJGKjr-iqADr"
!cp Latent\ Factor\ Modeling/*.py .
!cp Latent\ Space\ Mapping/*.py .
```

```python colab={"base_uri": "https://localhost:8080/"} id="xheHEwR6qOwX" executionInfo={"status": "ok", "timestamp": 1635689274906, "user_tz": -330, "elapsed": 195294, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dfe1941a-8852-4af0-9ed9-7c2b2addd22f"
!python BPR.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="iqXhXEUdqgDG" executionInfo={"status": "ok", "timestamp": 1635689283174, "user_tz": -330, "elapsed": 6260, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2c70ff4-031c-4d51-8848-833fed0b39c5"
!python LM.py
```

<!-- #region id="50ufr23_HldR" -->
## Citations
<!-- #endregion -->

<!-- #region id="8Dw7NE7UHmMb" -->
Cross-domain recommendation: an embedding and mapping approach. Man et al.. 2017. IJCAI. [https://www.ijcai.org/proceedings/2017/0343.pdf](https://www.ijcai.org/proceedings/2017/0343.pdf)
<!-- #endregion -->
