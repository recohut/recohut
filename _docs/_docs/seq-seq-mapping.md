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

<!-- #region id="PDXO8Wxc-Q-U" -->
### Summary

- There are 4 datasets and each contains two parts except dataset 3, so there is a total of 7 files. There are 2 additional files, which are modified versions of dataset 2. So, we are dealing with 9 files in total. 

- We explored different kinds of models that could potentially find the required function to map from input to output. We found the seq2seq model class to lead the criteria. So we selected Seq2Seq as our model class which is specially designed to map the input sequence to output sequence.

- We created 3 variants of seq2seq model. Additionally, we also tried two variants of datasets: one without removing colon sign in the output column and second with removing the colon sign.

- We split the dataset into Test (5%), Validation (15%), and Train (80%) partitions. The validation set was exposed to the model for cross-validation only and the test was not exposed at any stage during training and only exposed at the last stage during testing of the model's performance. 

- For implementation, the first variant was the Seq2Seq LSTM which was implemented in Keras. The second variant was Seq2Seq LSTM with Attention which was also implemented in Keras with the support of an additional library for the attention layer in the model. The third variant additionally included the GRU/LSTM choice in the training and Beam search facility during decoding and this variant was implemented in OpenNMT.
<!-- #endregion -->

<!-- #region id="wN6NdYUh-H0u" -->
### Techniques

1. Vanilla Seq2Seq
2. +Bidirectional
3. +Attention
4. +Beam Search
5. +Teacher Forcing
6. LSTM \ GRU \ RNN \ CNN \ Tranformer
<!-- #endregion -->

<!-- #region id="jTgT9gmW-LVp" -->
### Evaluation metrics

1. BLEU
2. Character-level accuracy
3. Top-K hit rate (HR@K)
4. Mean Reciprocal Rank (MRR)
<!-- #endregion -->

<!-- #region id="Wy3d5_T997Jp" -->
### Experiment scope

1. RNN ↔  LSTM ↔  GRU ↔  CNN ↔  Transformer
2. No BiRNN ↔  BiRNN
3. Dictionary mapping ↔  No mapping
4. Decoding with colons ↔  No colons
5. Optimizer Adam ↔  RMSprop
6. Attention
7. Greedy ↔  Beam Search
<!-- #endregion -->

<!-- #region id="-Bvj3gqv_oV9" -->
### Libraries

- OpenNMT from HarvardNLP
- FairSeq from FAIR
- Seq2Seq from Google
- FastAI
<!-- #endregion -->

<!-- #region id="0I6vgCVw99tR" -->
### Versions

| version | method | +/- |
| :------ | :----- | :------- |
| 1.0 | keras seq2seq | Keras Character level encoder decoder model |
| 1.1 | keras seq2seq | + attention |
| 1.2 | keras seq2seq | + beam search |
| 1.3 | keras seq2seq | + BLEU eval metric |
| 1.4 | keras seq2seq | + multilayer bidirectional LSTM/GRU |
| 1.5 | keras seq2seq | + attention |
| 2.0 | OpenNMT from Stanford | with vanila params |
| 2.1 | OpenNMT from Stanford | with custom params |
| 3.0 | KerasNMT | base |
<!-- #endregion -->

<!-- #region id="OCm-PejL_tPz" -->
## Background
<!-- #endregion -->

<!-- #region id="r3fTtiji_vXU" -->
### Attention
<!-- #endregion -->

<!-- #region id="zigbv2r9_xw2" -->
<!-- #endregion -->

<!-- #region id="0DXgSQyt_--4" -->
### Encoder-decoder
<!-- #endregion -->

<!-- #region id="TM27b_nfAAug" -->
<!-- #endregion -->

<!-- #region id="zRyEbIhZ_0ot" -->
### Seq2seq

![](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)
<!-- #endregion -->

<!-- #region id="wlLEkyBhGEVT" -->
## Notebooks

1. [Keras seq2seq on dataset1 v1](https://gist.github.com/sparsh-ai/efa15acb4184f166580f8bf63d97e2db)
2. [Keras seq2seq on dataset1 v2](https://gist.github.com/sparsh-ai/2406cf60d9a66fd5602dd124588b3552)
3. [Keras seq2seq on dataset1 v3](https://gist.github.com/sparsh-ai/f96c2cea2b55d45dcccf669326d84a93)
4. [Keras seq2seq on dataset1 v4](https://gist.github.com/sparsh-ai/a1de8e333c24521e5fa9da651b503922)
5. [Keras seq2seq on dataset2 v1](https://gist.github.com/sparsh-ai/5628980c6e681ee7e32b37c2333ebee2)
6. [Keras seq2seq on dataset2 v2](https://gist.github.com/sparsh-ai/3457813ddc89a69b60518aaea095e40b)
7. [Keras seq2seq on dataset3 v1](https://gist.github.com/sparsh-ai/46dea35d52a67887ed40e16b09e7aab9)
8. [Keras seq2seq on dataset4 v1](https://gist.github.com/sparsh-ai/cc00aab4424f5fd85df1fc94c8027bfa)
9. [OpenNMT on dataset1 v1](https://gist.github.com/sparsh-ai/29503de22a76d29cb233f845120187f9)
10. [OpenNMT on dataset2 v1](https://gist.github.com/sparsh-ai/f4c86eaefed4915006615c01dd455ff4)
11. [OpenNMT on dataset2 v2](https://gist.github.com/sparsh-ai/1675a3341799ec57f00c21c03b283b6a)
<!-- #endregion -->

<!-- #region id="BC7cgjeY7cAM" -->
## References

1. [https://github.com/graykode/nlp-tutorial](https://github.com/graykode/nlp-tutorial)
2. [https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/neural-machine-translation](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/neural-machine-translation)
3. [https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/seq2seq_translation_tutorial.ipynb](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/seq2seq_translation_tutorial.ipynb)
4. [https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/attention](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/attention)
5. [https://github.com/sourcecode369/deep-natural-language-processing/tree/master/machine translation](https://github.com/sourcecode369/deep-natural-language-processing/tree/master/machine%20translation)
6. [https://github.com/sourcecode369/deep-natural-language-processing/tree/master/recurrent neural networks](https://github.com/sourcecode369/deep-natural-language-processing/tree/master/recurrent%20neural%20networks)
7. [https://github.com/sourcecode369/deep-natural-language-processing/tree/master/language modelling](https://github.com/sourcecode369/deep-natural-language-processing/tree/master/language%20modelling)
8. [https://github.com/fastai/course-nlp/blob/master/7-seq2seq-translation.ipynb](https://github.com/fastai/course-nlp/blob/master/7-seq2seq-translation.ipynb)
9. [https://github.com/fastai/course-nlp/blob/master/7b-seq2seq-attention-translation.ipynb](https://github.com/fastai/course-nlp/blob/master/7b-seq2seq-attention-translation.ipynb)
10. [https://github.com/fastai/course-nlp/blob/master/8-translation-transformer.ipynb](https://github.com/fastai/course-nlp/blob/master/8-translation-transformer.ipynb)
11. [https://pypi.org/project/txt2txt/](https://pypi.org/project/txt2txt/)
12. [https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_fairseq_translation.ipynb](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_fairseq_translation.ipynb)
13. [https://colab.research.google.com/github/theamrzaki/text_summurization_abstractive_methods/blob/master/Implementation B (Pointer Generator seq2seq network)/Model_4_generator_.ipynb](https://colab.research.google.com/github/theamrzaki/text_summurization_abstractive_methods/blob/master/Implementation%20B%20(Pointer%20Generator%20seq2seq%20network)/Model_4_generator_.ipynb)
14. [https://www.tensorflow.org/tutorials/text/nmt_with_attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
15. [https://arxiv.org/abs/2005.10213](https://arxiv.org/abs/2005.10213)
16. [https://gist.github.com/udibr/67be473cf053d8c38730](https://gist.github.com/udibr/67be473cf053d8c38730)
17. [Attention and Beam Search - a theoretical note](https://hackernoon.com/beam-search-attention-for-text-summarization-made-easy-tutorial-5-3b7186df7086)
18. [https://heyneat.ca/posts/python-code-character-prediction-with-lstm-rnn/](https://heyneat.ca/posts/python-code-character-prediction-with-lstm-rnn/)
19. [https://deeplanguageclass.github.io/lab/fairseq-transliteration/](https://deeplanguageclass.github.io/lab/fairseq-transliteration/)
20. [https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53](https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53)
<!-- #endregion -->

```python id="6-VoDIRfALa6"

```
