# Matching and Ranking models in Tensorflow

***Matching models on ML-1m and ranking models on Criteo (sample) dataset in TF v2.5***

## Matching models on Movielens ML-1m dataset

### 1. Bayesian Personalized Ranking (BPR)

In matrix factorization (MF), to compute the prediction we have to multiply the user factors to the item factors:

$$
\hat{x}_{ui} = \langle w_u,h_i \rangle = \sum_{f=1}^k w_{uf} \cdot h_{if}
$$

The usual approach for item recommenders is to predict a personalized score $\hat{x}_{ui}$ for an item that reflects the preference of the user for the item. Then the items are ranked by sorting them according to that score. Machine learning approaches are typically fit by using observed items as a positive sample and missing ones for the negative class. A perfect model would thus be useless, as it would classify as negative (non-interesting) all the items that were non-observed at training time. The only reason why such methods work is regularization.

BPR use a different approach. The training dataset is composed by triplets $(u,i,j)$ representing that user $u$ is assumed to prefer $i$ over $j$. For an implicit dataset this means that $u$ observed $i$ but not $j$:

$$
D_S := \{(u,i,j) \mid i \in I_u^+ \wedge j \in I \setminus I_u^+\}
$$

A machine learning model can be represented by a parameter vector $Θ$ which is found at fitting time. BPR wants to find the parameter vector that is most probable given the desired, but latent, preference structure $>_u$:

$$
\begin{align} p(\Theta \mid >_u) \propto p(>_u \mid \Theta)p(\Theta) \\ \prod_{u\in U} p(>_u \mid \Theta) = \dots = \prod_{(u,i,j) \in D_S} p(i >_u j \mid \Theta) \end{align}
$$

The probability that a user really prefers item $i$ to item $j$ is defined as:

$$
\begin{align} p(i >_u j \mid \Theta) := \sigma(\hat{x}_{uij}(\Theta)) \end{align}
$$

Where $σ$ represent the logistic sigmoid and $\hat{x}_{uij}(Θ)$ is an arbitrary real-valued function of $Θ$ (the output of your arbitrary model).

User $u_1$ has interacted with item $i_2$ but not item $i_1$, so we assume that this user prefers item $i_2$ over $i_1$. For items that the user have both interacted with, we cannot infer any preference. The same is true for two items that a user has not interacted yet (e.g. item $i_1$ and $i_4$ for user $u_1$).

![/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled.png)

To complete the Bayesian setting, we define a prior density for the parameters:

$$
p(\Theta) \sim N(0, \Sigma_\Theta)
$$

And we can now formulate the maximum posterior estimator:

$$
\begin{equation}
\begin{split}   BPR-OPT &=\log p(\Theta \mid >_u)\\
      &=\log p(>_u \mid \Theta) p(\Theta) \\ &= \log \prod_{(u,i,j) \in D_S} \sigma(\hat{x}_{uij})p(\Theta) \\ &= \sum_{(u,i,j) \in D_S} \log \sigma(\hat{x}_{uij}) + \log p(\Theta) \\ &= \sum_{(u,i,j) \in D_S} \log \sigma(\hat{x}_{uij}) - \lambda_\Theta ||\Theta||^2
\end{split}
\end{equation} 
$$

Where $λ_Θ$ are model specific regularization parameters.

Once obtained the log-likelihood, we need to maximize it in order to find our optimal $Θ$. As the criterion is differentiable, gradient descent algorithms are an obvious choiche for maximization.

The basic version of gradient descent consists in evaluating the gradient using all the available samples and then perform a single update. The problem with this is, in our case, that our training dataset is very skewed. Suppose an item $i$ is very popular. Then we have many terms of the form $\hat{x}_{uij}$ in the loss because for many users $u$ the item $i$ is compared against all negative items $j$. The other popular approach is stochastic gradient descent, where for each training sample an update is performed. This is a better approach, but the order in which the samples are traversed is crucial. To solve this issue BPR uses a stochastic gradient descent algorithm that chooses the triples randomly.

The gradient of BPR-OPT with respect to the model parameters is:

$$
\begin{equation}
\begin{split}   \frac{\partial BPR-OPT}{\partial \Theta} &= \sum_{(u,i,j) \in D_S} \frac{\partial}{\partial \Theta} \log \sigma (\hat{x}_{uij}) - \lambda_\Theta \frac{\partial}{\partial\Theta} || \Theta ||^2\\
      &=  \sum_{(u,i,j) \in D_S} \frac{-e^{-\hat{x}_{uij}}}{1+e^{-\hat{x}_{uij}}} \frac{\partial}{\partial \Theta}\hat{x}_{uij} - \lambda_\Theta \Theta
\end{split}
\end{equation} 
$$

In order to practically apply this learning schema to an existing algorithm, we first split the real valued preference term: $\hat{x}_{uij} := \hat{x}_{ui} − \hat{x}_{uj}$. And now we can apply any standard collaborative filtering model that predicts $\hat{x}_{ui}$.

The problem of predicting $\hat{x}_{ui}$ can be seen as the task of estimating a matrix $X:U×I$. With matrix factorization the target matrix $X$ is approximated by the matrix product of two low-rank matrices $W:|U|×k$ and $H:|I|×k$:

$$
X := WH^t
$$

The prediction formula can also be written as:

$$
\hat{x}_{ui} = \langle w_u,h_i \rangle = \sum_{f=1}^k w_{uf} \cdot h_{if}
$$

Besides the dot product ⟨⋅,⋅⟩, in general any kernel can be used.

We can now specify the derivatives:

$$
\frac{\partial}{\partial \theta} \hat{x}_{uij} = \begin{cases}
(h_{if} - h_{jf}) \text{ if } \theta=w_{uf}, \\
w_{uf} \text{ if } \theta = h_{if}, \\
-w_{uf} \text{ if } \theta = h_{jf}, \\
0 \text{ else }
\end{cases}
$$

Which basically means: user $u$ prefer $i$ over $j$, let's do the following:

- Increase the relevance (according to $u$) of features belonging to $i$ but not to $j$ and vice-versa
- Increase the relevance of features assigned to $i$
- Decrease the relevance of features assigned to $j$

Here is the implementation:

```python
class BPR(Model):
    def __init__(self, feature_columns, mode='inner', embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :mode: A string. 'inner' or 'dist'.
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
        super(BPR, self).__init__()
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # mode
        self.mode = mode
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1)
        # user info
        user_embed = self.user_embedding(user_inputs)  # (None, 1, dim)
        # item
        pos_embed = self.item_embedding(pos_inputs)  # (None, 1, dim)
        neg_embed = self.item_embedding(neg_inputs)  # (None, 1, dim)
        if self.mode == 'inner':
            # calculate positive item scores and negative item scores
            pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=-1)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=-1)  # (None, 1)
            # add loss. Computes softplus: log(exp(features) + 1)
            # self.add_loss(tf.reduce_mean(tf.math.softplus(neg_scores - pos_scores)))
1➡️        self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            # clip by norm
            # user_embed = tf.clip_by_norm(user_embed, 1, -1)
            # pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            # neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_scores = tf.reduce_sum(tf.square(user_embed - pos_embed), axis=-1)
            neg_scores = tf.reduce_sum(tf.square(user_embed - neg_embed), axis=-1)
            self.add_loss(tf.reduce_sum(tf.nn.relu(pos_scores - neg_scores + 0.5)))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
            outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()
```

1➡️ To calculate loss, we are first taking the sigmoid of the difference between positive and negative pairs, and then taking mean of the log of these sigmoid values. This is same as equation (4), without regularization term.

[Here](https://nbviewer.org/gist/sparsh-ai/cd6171c364ecc51a912cc0d50fa3ccdc) is the Jupyter notebook of this experiment.

---

### 2. Neural Collaborative Filtering (NCF)

NCF is a deep learning-based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Despite the effectiveness of MF for collaborative filtering, it is well-known that its performance can be hindered by the simple choice of the interaction function — the inner product. The hypothesis is that the inner product, which simply combines the multiplication of latent features linearly, may not be sufficient to capture the complex structure of user interaction data. NCF tries to learn this interaction function from data. 

![[Source](https://d2l.ai/chapter_recommender-systems/neumf.html)](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-1.png)

[Source](https://d2l.ai/chapter_recommender-systems/neumf.html)

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings and combines the two models by concatenating their last hidden layer.

The interaction matrix is based on implicit data and contains only 1 or 0. Here a value of 1 indicates that there is an interaction between user u and item i; however, it does not mean u actually likes i. Similarly, a value of 0 does not necessarily mean u does not like i, it can be that the user is not aware of the item. This poses challenges in learning from implicit data since it provides only noisy signals about users’ preferences. While observed entries at least reflect users’ interest in items, the unobserved entries can be just missing data and there is a natural scarcity of negative feedback.

The loss function can either be pointwise or pairwise. Due to the non-convexity of the objective function of NeuMF, gradient-based optimization methods only find locally optimal solutions. It is reported that initialization plays an important role in the convergence and performance of deep learning models. Since NeuMF is an ensemble of GMF and MLP, we usually initialize NeuMF using the pre-trained models of GMF and MLP.

Here is the implementation:

```python
class DNN(Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
		"""
		DNN part
		:param hidden_units: A list. List of hidden layer units's numbers
		:param activation: A string. Activation function
		:param dnn_dropout: A scalar. dropout number
		"""
		super(DNN, self).__init__(**kwargs)
		self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
		self.dropout = Dropout(dnn_dropout)

	def call(self, inputs, **kwargs):
		x = inputs
		for dnn in self.dnn_network:
1➡️		x = dnn(x)
		x = self.dropout(x)
		return x
```

```python
class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        """
        NCF model
        :param feature_columns: A list. user feature columns + item feature columns
        :param hidden_units: A list.
        :param dropout: A scalar.
        :param activation: A string.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NCF, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [64, 32, 16, 8]
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # MF user embedding
        self.mf_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.user_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MF item embedding
        self.mf_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.item_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MLP user embedding
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # MLP item embedding
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # dnn
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1), (None, 1/101)
        # user info
        mf_user_embed = self.mf_user_embedding(user_inputs)  # (None, 1, dim)
        mlp_user_embed = self.mlp_user_embedding(user_inputs)  # (None, 1, dim)
        # item
        mf_pos_embed = self.mf_item_embedding(pos_inputs)  # (None, 1, dim)
        mf_neg_embed = self.mf_item_embedding(neg_inputs)  # (None, 1/101, dim)
        mlp_pos_embed = self.mlp_item_embedding(pos_inputs)  # (None, 1, dim)
        mlp_neg_embed = self.mlp_item_embedding(neg_inputs)  # (None, 1/101, dim)
        # MF
2➡️    mf_pos_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_pos_embed))  # (None, 1, dim)
        mf_neg_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_neg_embed))  # (None, 1, dim)
        # MLP
3➡️    mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)  # (None, 1, 2 * dim)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)  # (None, 1/101, 2 * dim)
        mlp_pos_vector = self.dnn(mlp_pos_vector)  # (None, 1, dim)
        mlp_neg_vector = self.dnn(mlp_neg_vector)  # (None, 1/101, dim)
        # concat
        pos_vector = tf.concat([mf_pos_vector, mlp_pos_vector], axis=-1)  # (None, 1, 2 * dim)
        neg_vector = tf.concat([mf_neg_vector, mlp_neg_vector], axis=-1)  # (None, 1/101, 2 * dim)
        # pos_vector = mlp_pos_vector
        # neg_vector = mlp_neg_vector
        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)
        # loss
4➡️    losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()
```

1➡️ In `self.dnn_network`, we created a list of Dense layers as per the given list of hidden nodes. Now, we are stitching them up in a linear fashion, passing the output of first hidden layer to the next one.

2➡️ In MF part, we are multiplying the user and positive (resp. negative) item embeddings.

3➡️ In MLP part, we are concatenating these embeddings.

4➡️ We are calculating cross-entropy loss.

[Here](https://nbviewer.org/gist/sparsh-ai/1b43e0adba59114598f18dcd9ad239b0) is the Jupyter notebook of this experiment.

---

### 3. Convolutional Sequence Embedding Recommendation (Caser)

Top-N sequential recommendation models each user as a sequence of items interacted in the past and aims to predict top-N ranked items that a user will likely interact in a 'near future'. The order of interaction implies that sequential patterns play an important role where more recent items in a sequence have a larger impact on the next item. Convolutional Sequence Embedding Recommendation Model (Caser) address this requirement by embedding a sequence of recent items into an image' in the time and latent spaces and learn sequential patterns as local features of the image using convolutional filters. This approach provides a unified and flexible network structure for capturing both general preferences and sequential patterns.

![/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-2.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-2.png)

Caser adopts convolutional neural networks capture the dynamic pattern influences of users’ recent activities. The main component of Caser consists of a horizontal convolutional network and a vertical convolutional network, aiming to uncover the union-level and point-level sequence patterns, respectively. Point-level pattern indicates the impact of single item in the historical sequence on the target item, while union level pattern implies the influences of several previous actions on the subsequent target. For example, buying both milk and butter together leads to higher probability of buying flour than just buying one of them. Moreover, users’ general interests, or long term preferences are also modeled in the last fully-connected layers, resulting in a more comprehensive modeling of user interests.

Here is the implementation:

```python
class Caser(Model):
    def __init__(self, feature_columns, maxlen=40, hor_n=2, hor_h=8, ver_n=8, dropout=0.5, activation='relu', embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param dropout: A scalar. The number of dropout.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Caser, self).__init__()
        # maxlen
        self.maxlen = maxlen
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # total number of item set
        self.total_item = self.item_fea_col['feat_num']
        # horizontal filters
        self.hor_n = hor_n
        self.hor_h = hor_h if hor_h <= self.maxlen else self.maxlen
        # vertical filters
        self.ver_n = ver_n
        self.ver_w = 1
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'] * 2,
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # horizontal conv
        self.hor_conv = Conv1D(filters=self.hor_n, kernel_size=self.hor_h)
        # vertical conv, should transpose
        self.ver_conv = Conv1D(filters=self.ver_n, kernel_size=self.ver_w)
        # max_pooling
        self.pooling = GlobalMaxPooling1D()
        # dense
        self.dense = Dense(self.embed_dim, activation=activation)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        # input
        user_inputs, seq_inputs, item_inputs = inputs
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # horizontal conv (None, (maxlen - kernel_size + 2 * pad) / stride +1, hor_n)
        hor_info = self.hor_conv(seq_embed)
        hor_info = self.pooling(hor_info)  # (None, hor_n)
        # vertical conv  (None, (dim - 1 + 2 * pad) / stride + 1, ver_n)
        ver_info = self.ver_conv(tf.transpose(seq_embed, perm=(0, 2, 1)))
        ver_info = tf.reshape(ver_info, shape=(-1, ver_info.shape[1] * ver_info.shape[2]))  # (None, ?)
        # info
        seq_info = self.dense(tf.concat([hor_info, ver_info], axis=-1))  # (None, d)
1➡️    seq_info = self.dropout(seq_info)
        # concat
2➡️     info = tf.concat([seq_info, user_embed], axis=-1)  # (None, 2 * d)
        # item info
        item_embed = self.item2_embedding(tf.squeeze(item_inputs, axis=-1))  # (None, dim)
        # predict
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(info, item_embed), axis=1, keepdims=True))
        return outputs

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        item_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, item_inputs],
              outputs=self.call([user_inputs, seq_inputs, item_inputs])).summary()
```

1➡️ We applied the horizontal and vertical convolution filter on the sequence data and concatenated the embeddings. Now, we are applying dense layer and a dropout layer.

2➡️ We concatenated this sequence embedding with user embeddings to get the final user vector. It is now assumed that the whole user behaviour is being summarized with this embedding.

[Here](https://nbviewer.org/gist/sparsh-ai/79b9cba4161439e59caec63bbcd9fe69) is the Jupyter notebook of this experiment.

---

### 4. Self-Attentive Sequential Recommendation (SASRec)

Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the ‘context’ of users’ activities on the basis of actions they have performed recently. To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user’s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. SASRec captures the long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC).

![US512148 _ General Recommenders-L186674 _ SASRec Model.drawio.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow---.drawio.png)

At each time step, SASRec seeks to identify which items are ‘relevant’ from a user’s action history, and use them to predict the next item. Extensive empirical studies show that this method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models.

We adopt the binary cross entropy loss as the objective function:

$$
-\sum_{S^u\in S} \sum_{t \in [1,2,\dots,n]}\left[ log(\sigma(r_{o_t,t})) + \sum_{j \notin S^u} log(1-\sigma(r_{j,t})) \right]
$$

Here is the implementation:

```python
class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """
        Feed Forward Network
        :param hidden_unit: A scalar. W1
        :param d_model: A scalar. W2
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
1➡️    output = self.conv2(x)
        return output

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., norm_training=True, causality=True):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, causality)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=norm_training)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # （None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
  2➡️  out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2
```

```python
class SASRec(tf.keras.Model):
    def __init__(self, item_fea_col, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-6):
        """
        SASRec model
        :param item_fea_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param maxlen: A scalar. Number of length of sequence
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        :param embed_reg: A scalar. The regularizer of embedding
        """
        super(SASRec, self).__init__()
        # sequence length
        self.maxlen = maxlen
        # item feature columns
        self.item_fea_col = item_fea_col
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # d_model must be the same as embedding_dim, because of residual connection
        self.d_model = self.embed_dim
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=self.maxlen,
                                       input_length=1,
                                       output_dim=self.embed_dim,
                                       mask_zero=False,
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dropout)
        # attention block
3➡️    self.encoder_layer = [EncoderLayer(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]

    def call(self, inputs, training=None):
        # inputs
        seq_inputs, pos_inputs, neg_inputs = inputs  # (None, maxlen), (None, 1), (None, 1)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # pos encoding
        # pos_encoding = positional_encoding(seq_inputs, self.embed_dim)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed  # (None, maxlen, dim)
        att_outputs *= mask

        # self-attention
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
4➡️     user_info = tf.expand_dims(att_outputs[:, -1], axis=1)  # (None, 1, dim)
        # item info
        pos_info = self.item_embedding(pos_inputs)  # (None, 1, dim)
        neg_info = self.item_embedding(neg_inputs)  # (None, 1/100, dim)
        pos_logits = tf.reduce_sum(user_info * pos_info, axis=-1)  # (None, 1)
        neg_logits = tf.reduce_sum(user_info * neg_info, axis=-1)  # (None, 1)
        # loss
5➡️    losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[seq_inputs, pos_inputs, neg_inputs],
                       outputs=self.call([seq_inputs, pos_inputs, neg_inputs])).summary()
```

1➡️ FFN layer consists of two 1-d convolution layers stitched in a linear fashion.

2➡️ Encoder layer consists of encoded output from multi-head attention layer, which is being passed to FFN layer.

3➡️ `self.encoder_layer` is a list of encoder blocks. blocks=1 would mean single block only.

4➡️ Final user vector is being extracted from the last layer of encoder blocks.

5➡️ We are using cross-entropy loss.

[Here](https://nbviewer.org/gist/sparsh-ai/43cc1b4102c9b9f9c93a1c1289057746) is the Jupyter notebook of this experiment.

---

### 5. Self-Attentive Sequential Recommendation

The loss function can either be pointwise or pairwise. Due to the non-convexity of the objective function of NeuMF, gradient-based optimization methods only find locally optimal solutions. It is reported that initialization plays an important role in the convergence and performance of deep learning models. Since NeuMF is an ensemble of GMF and MLP, we usually initialize NeuMF using the pre-trained models of GMF and MLP.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-3.png)

Here is the implementation:

```python
class AttRec(Model):
    def __init__(self, feature_columns, maxlen=40, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6, **kwargs):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param gamma: A scalar. if mode == 'dist', gamma is the margin.
        :param mode: A string. inner or dist.
        :param w: A scalar. The weight of short interest.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(AttRec, self).__init__(**kwargs)
        # maxlen
        self.maxlen = maxlen
        # w
        self.w = w
        self.gamma = gamma
        self.mode = mode
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding, not share embedding
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # self-attention
        self.self_attention = SelfAttention_Layer()

    def call(self, inputs, **kwargs):
        # input
        user_inputs, seq_inputs, pos_inputs, neg_inputs = inputs
        # mask
        # mask = self.item_embedding.compute_mask(seq_inputs)
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # item
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)
        # item2 embed
        pos_embed2 = self.item2_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed2 = self.item2_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)

        # short-term interest
1➡️    short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)

        # mode
        if self.mode == 'inner':
            # long-term interest, pos and neg
2➡️        pos_long_interest = tf.multiply(user_embed, pos_embed2)
            neg_long_interest = tf.multiply(user_embed, neg_embed2)
            # combine
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            # clip by norm
            user_embed = tf.clip_by_norm(user_embed, 1, -1)
            pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_embed2 = tf.clip_by_norm(pos_embed2, 1, -1)
            neg_embed2 = tf.clip_by_norm(neg_embed2, 1, -1)
            # distance
            # long-term interest, pos and neg
            pos_long_interest = tf.square(user_embed - pos_embed2)  # (None, dim)
            neg_long_interest = tf.square(user_embed - neg_embed2)  # (None, dim)
            # combine. Here is a difference from the original paper.
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - neg_embed), axis=-1, keepdims=True)
            # minimize loss
            # self.add_loss(tf.reduce_sum(tf.maximum(pos_scores - neg_scores + self.gamma, 0)))
3➡️        self.add_loss(tf.reduce_sum(tf.nn.relu(pos_scores - neg_scores + self.gamma)))
        return pos_scores, neg_scores

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs], 
            outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs])).summary()
```

1➡️ Short-term interest vector is calculated by encoding the item sequences using self-attention mechanism.

2➡️ Long-term interest vector is the vector multiplication of user embedding and positive (resp. negative) item embedding.

3➡️ Loss is the sum of euclidean distance loss of both short and long-term interest vectors.

[Here](https://nbviewer.org/gist/sparsh-ai/eb21eee590e13931dac8019cd4dcb576) is the Jupyter notebook of this experiment.

---

## Ranking models on Criteo (sample) Ad display dataset

### 1. Factorization Machines (FM)

Factorization Machines (FMs) are a supervised learning approach that enhances the linear regression model by incorporating the second-order feature interactions. Factorization Machine type algorithms are a combination of linear regression and matrix factorization, the cool idea behind this type of algorithm is it aims model interactions between features (a.k.a attributes, explanatory variables) using factorized parameters. By doing so it has the ability to estimate all interactions between features even with extremely sparse data.

Factorization machines (FM) [Rendle, 2010], proposed by Steffen Rendle in 2010, is a supervised algorithm that can be used for classification, regression, and ranking tasks. It quickly took notice and became a popular and impactful method for making predictions and recommendations. Particularly, it is a generalization of the linear regression model and the matrix factorization model. Moreover, it is reminiscent of support vector machines with a polynomial kernel. The strengths of factorization machines over the linear regression and matrix factorization are: (1) it can model χ -way variable interactions, where χ is the number of polynomial order and is usually set to two. (2) A fast optimization algorithm associated with factorization machines can reduce the polynomial computation time to linear complexity, making it extremely efficient especially for high dimensional sparse inputs. For these reasons, factorization machines are widely employed in modern advertisement and products recommendations.

Most recommendation problems assume that we have a consumption/rating dataset formed by a collection of *(user, item, rating*) tuples. This is the starting point for most variations of Collaborative Filtering algorithms and they have proven to yield nice results; however, in many applications, we have plenty of item metadata (tags, categories, genres) that can be used to make better predictions. This is one of the benefits of using Factorization Machines with feature-rich datasets, for which there is a natural way in which extra features can be included in the model and higher-order interactions can be modeled using the dimensionality parameter d. For sparse datasets, a second-order FM model suffices, since there is not enough information to estimate more complex interactions.

![/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-4.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-4.png)

$$
f(x) = w_0 + \sum_{p=1}^Pw_px_p + \sum_{p=1}^{P-1}\sum_{q=p+1}^Pw_{p,q}x_px_q
$$

This model formulation may look familiar — it's simply a quadratic linear regression. However, unlike polynomial linear models which estimate each interaction term separately, FMs instead use factorized interaction parameters: feature interaction weights are represented as the inner product of the two features' latent factor space embeddings:

$$
f(x) = w_0 + \sum_{p=1}^Pw_px_p + \sum_{p=1}^{P-1}\sum_{q=p+1}^P\langle v_p,v_q \rangle x_px_q
$$

This greatly decreases the number of parameters to estimate while at the same time facilitating more accurate estimation by breaking the strict independence criteria between interaction terms. Consider a realistic recommendation data set with 1,000,000 users and 10,000 items. A quadratic linear model would need to estimate U + I + UI ~ 10 billion parameters. A FM model of dimension F=10 would need only U + I + F(U + I) ~ 11 million parameters. Additionally, many common MF algorithms (including SVD++, ALS) can be re-formulated as special cases of the more general/flexible FM model class.

The above equation can be rewritten as:

$$
\begin{align*}
\hat{y}(\textbf{x}) = w_{0} + \sum_{i=1}^{n} w_{i} x_{i} +  \sum_{i=1}^n \sum_{j=i+1}^n \hat{w}_{ij} x_{i} x_{j}
\end{align*}
$$

where,

- $w_0$ is the global bias
- $w_i$ denotes the weight of the i-th feature,
- $\hat{w}_{ij} = v_i^Tv_j$ denotes the weight of the cross feature $x_ix_j$
- $v_i \in \mathcal{R}^k$ denotes the embedding vector for feature $i$
- $k$ denotes the size of embedding vector

<aside>
💡 For large, sparse datasets...FM and FFM is good. But for small, dense datasets...try to avoid.

</aside>

Factorization machines appeared to be the method which answered the challenge!

|  | Accuracy | Speed | Sparsity |
| --- | --- | --- | --- |
| Collaborative Filtering | Too Accurate | Suitable | Suitable |
| SVM | Too Accurate | Suitable | Unsuitable |
| Random Forest/CART | General Accuracy | Unsuitable | Unsuitable |
| Factorization Machines (FM) | General Accuracy | Quick | Designed for it |

To learn the FM model, we can use the MSE loss for regression task, the cross entropy loss for classification tasks, and the BPR loss for ranking task. Standard optimizers such as SGD and Adam are viable for optimization.

Here is the implementation:

```python
class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        # mapping
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
1➡️    first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
2➡️    second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs

class FM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ First-order embeddings are in the simple linear-regression style. It contains a bias term and a list of weights.

2➡️ Second-order embeddings is a k-length embedding matrix for each field.

[Here](https://nbviewer.org/gist/sparsh-ai/f070e6b7cf99eb2a5b796447c011890e) is the Jupyter notebook of this experiment.

---

### 2. Field-aware Factorization Machines (FFM)

Despite effectiveness, FM can be hindered by its modelling of all feature interactions with the same weight, as not all feature interactions are equally useful and predictive. For example, the interactions with useless features may even introduce noises and adversely degrade the performance.

|  | For each | Learn |
| --- | --- | --- |
| Linear | feature | a weight |
| Poly | feature pair | a weight |
| FM | feature | a latent vector |
| FFM | feature | multiple latent vectors |

Field-aware factorization machine (FFM) is an extension to FM. It was originally introduced in [2]. The advantage of FFM over FM is that it uses different factorized latent factors for different groups of features. The "group" is called "field" in the context of FFM. Putting features into fields resolves the issue that the latent factors shared by features that intuitively represent different categories of information may not well generalize the correlation.

FFM addresses this issue by splitting the original latent space into smaller latent spaces specific to the fields of the features.

$$
\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n \langle \mathbf{v}_{i, f_{2}} \cdot \mathbf{v}_{j, f_{1}} \rangle x_i x_j
$$

Here is the implementation:

```python
class FFM_Layer(Layer):
    def __init__(self, sparse_feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        :param dense_feature_columns: A list. sparse column feature information.
        :param k: A scalar. The latent vector
        :param w_reg: A scalar. The regularization coefficient of parameter w
		:param v_reg: A scalar. The regularization coefficient of parameter v
        """
        super(FFM_Layer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.field_num = len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # field second order
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
1➡️             second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        return first_order + second_order
```

```python
class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        FFM architecture
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(self.sparse_feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ Second-order features are being calculated as per the last-part of this equation: 

$$
\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n \langle \mathbf{v}_{i, f_{2}} \cdot \mathbf{v}_{j, f_{1}} \rangle x_i x_j
$$

[Here](https://nbviewer.org/gist/sparsh-ai/bb4620cf109c34622b5f10fa4e332445) is the Jupyter notebook of this experiment.

---

### 3. Wide & Deep Learning (W&D)

Wide and Deep Learning Model, proposed by Google, 2016, is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-5.png)

To understand the concept of deep & wide recommendations, it’s best to think of it as two separate, but collaborating, engines. The wide model, often referred to in the literature as the linear model, memorizes users and their past product choices. Its inputs may consist simply of a user identifier and a product identifier, though other attributes relevant to the pattern (such as time of day) may also be incorporated.

![/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-6.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-6.png)

The deep portion of the model, so named as it is a deep neural network, examines the generalizable attributes of a user and their product choices. From these, the model learns the broader characteristics that tend to favor users’ product selections.

Together, the wide and deep submodels are trained on historical product selections by individual users to predict future product selections. The end result is a single model capable of calculating the probability with which a user will purchase a given item, given both memorized past choices and generalizations about a user’s preferences. These probabilities form the basis for user-specific product rankings, which can be used for making recommendations.

The goal with wide and deep recommenders is to provide the same level of customer intimacy that, for example, our favorite barista does. This model uses explicit and implicit feedback to expand the considerations set for customers. Wide and deep recommenders go beyond simple weighted averaging of customer feedback found in some collaborative filters to balance what is understood about the individual with what is known about similar customers. If done properly, the recommendations make the customer feel understood and this should translate into greater value for both the customer and the business.

The intuitive logic of the wide-and-deep recommender belies the complexity of its actual construction. Inputs must be defined separately for each of the wide and deep portions of the model and each must be trained in a coordinated manner to arrive at a single output, but tuned using optimizers specific to the nature of each submodel. Thankfully, the **[Tensorflow DNNLinearCombinedClassifier estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)** provides a pre-packaged architecture, greatly simplifying the assembly of the overall model.


Here is the implementation:

```python
class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """
        Linear Part
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        return result

class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """Deep Neural Network
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
```

```python
class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, w_reg=1e-6):
        """
        Wide&Deep
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_reg: A scalar. The regularizer of Linear.
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear(self.feature_length, w_reg=w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i])
                                  for i in range(inputs.shape[1])], axis=-1)
        x = sparse_embed  # (batch_size, field * embed_dim)
        # Wide
        wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
1➡️    wide_out = self.linear(wide_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
2➡️    outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ In wide-part, we are passing the inputs and index mappings of sparse features to the linear layer.

2➡️ Final output is the average weight on the wide and deep model outputs.

[Here](https://nbviewer.org/gist/sparsh-ai/966844969941e710da1e204831a37638) is the Jupyter notebook of this experiment.

---

### 4. Deep Crossing

The input of Deep Crossing is a set of individual features that can be either dense or sparse. The important crossing features are discovered implicitly by the networks, which are comprised of an embedding and stacking layer, as well as a cascade of Residual Units.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-7.png)

Here is the implementation:

```python
class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: A list. Neural network hidden units.
        :param dim_stack: A scalar. The dimension of inputs unit.
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
1➡️    outputs = self.relu(x + inputs)
        return outputs
```

```python
class Deep_Crossing(Model):
    def __init__(self, feature_columns, hidden_units, res_dropout=0., embed_reg=1e-6):
        """
        Deep&Crossing
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param res_dropout: A scalar. Dropout of resnet.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Deep_Crossing, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding layers
        embed_layers_len = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        self.res_network = [Residual_Units(unit, embed_layers_len) for unit in hidden_units]
        self.res_dropout = Dropout(res_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        r = sparse_embed
2➡️    for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ In residual units, we are applying relu activation on input and dense layer output of these inputs.

2➡️ This is how we are stacking multiple residual units in a sequential fashion.

[Here](https://nbviewer.org/gist/sparsh-ai/d0c5c206daea9ffece59d587ceb02e42) is the Jupyter notebook of this experiment.

---

### 5. Product-based Neural Network (PNN)

PNN uses an embedding layer to learn a distributed representation of the categorical data, a product layer to capture interactive patterns between inter-field categories, and further fully connected layers to explore high-order feature interactions.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-8.png)

Here is the implementation:

```python
class PNN(Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-6, w_z_reg=1e-6, w_p_reg=1e-6, l_b_reg=1e-6):
        """
        Product-based Neural Networks
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param mode: A string. 'in' IPNN or 'out'OPNN.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_z_reg: A scalar. The regularizer of w_z_ in product layer
        :param w_p_reg: A scalar. The regularizer of w_p in product layer
        :param l_b_reg: A scalar. The regularizer of l_b in product layer
        """
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.sparse_feature_columns = feature_columns
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              hidden_units[0]),
                                       initializer='random_uniform',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        # out
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = DNN(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        sparse_inputs = inputs
        sparse_embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])  # (None, field_num, embed_dim)
        # product layer
        row = []
        col = []
        for i in range(len(self.sparse_feature_columns) - 1):
            for j in range(i + 1, len(self.sparse_feature_columns)):
                row.append(i)
                col.append(j)
        p = tf.gather(sparse_embed, row, axis=1)
        q = tf.gather(sparse_embed, col, axis=1)
        if self.mode == 'in':
            l_p = tf.tensordot(p*q, self.w_p, axes=2)  # (None, hidden[0])
        else:  # out
            u = tf.expand_dims(q, 2)  # (None, field_num(field_num-1)/2, 1, emb_dim)
            v = tf.expand_dims(p, 2)  # (None, field_num(field_num-1)/2, 1, emb_dim)
            l_p = tf.tensordot(tf.matmul(tf.transpose(u, [0, 1, 3, 2]), v), self.w_p, axes=3)  # (None, hidden[0])

        l_z = tf.tensordot(sparse_embed, self.w_z, axes=2)  # (None, hidden[0])
1➡️    l_1 = tf.nn.relu(tf.concat([l_z + l_p + self.l_b], axis=-1))
        # dnn layer
        dnn_x = self.dnn_network(l_1)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ Concatenating the sparse embeddings, product-layer dense embeddings, and bias, and the Relu activation on this feature vector.

[Here](https://nbviewer.org/gist/sparsh-ai/1547bd11bdf5feed5dc68cb2ff6d1a08) is the Jupyter notebook of this experiment.

---

### 6. Deep and Cross Network (DCN)

In real world recommendation systems, we often have large and sparse feature space. So identifying effective feature processes in this setting would often require manual feature engineering or exhaustive search, which is highly inefficient. To tackle this issue, Google Research team has proposed Deep and Cross Network, DCN**.**

It starts with an input layer, typically an embedding layer, followed by a cross network containing multiple cross layers that models explicitly feature interactions, and then combines with a deep network that models implicit feature interactions. The deep network is just a traditional multilayer construction. But the core of DCN is really the cross network. It explicitly applies feature crossing at each layer. And the highest polynomial degree increases with layer depth. The figure here shows the deep and cross layer in the mathematical form.

There are a couple of ways to combine the cross network and the deep network:

- Stack the deep network on top of the cross network.
- Place deep & cross networks in parallel.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-9.png)

Here is the implementation:

```python
class CrossNetwork(Layer):
    def __init__(self, layer_num, reg_w=1e-6, reg_b=1e-6):
        """CrossNetwork
        :param layer_num: A scalar. The depth of cross network
        :param reg_w: A scalar. The regularizer of w
        :param reg_b: A scalar. The regularizer of b
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0  # (None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, dim, dim)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l

class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """Deep Neural Network
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
```

```python
class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, cross_w_reg=1e-6, cross_b_reg=1e-6):
        """
        Deep&Cross Network
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cross_w_reg: A scalar. The regularizer of cross network.
        :param cross_b_reg: A scalar. The regularizer of cross network.
        """
        super(DCN, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = sparse_embed
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
1➡️    total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ For each sparse feature, we first created an embedding vector and then passed the concatenated vector of these embeddings to the cross and deep network. Now, we are concatenating the outputs we received from these two networks and this vector we will pass to the final dense and sigmoid layer.

[Here](https://nbviewer.org/gist/sparsh-ai/c29b15a5b642f95b5c47dd24fcfbf534) is the Jupyter notebook of this experiment.

---

### 7. Neural Factorization Machine (NFM)

NFM seamlessly combines the linearity of FM in modelling second-order feature interactions and the non-linearity of neural network in modelling higher-order feature interactions. Conceptually, NFM is more expressive than FM since FM can be seen as a special case of NFM without hidden layers.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-10.png)

Here is the implementation:

```python
class NFM(Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=1e-6):
        """
        NFM architecture
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param bn_use: A Boolean. Use BatchNormalization or not.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        # Inputs layer
        sparse_inputs = inputs
        # Embedding layer
        sparse_embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])  # (None, filed_num, embed_dim)
        # Bi-Interaction Layer
1➡️    sparse_embed = 0.5 * (tf.pow(tf.reduce_sum(sparse_embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(sparse_embed, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = sparse_embed
        # BatchNormalization
        x = self.bn(x, training=self.bn_use)
        # Hidden Layers
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ The bi-interaction pooling feature is being calculated as per this equation:

$$
f_{BI}(\mathcal{V}_x) = \dfrac{1}{2}\left[ (\sum_{i=1}^n x_iv_i)^2 - \sum_{i=1}^n(x_iv_i)^2 \right]
$$

[Here](https://nbviewer.org/gist/sparsh-ai/b16298b515077e07ad4dd4d2d575e430) is the Jupyter notebook of this experiment.

---

### 8. Attentional Factorization Machines (AFM)

Improves FM by discriminating the importance of different feature interactions. It learns the importance of each feature interaction from data via a neural attention network. Empirically, it is shown on regression task AFM betters FM with a 8.6% relative improvement, and consistently outperforms the state-of-the-art deep learning methods Wide&Deep and DeepCross with a much simpler structure and fewer model parameters.

![Untitled](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-11.png)

Formally, the AFM model can be defined as:

$$
\hat{y}_{AFM} (x) = w_0 + \sum_{i=1}^nw_ix_i + p^T\sum_{i=1}^n\sum_{j=i+1}^na_{ij}(v_i\odot v_j)x_ix_j
$$

Here is the implementation:

```python
class AFM(Model):
    def __init__(self, feature_columns, mode, att_vector=8, activation='relu', dropout=0.5, embed_reg=1e-6):
        """
        AFM 
        :param feature_columns: A list. sparse column feature information.
        :param mode: A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param att_vector: A scalar. attention vector.
        :param activation: A string. Activation function of attention.
        :param dropout: A scalar. Dropout.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        if self.mode == 'att':
            self.attention_W = Dense(units=att_vector, activation=activation, use_bias=True)
            self.attention_dense = Dense(units=1, activation=None)
        self.dropout = Dropout(dropout)
        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Input Layer
        sparse_inputs = inputs
        # Embedding Layer 
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])  # (None, len(sparse_inputs), embed_dim)
        # Pair-wise Interaction Layer
        row = []
        col = []
        for r, c in itertools.combinations(range(len(self.sparse_feature_columns)), 2):
            row.append(r)
            col.append(c)
        p = tf.gather(embed, row, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        q = tf.gather(embed, col, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        bi_interaction = p * q  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
1➡️    # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(bi_interaction, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(bi_interaction, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self.attention(bi_interaction)  # (None, k)
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ The output feature vector from the bi-interaction layer can be pooled either using max or average mode or using attention mechanism.

[Here](https://nbviewer.org/gist/sparsh-ai/0a1aa59c812f5ed2b5fd6cd6760e81a9) is the Jupyter notebook of this experiment.

---

### 9. Deep Factorization Machines (DeepFM)

DeepFM consists of an FM component and a deep component which are integrated in a parallel structure. The FM component is the same as the 2-way factorization machines which is used to model the low-order feature interactions. The deep component is a multi-layered perceptron that is used to capture high-order feature interactions and nonlinearities. These two components share the same inputs/embeddings and their outputs are summed up as the final prediction. It is worth pointing out that the spirit of DeepFM resembles that of the Wide & Deep architecture which can capture both memorization and generalization. The advantages of DeepFM over the Wide & Deep model is that it reduces the effort of hand-crafted feature engineering by identifying feature combinations automatically.

![/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-12.png](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-12.png)

Here is the implementation:

```python
class DeepFM(Model):
	def __init__(self, feature_columns, hidden_units=(200, 200, 200), dnn_dropout=0.,
				 activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
		"""
		DeepFM
		:param feature_columns: A list. sparse column feature information.
		:param hidden_units: A list. A list of dnn hidden units.
		:param dnn_dropout: A scalar. Dropout of dnn.
		:param activation: A string. Activation function of dnn.
		:param fm_w_reg: A scalar. The regularizer of w in fm.
		:param embed_reg: A scalar. The regularizer of embedding.
		"""
		super(DeepFM, self).__init__()
		self.sparse_feature_columns = feature_columns
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
										 input_length=1,
										 output_dim=feat['embed_dim'],
										 embeddings_initializer='random_normal',
										 embeddings_regularizer=l2(embed_reg))
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.index_mapping = []
		self.feature_length = 0
		for feat in self.sparse_feature_columns:
			self.index_mapping.append(self.feature_length)
			self.feature_length += feat['feat_num']
		self.embed_dim = self.sparse_feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
		self.fm = FM(self.feature_length, fm_w_reg)
		self.dnn = DNN(hidden_units, activation, dnn_dropout)
		self.dense = Dense(1, activation=None)

	def call(self, inputs, **kwargs):
		sparse_inputs = inputs
		# embedding
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)
		# wide
		sparse_inputs = sparse_inputs + tf.convert_to_tensor(self.index_mapping)
		wide_inputs = {'sparse_inputs': sparse_inputs,
					   'embed_inputs': tf.reshape(sparse_embed, shape=(-1, sparse_inputs.shape[1], self.embed_dim))}
		wide_outputs = self.fm(wide_inputs)  # (batch_size, 1)
		# deep
		deep_outputs = self.dnn(sparse_embed)
		deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
		# outputs
1➡️	outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
		return outputs

	def summary(self):
		sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
		Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ We are adding the outputs of FM layer and dense layer and passing through sigmoid activation to get the final model output.

[Here](https://nbviewer.org/gist/sparsh-ai/c4840698c73d1d7139147a654736fb2d) is the Jupyter notebook of this experiment.

---

### 10. Extreme Deep Factorization Machines (xDeepFM)

xDeepFM combines the CIN and a classical DNN into one unified model. xDeepFM is able to learn certain bounded-degree feature interactions explicitly; on the other hand, it can learn arbitrary low- and high-order feature interactions implicitly.

![The architecture of xDeepFM.](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-13.png)

The architecture of xDeepFM.

Compressed Interaction Network (CIN) aims to generate feature interactions in an explicit fashion and at the vector-wise level. CIN share some functionalities with convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

![Components and architecture of the Compressed Interaction Network (CIN).](/img/content-tutorials-raw-r170254-matching-and-ranking-models-in-tensorflow-untitled-14.png)

Components and architecture of the Compressed Interaction Network (CIN).

Here is the implementation:

```python
class CIN(Layer):
    def __init__(self, cin_size, l2_reg=1e-4):
        """CIN
        :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
        :param l2_reg: A scalar. L2 regularization.
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  # dim * (None, filed_nums[i], 1)

            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)

        return result
```

```python
class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-6, cin_reg=1e-6, w_reg=1e-6):
        """
        xDeepFM
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cin_reg: A scalar. The regularizer of cin.
        :param w_reg: A scalar. The regularizer of Linear.
        """
        super(xDeepFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.linear = Linear(self.feature_length, w_reg)
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        # Linear
        linear_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        linear_out = self.linear(linear_inputs)  # (batch_size, 1)
        # cin
        embed = [self.embed_layers['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])]
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (batch_size, dim)
        cin_out = self.cin_dense(cin_out)  # (batch_size, 1)
        # dnn
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)  # (batch_size, 1))
        # output
1➡️    output = tf.nn.sigmoid(linear_out + cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
```

1➡️ We are adding the output of CIN and DNN networks with the direct input features via Linear layer.

[Here](https://nbviewer.org/gist/sparsh-ai/99b99eaecfac4b58ffa60603eb4bbd9c) is the Jupyter notebook of this experiment.

---

## Summary/Footnotes

1. There are 5 matching-class models and 10 ranking-class models.
2. ML-1m dataset is used for training and evaluation of matching models, and Criteo dataset for ranking models.
3. Jupyter notebook (with colab support) is available for each of these 15 experiments.
4. Tensorflow version 2.5 is used for implementation of these models.
5. Code is also available on GitHub [here](https://github.com/sparsh-ai/general-recsys/tree/T021355).