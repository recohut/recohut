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

<!-- #region id="Xb5oBDWUd86l" -->
# MoE in Tensorflow
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jboou3AEhu1-" executionInfo={"status": "ok", "timestamp": 1629646959362, "user_tz": -330, "elapsed": 642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="733bc9f0-97af-46ec-bda9-02a572bb18e3"
%tensorflow_version 1.x
```

```python id="yM_YX_Sbhxjf"
import numpy as np
import tensorflow as tf
```

<!-- #region id="wGfJ5ds5iepc" -->
## Multi-task
<!-- #endregion -->

```python id="RMyoWKz4iDwH"
def task_network(inputs,
                 hidden_units,
                 hidden_activation=tf.nn.relu,
                 output_activation=tf.nn.sigmoid,
                 hidden_dropout=None,
                 initializer=None):

    x = inputs
    for units in hidden_units:
        x = tf.layers.dense(x,
                            units,
                            activation=hidden_activation,
                            kernel_initializer=initializer)

        if hidden_dropout is not None:
            x = tf.layers.dropout(x, rate=hidden_dropout)

    outputs = tf.layers.dense(x, 1, kernel_initializer=initializer)

    if output_activation is not None:
        outputs = output_activation(outputs)
    return outputs
```

```python id="gCao5IgOh2_o"
def multi_task(inputs,
               num_tasks,
               task_hidden_units,
               task_output_activations,
               **kwargs):

    outputs = []

    for i in range(num_tasks):

        task_inputs = inputs[i] if isinstance(inputs, list) else inputs

        output = task_network(task_inputs,
                              task_hidden_units,
                              output_activation=task_output_activations[i],
                              **kwargs)
        outputs.append(output)

    return outputs
```

<!-- #region id="uJnNhkcmih1v" -->
## Mixture-of-experts
<!-- #endregion -->

```python id="HPp49VXdiOdG"
def _synthetic_data(num_examples, example_dim=100, c=0.3, p=0.8, m=5):

    mu1 = np.random.normal(size=example_dim)
    mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(example_dim))

    mu2 = np.random.normal(size=example_dim)
    mu2 -= mu2.dot(mu1) * mu1
    mu2 /= np.linalg.norm(mu2)

    w1 = c * mu1
    w2 = c * (p * mu1 + np.sqrt(1. - p ** 2) * mu2)

    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)

    examples = np.random.normal(size=(num_examples, example_dim))

    w1x = np.matmul(examples, w1)
    w2x = np.matmul(examples, w2)

    sin1, sin2 = 0., 0.
    for i in range(m):
        sin1 += np.sin(alpha[i] * w1x + beta[i])
        sin2 += np.sin(alpha[i] * w2x + beta[i])

    y1 = w1x + sin1 + np.random.normal(size=num_examples, scale=0.01)
    y2 = w2x + sin2 + np.random.normal(size=num_examples, scale=0.01)

    return examples.astype(np.float32), (y1.astype(np.float32), y2.astype(np.float32))
```

```python id="XRQNSwe3iSra"
def synthetic_data_input_fn(num_examples, epochs=1, batch_size=256, buffer_size=256, **kwargs):

    synthetic_data = _synthetic_data(num_examples, **kwargs)

    dataset = tf.data.Dataset.from_tensor_slices(synthetic_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(buffer_size)

    return dataset
```

```python id="sDt_NuQHiUc0"
def gating_network(inputs, num_experts, expert_index=None):
    """
    Gating network: y = SoftMax(W * inputs)
    :param inputs: tf.Tensor
    :param num_experts: Int > 0, number of expert networks.
    :param expert_index: Int, index of expert network.
    :return: tf.Tensor
    """

    x = tf.layers.dense(inputs,
                        units=num_experts,
                        use_bias=False,
                        name="expert{}_gate".format(expert_index))

    return tf.nn.softmax(x)
```

```python id="cRwWfO95iXkB"
def one_gate(inputs,
             num_tasks,
             num_experts,
             task_hidden_units,
             task_output_activations,
             expert_hidden_units,
             expert_hidden_activation=tf.nn.relu,
             task_hidden_activation=tf.nn.relu,
             task_initializer=None,
             task_dropout=None):

    experts_gate = gating_network(inputs, num_experts)

    experts_outputs = []
    for i in range(num_experts):
        x = inputs
        for j, units in enumerate(expert_hidden_units):
            x = tf.layers.dense(x, units, activation=expert_hidden_activation, name="expert{}_dense{}".format(i, j))
        experts_outputs.append(x)

    experts_outputs = tf.stack(experts_outputs, axis=1)
    experts_selector = tf.expand_dims(experts_gate, axis=1)

    outputs = tf.linalg.matmul(experts_selector, experts_outputs)

    multi_task_inputs = tf.squeeze(outputs)

    return multi_task(multi_task_inputs,
                      num_tasks,
                      task_hidden_units,
                      task_output_activations,
                      hidden_activation=task_hidden_activation,
                      hidden_dropout=task_dropout,
                      initializer=task_initializer)
```

```python id="6A7TUMWYhtUY"
def multi_gate(inputs,
               num_tasks,
               num_experts,
               task_hidden_units,
               task_output_activations,
               expert_hidden_units,
               expert_hidden_activation=tf.nn.relu,
               task_hidden_activation=tf.nn.relu,
               task_initializer=None,
               task_dropout=None):

    experts_outputs = []
    for i in range(num_experts):
        x = inputs
        for j, units in enumerate(expert_hidden_units[:-1]):
            x = tf.layers.dense(x, units, activation=expert_hidden_activation, name="expert{}_dense{}".format(i, j))

        x = tf.layers.dense(x, expert_hidden_units[-1], name="expert{}_out".format(i))

        experts_outputs.append(x)

    experts_outputs = tf.stack(experts_outputs, axis=1)

    outputs = []
    for i in range(num_experts):
        expert_gate = gating_network(inputs, num_experts, expert_index=i)
        expert_selector = tf.expand_dims(expert_gate, axis=1)

        output = tf.linalg.matmul(expert_selector, experts_outputs)

        outputs.append(tf.squeeze(output))

    return multi_task(outputs,
                      num_tasks,
                      task_hidden_units,
                      task_output_activations,
                      hidden_activation=task_hidden_activation,
                      hidden_dropout=task_dropout,
                      initializer=task_initializer)
```

<!-- #region id="-D70ghIDjDj4" -->
## Testing
<!-- #endregion -->

```python id="WVUffDuwjUWD"
from absl.testing import parameterized
import sys
```

```python id="yZ2Ec1f6jL4q"
tf.disable_eager_execution()
sys.dont_write_bytecode = True
sys.argv = sys.argv[:1]
old_sysexit = sys.exit
tf.logging.set_verbosity(tf.logging.INFO)
```

```python id="YgMk_BVUizdn"
class TestMixtureOfExperts(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(42, 256, 1024, 2021)
    def test_synthetic_data(self, random_seed):
        np.random.seed(random_seed)
        _, (y1, y2) = _synthetic_data(1000, p=0.8)
        cor = np.corrcoef(y1, y2)
        print(cor)

    def test_one_gate(self):

        num_examples = 1000
        example_dim = 128

        inputs = tf.random.normal(shape=(num_examples, example_dim))

        outputs = one_gate(inputs,
                       num_tasks=2,
                       num_experts=3,
                       task_hidden_units=[10, 5],
                       task_output_activations=[None, None],
                       expert_hidden_units=[64, 32],
                       expert_hidden_activation=tf.nn.relu,
                       task_hidden_activation=tf.nn.relu,
                       task_initializer=None,
                       task_dropout=None)

    def test_multi_gate(self):

        num_examples = 1000
        example_dim = 128

        inputs = tf.random.normal(shape=(num_examples, example_dim))

        outputs = multi_gate(inputs,
                       num_tasks=2,
                       num_experts=3,
                       task_hidden_units=[10, 5],
                       task_output_activations=[None, None],
                       expert_hidden_units=[64, 32],
                       expert_hidden_activation=tf.nn.relu,
                       task_hidden_activation=tf.nn.relu,
                       task_initializer=None,
                       task_dropout=None)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RQwf3VdXj_5C" executionInfo={"status": "ok", "timestamp": 1629647970944, "user_tz": -330, "elapsed": 1439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2015e995-3775-4a4e-e99c-1469d4014661"
try:
    sys.exit = lambda *args: None
    tf.test.main()
finally:
    sys.exit = old_sysexit
```
