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

<!-- #region id="bTIzHosJUFYe" -->
# Shared Bottom in Tensorflow
<!-- #endregion -->

<!-- #region id="SnGjzIeCUHCW" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jboou3AEhu1-" executionInfo={"status": "ok", "timestamp": 1629648279360, "user_tz": -330, "elapsed": 676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d6f06bc3-47b9-41f8-c1c9-b3457564beb7"
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
## Shared bottom strategy
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

```python id="pWIZROtdm_RG"
def _dense(x, units, activation=None, dropout=None, name=None):
    weights = tf.get_variable("w{}".format(name),
                              shape=(x.shape[-1], units),
                              dtype=tf.float32)
    bias = tf.get_variable("b{}".format(name),
                           shape=(units,),
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, bias)

    if dropout is not None:
        x = tf.nn.dropout(x, rate=dropout)

    if activation is not None:
        x = activation(x)

    return x
```

```python id="qBfFcvvTnB4B"
def shared_bottom(x: tf.Tensor,
                  num_tasks: int,
                  bottom_units: list,
                  task_units: list,
                  task_output_activation: list,
                  bottom_initializer: tf.Tensor = tf.truncated_normal_initializer(),
                  bottom_activation=tf.nn.relu,
                  bottom_dropout: float = None,
                  task_initializer: tf.Tensor = tf.truncated_normal_initializer(),
                  task_dropout: float = None,
                  task_activation=tf.nn.relu):

    with tf.variable_scope("bottom", initializer=bottom_initializer):
        for i, units in enumerate(bottom_units[:-1]):
            x = _dense(x, units, activation=bottom_activation, dropout=bottom_dropout, name=i)

        bottom_out = _dense(x, bottom_units[-1], name="out")

    outputs = []

    for task_idx in range(num_tasks):
        x = bottom_out
        with tf.variable_scope("task{}".format(task_idx), initializer=task_initializer):
            for i, units in enumerate(task_units):
                x = _dense(x, units, activation=task_activation, dropout=task_dropout, name=i)

            task_out = _dense(x, 1, name="out")

            output_activation = task_output_activation[task_idx]
            if output_activation == "sigmoid":
                task_out = tf.nn.sigmoid(task_out)

            outputs.append(task_out)

    return outputs
```

```python id="OZUN1IIrnEKN"
def shared_bottom_v2(x: tf.Tensor,
                     num_tasks: int,
                     bottom_units: list,
                     task_hidden_units: list,
                     task_output_activations: list,
                     bottom_initializer: tf.Tensor = None,
                     bottom_activation=tf.nn.relu,
                     bottom_dropout: float = None,
                     task_initializer: tf.Tensor = None,
                     task_dropout: float = None,
                     task_activation=tf.nn.relu):

    for i, units in enumerate(bottom_units[:-1]):
        x = tf.layers.dense(x,
                            units,
                            activation=bottom_activation,
                            kernel_initializer=bottom_initializer,
                            name="bottom_dense{}".format(i))

        if bottom_dropout is not None:
            x = tf.layers.dropout(x, rate=bottom_dropout, name="bottom_dropout{}".format(i))

    bottom_out = tf.layers.dense(x, bottom_units[-1], kernel_initializer=bottom_initializer, name="bottom_out")

    outputs = multi_task(bottom_out,
                         num_tasks,
                         task_hidden_units,
                         task_output_activations,
                         hidden_activation=task_activation,
                         hidden_dropout=task_dropout,
                         initializer=task_initializer)

    return outputs
```

```python id="EIISACIpnJx9"
def model_fn(features, labels, mode, params):

    outputs = shared_bottom_v2(features["inputs"],
                               num_tasks=params.get("num_tasks"),
                               bottom_units=params.get("bottom_units"),
                               task_hidden_units=params.get("task_units"),
                               task_output_activations=params.get("task_output_activations"))
    predictions = {
        "y{}".format(i): y
        for i, y in enumerate(outputs)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    task_losses = params.get("task_losses")

    total_loss = tf.Variable(0., name="total_loss")
    losses = []
    metrics = {}

    for i, (t, y) in enumerate(predictions.items()):

        y = tf.squeeze(y)

        if task_losses[i] == 'log_loss':
            loss = tf.losses.log_loss(labels=labels[i], predictions=y)
            auc_op = tf.metrics.auc(labels=labels, predictions=y, name='auc_op')
            tf.summary.scalar("auc", auc_op[-1])
            metrics["auc"] = auc_op

        elif task_losses[i] == 'mse':
            loss = tf.losses.mean_squared_error(labels=labels[i], predictions=y)

        else:
            loss = tf.losses.mean_squared_error(labels=labels[i], predictions=y)

        losses.append(loss)
        total_loss = total_loss + loss
        metrics["loss_{}".format(t)] = loss
        tf.summary.scalar("loss_{}".format(t), loss)

    tf.summary.scalar("total_loss", total_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

    train_op = tf.group(*[
        optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        for loss in losses
    ])

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
```

```python id="ZduyuZ_xm2KQ"
def shared_bottom_estimator(model_dir, inter_op, intra_op, params):

    config_proto = tf.ConfigProto(device_count={'GPU': 0},
                                  inter_op_parallelism_threads=inter_op,
                                  intra_op_parallelism_threads=intra_op)

    run_config = tf.estimator.RunConfig().replace(
        tf_random_seed=42,
        keep_checkpoint_max=10,
        save_checkpoints_steps=200,
        log_step_count_steps=10,
        session_config=config_proto)

    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=model_dir,
                                  params=params,
                                  config=run_config)
```

<!-- #region id="-D70ghIDjDj4" -->
## Testing
<!-- #endregion -->

```python id="WVUffDuwjUWD"
from absl.testing import parameterized
import sys
import tempfile
```

```python id="yZ2Ec1f6jL4q"
tf.disable_eager_execution()
sys.dont_write_bytecode = True
sys.argv = sys.argv[:1]
old_sysexit = sys.exit
tf.logging.set_verbosity(tf.logging.INFO)
```

```python id="YgMk_BVUizdn"
class TestSharedBottom(tf.test.TestCase, parameterized.TestCase):

    def test_shared_bottom(self):
        num_examples = 100
        example_dim = 10
        x = tf.random_normal(shape=(num_examples, example_dim))

        with self.session() as sess:
            y = shared_bottom(x, 2, [32, 16], [10, 5], [None, None])
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(y)

    @parameterized.parameters(
        {
            "num_tasks": 2,
            "bottom_units": [32, 16],
            "task_units": [10, 5],
            "task_output_activations": [None, None],
            "task_losses": ["mse", "mse"],
            "lr": 0.001
        }
    )
    def test_shared_bottom_estimator(self, **params):

        def _map_fn(x, y):
            return {"inputs": x}, y

        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = shared_bottom_estimator(temp_dir, 8, 8, params)
            estimator.train(input_fn=lambda: synthetic_data_input_fn(1000).map(map_func=_map_fn))

            features = {"inputs": tf.placeholder(tf.float32, (None, 100), name="inputs")}

            serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

            estimator.export_saved_model(temp_dir + "/saved_model", serving_input_receiver_fn)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RQwf3VdXj_5C" executionInfo={"status": "ok", "timestamp": 1629648725951, "user_tz": -330, "elapsed": 7193, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a5c3a64-6416-451e-ca31-d792dec2cdd0"
try:
    sys.exit = lambda *args: None
    tf.test.main()
finally:
    sys.exit = old_sysexit
```
