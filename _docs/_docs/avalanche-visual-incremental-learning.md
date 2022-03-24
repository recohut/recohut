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

<!-- #region id="Id8YkAA6ral_" -->
# Avalanche Visual Incremental Learning
<!-- #endregion -->

<!-- #region id="hL9OJNlgp38o" -->
## Setup
<!-- #endregion -->

```python id="S5Kf2GJtp06U"
!pip install git+https://github.com/ContinualAI/avalanche.git
```

```python id="9gGo8L6Mp1c4"
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from types import SimpleNamespace
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import StrategyPlugin
```

<!-- #region id="R33HzLiWp6Ic" -->
## A comprehensive example
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["c3922ac9ae2f4457a9dfc75b6e6dbfb3", "d030b7d51d8a4b2b9a5429eec73ea9b9", "b934588fc39846d9993d88603a6a5ed4", "bc21ee433af74d62a771cbdd9bbacdc8", "6f1e2064b03746fcaad01e2173205ef9", "c29caf0398064d90904c7b74620500f5", "4d632c0000924e43a239aa677a6099ff", "db9bc562070c445792f48d4ec4dea951", "606d13e119ac4f3080134ea4b8672677", "986fab2f7e5e4a6e956fc925e430130a", "319a6797792940ee9929f7d912b20c5d", "d23681ef255f41b0af7922fa794da80c", "fe9f4b35016e43c2928aa92f98d1bacf", "857cd7bccb2641b995cfd24b20cc9df2", "b6fbc52623ea4d3985fbc268fbaccf0a", "254793c1924b47279aa8e49dbb809460", "0f54ca8747004ceba63e89ebc0332696", "f8631d811fc44b01b480a65373181f53", "4b179f93425b49329ae123b671cbdfae", "fe515221a2e545338e4b69174003ae50", "d27f2ae6e7c340b3996c6f05540b83da", "a5caa909fe3e4fb1828d0f2794e6e7ce", "a4301fb5c7ac48a3ae474b21e609fb16", "94459652afb1480285306d9bdeaf9bd1", "7b50321571404a7b8dd1ce06bf85dfbd", "e19719f07b154d22b5277a1fee73ad41", "93306de6c4834f5485aa11ef2172cb05", "527104b4f7df470aaca3a54538483802", "87bbe9aa92d846b19bcfd02012ed465b", "7cc2f0ba548f4d13986d732c3c3c7676", "ce732bab9b1a49d0b9f94870d452dc90", "d9c78e4d45aa45d5a9dbcb01df5e0ede", "743cf65f52b8463099d1047dd41383c4", "ab28a2655e004b5a8fe5866959f4313a", "7199654a7fc14c628601a67da086bcbf", "1b2f9222b80e4cad97ecd19f45f06f37", "0926719c39324b0799d815334e7b4b86", "98e4369eb3bd4c3c90805fd54c7a31ed", "cf57f33031d6490caeae476d3786a6e1", "ee72a2bcabfd423db2ed097b313907a8", "00e2161d622d4f98be8bd60beff27fdb", "ffab7da57bc747fd8fd2ce5331258ea9", "0e094f2ae2464e3498da3ed8afd76046", "2063812629ad4a4ca3dac8901141c6c2"]} id="lzS_8A5TqAgi" executionInfo={"status": "ok", "timestamp": 1635252780344, "user_tz": -330, "elapsed": 36164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="799fe8de-daa6-42f3-840f-cf6681e7579b"
scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))
```

<!-- #region pycharm={"name": "#%% md\n"} id="22JZv6zTqK2b" -->
## Dataloading, Memory Buffers, and Replay

Avalanche provides several components that help you to balance data loading and implement rehearsal strategies.

**Dataloaders** are used to provide balancing between groups (e.g. tasks/classes/experiences). This is especially useful when you have unbalanced data.

**Buffers** are used to store data from the previous experiences. They are dynamic datasets with a fixed maximum size, and they can be updated with new data continuously.

Finally, **Replay** strategies implement rehearsal by using Avalanche's plugin system. Most rehearsal strategies use a custom dataloader to balance the buffer with the current experience and a buffer that is updated for each experience.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} id="wxR7kLyYqK2g" -->
### Dataloaders
Avalanche dataloaders are simple iterators, located under `avalanche.benchmarks.utils.data_loader`. Their interface is equivalent to pytorch's dataloaders. For example, `GroupBalancedDataLoader` takes a sequence of datasets and iterates over them by providing balanced mini-batches, where the number of samples is split equally among groups. Internally, it instantiate a `DataLoader` for each separate group. More specialized dataloaders exist such as `TaskBalancedDataLoader`.

All the dataloaders accept keyword arguments (`**kwargs`) that are passed directly to the dataloaders for each group.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="R6VKo17IqK2i" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252814039, "user_tz": -330, "elapsed": 1781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dbeb0635-0f9a-417b-8857-ff2313261acf"
benchmark = SplitMNIST(5, return_task_id=True)

dl = GroupBalancedDataLoader([exp.dataset for exp in benchmark.train_stream], batch_size=4)
for x, y, t in dl:
    print(t.tolist())
    break
```

<!-- #region pycharm={"name": "#%% md\n"} id="pDBr8nrrqK2l" -->
### Memory Buffers
Memory buffers store data up to a maximum capacity, and they implement policies to select which data to store and which the to remove when the buffer is full. They are available in the module `avalanche.training.storage_policy`. The base class is the `ExemplarsBuffer`, which implements two methods:
- `update(strategy)` - given the strategy's state it updates the buffer (using the data in `strategy.experience.dataset`).
- `resize(strategy, new_size)` - updates the maximum size and updates the buffer accordingly.

The data can be access using the attribute `buffer`.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="VievPzGAqK2n" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252837631, "user_tz": -330, "elapsed": 1635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2c84b9f3-9ff1-41a5-e447-8795b2d59d16"
benchmark = SplitMNIST(5, return_task_id=False)
storage_p = ReservoirSamplingBuffer(max_size=30)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
```

<!-- #region pycharm={"name": "#%% md\n"} id="0x6SPZKwqK2o" -->
At first, the buffer is empty. We can update it with data from a new experience.

Notice that we use a `SimpleNamespace` because we want to use the buffer standalone, without instantiating an Avalanche strategy. Reservoir sampling requires only the `experience` from the strategy's state.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="FzoX7ApzqK2p" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252839890, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0388a2a8-cea4-4aeb-f812-ad307842aa59"
for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets}\n")
```

<!-- #region pycharm={"name": "#%% md\n"} id="ehtHBo9iqK2r" -->
Notice after each update some samples are substituted with new data. Reservoir sampling select these samples randomly.

Avalanche offers many more storage policies. For example, `ParametricBuffer` is a buffer split into several groups according to the `groupby` parameters (`None`, 'class', 'task', 'experience'), and according to an optional `ExemplarsSelectionStrategy` (random selection is the default choice).
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="T1Gh25nxqK2t" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252868668, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="254a672c-7260-43ad-8ee8-41a77e1330fc"
storage_p = ParametricBuffer(
    max_size=30,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets}\n")
```

<!-- #region pycharm={"name": "#%% md\n"} id="PfPvkBeaqK2w" -->
The advantage of using grouping buffers is that you get a balanced rehearsal buffer. You can even access the groups separately with the `buffer_groups` attribute. Combined with balanced dataloaders, you can ensure that the mini-batches stay balanced during training.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="INhxgBToqK2x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252870657, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ebd42cb7-3a49-461a-8068-8b1e16f50e72"
for k, v in storage_p.buffer_groups.items():
    print(f"(group {k}) -> size {len(v.buffer)}")
```

```python pycharm={"name": "#%%\n"} id="-lmtS6lrqK2y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635252872548, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b71e773f-e687-4ece-a44f-aabf66ced2e4"
datas = [v.buffer for v in storage_p.buffer_groups.values()]
dl = GroupBalancedDataLoader(datas)

for x, y, t in dl:
    print(y.tolist())
    break
```

<!-- #region pycharm={"name": "#%% md\n"} id="5RGMxaRBqK2y" -->
### Replay Plugins

Avalanche's strategy plugins can be used to update the rehearsal buffer and set the dataloader. This allows to easily implement replay strategies:
<!-- #endregion -->

```python id="KZ3KQgPSq-Fy"
from avalanche.training import Naive
```

```python pycharm={"name": "#%%\n"} id="cCY_t3qWqK2z"
class CustomReplay(StrategyPlugin):
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)

```

<!-- #region pycharm={"name": "#%% md\n"} id="JTWCwEFyqK20" -->
And of course, we can use the plugin to train our continual model
<!-- #endregion -->

```python pycharm={"name": "#%%\n"} id="9p96kLvWqK20" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635253017083, "user_tz": -330, "elapsed": 55469, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="485c574b-7445-4a5f-aa23-eb991160827f"
scenario = SplitMNIST(5)
model = SimpleMLP(num_classes=scenario.n_classes)
storage_p = ParametricBuffer(
    max_size=500,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

# choose some metrics and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loggers=[interactive_logger])

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(model, torch.optim.Adam(model.parameters(), lr=0.001),
                    CrossEntropyLoss(),
                    train_mb_size=100, train_epochs=1, eval_mb_size=100,
                    plugins=[CustomReplay(storage_p)],
                    evaluator=eval_plugin
                    )

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience ", experience.current_experience)
    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))
```
