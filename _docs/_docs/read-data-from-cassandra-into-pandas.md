---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: 'Python 3.7.1 64-bit (''env'': conda)'
    name: python371jvsc74a57bd0e09bc63938c7c12a1d261af93fef4c49c4bc4ec4b59849b02ca92b0abe12c74c
---

<!-- #region id="Ezeu9nViz3yv" -->
# Read Cassandra as DataFrame
> Short guide on how to read data from Cassandra into pandas dataframe format

- toc: true
- badges: true
- comments: true
- categories: [Cassandra]
- image:
<!-- #endregion -->

```python id="fYlxGtNWzxrz"
import os
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import columns
from datetime import datetime
import pandas as pd
```

```python id="PUfn2tsyzxr4"
import os
from datetime import datetime

from cassandra.cqlengine.management import sync_table
from cassandra.policies import TokenAwarePolicy
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import (
    Cluster,
    DCAwareRoundRobinPolicy
)
from cassandra.cqlengine.connection import (
    register_connection,
    set_default_connection
)
```

```python id="J_ZJSfPWzxr6"
CASSANDRA_USERNAME='cassandra'
CASSANDRA_PASSWORD='cassandra'
CASSANDRA_HOST='127.0.0.1'
CASSANDRA_PORT=9042
```

```python id="vVnyNK7Szxr7" outputId="be54579a-11b7-4496-be81-a3de8cb9e543"
session = None
cluster = None

auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)
cluster = Cluster([CASSANDRA_HOST],
load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
port=CASSANDRA_PORT,
auth_provider=auth_provider,
executor_threads=2,
protocol_version=4,
)           
```

```python id="tz58KhpFzxsB"
session = cluster.connect()
register_connection(str(session), session=session)
set_default_connection(str(session))
```

```python id="m2aSYu5ZzxsC" outputId="f603f5ce-e087-4f93-9209-aa4a27d43823"
rows = session.execute('select * from demo.click_stream;')
df = pd.DataFrame(list(rows))
df.head()
```

```python id="0i4RM91czxsD" outputId="fee58e18-97e9-4a4c-c72f-165fedd8431b"
df.info()
```

```python id="2ZE4ekoEzxsE" outputId="7c32c54d-deb2-48ad-b91d-8fe725d65f01"
df.describe()
```

```python id="UVfC-exMzxsF" outputId="854da3c4-6219-4074-8c15-29d21017fbad"
df.item_id.value_counts()
```

```python id="YhMWrOQAzxsG"
df.to_pickle('../recommender/data/logs_test_020521_1.p')
```
