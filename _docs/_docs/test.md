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

```python id="8tIziHn5H6ye" colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"status": "ok", "timestamp": 1627066722001, "user_tz": -330, "elapsed": 961, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03043dee-9934-4559-9b0c-55869889698e"
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()
```

<!-- #region id="aWLRvtPqIrH0" -->
<!-- #endregion -->
