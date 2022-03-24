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

```python colab={"base_uri": "https://localhost:8080/"} id="1epRrWTsk8PS" executionInfo={"status": "ok", "timestamp": 1619516997673, "user_tz": -330, "elapsed": 16197, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22c0d1be-52ff-40f7-a93d-065381ced2d1"
!pip install dash dash-html-components dash-core-components dash-table jupyter-dash
```

```python id="U8GVDqdeuYJa" executionInfo={"status": "ok", "timestamp": 1619517134921, "user_tz": -330, "elapsed": 1075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
```

```python id="AygtvkbYlLte" executionInfo={"status": "ok", "timestamp": 1619516999534, "user_tz": -330, "elapsed": 18011, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Load Data
df = px.data.tips()

# Build App
app = JupyterDash(__name__)
app.layout = html.Div([
    html.H1("RecoDash"),
    dcc.Graph(id='graph'),
    html.Label([
        "colorscale",
        dcc.Dropdown(
            id='colorscale-dropdown', clearable=False,
            value='plasma', options=[
                {'label': c, 'value': c}
                for c in px.colors.named_colorscales()
            ])
    ]),
])

# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("colorscale-dropdown", "value")]
)
def update_figure(colorscale):
    return px.scatter(
        df, x="total_bill", y="tip", color="size",
        color_continuous_scale=colorscale,
        render_mode="webgl", title="Tips"
    )
```

```python colab={"base_uri": "https://localhost:8080/", "height": 671} id="XRQ6WMuAlwO1" executionInfo={"status": "ok", "timestamp": 1619517020941, "user_tz": -330, "elapsed": 992, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e412074-42ab-4fbf-afc0-9fd9801dbe41"
# Run app and display result inline in the notebook
# !kill -9 $(lsof -t -i:8081)
app.run_server(mode='inline', width='80%', port=8081)
```

```python id="jCH3ue7IzDur"

```

```python id="VfgvDIxozDsV"

```

```python id="0zZJOCBDzDph"

```

```python id="qirj0NOHzDmS"

```
