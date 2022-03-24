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

<!-- #region id="8n6CYTIja4hq" -->
To start this Jupyter Dash app, please run all the cells below. Then, click on the **temporary** URL at the end of the last cell to open the app.
<!-- #endregion -->

```python id="XSlKBZjYotV1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1608615595960, "user_tz": -330, "elapsed": 17765, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d638c4f-c02c-478b-e0ee-8e62c344b547"
!pip install -q jupyter-dash==0.3.0rc1 dash-bootstrap-components transformers
```

```python id="XTZjvYnkn45v"
import time

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
```

```python id="aN5EHFXSoDTv" colab={"base_uri": "https://localhost:8080/", "height": 279, "referenced_widgets": ["c9fd348466e24635b6ce5c2b8a370fe9", "ac64d13c9658410ebcb1ae6404bf43e2", "fc37cd36c95548d79c6e794401c62b57", "4fc4cd332ece441d91ca2566630e1a0b", "b6f10616b09942d48ce5dd61b641acb4", "bd68484e07c84148a85e06b251a9219b", "5422d88b124c4bf381bb298ecce7f4bf", "47f263b2938743478e73103f03d45c9c", "7d46c739be47496394896456d30013e2", "af4c4c3ae45b450cbf43cd6743dfb209", "d6bda05cf165447c8eab2690bd9a59b8", "b7c6ee82fb864fac8c48a77f095c5b5f", "5266fc223e994e25ab90bc8b7ec146b3", "0ce82a2f9dea4f98b331caccf1b085d2", "24be4ac68030439eb01646eeaf4b52f0", "84aa8dcbbe83461b97965fcacf153ad6", "36bd48bd6a154d849ed330d20eedb766", "fe77427e496047218c64fb6bec560865", "d019c03eba914cca98b9a06f8f1fde47", "2b51c157493249039fafe0ffffc8a6e4", "b481499d5a834138b1525452d3b07a75", "ff5cdc19845e4f0881b7e8a5b87c09a1", "28eb0490be1245fbadf810cc7e26899a", "ae431eb5d1a14059997bdddd87f73397", "2a1d1775689d4608a4946db3b760bc92", "10c72689e25b4e8bb730cc0550476e98", "4bbe5f7504354cc09f3953a5dcba7602", "d9e1b545d52a425eacf79fc5036cb956", "0421246e71fe4bfeb16be56f458b69d4", "e55c028876ca453286400f6ee9aa7327", "eee7d6160aef4dfe8422296aa1fcd569", "7032d53c564a4371a9f41328dc7e8e15", "74b45196c09442d7ada99b1bedf81f35", "e3c03fa0fb6c4f4b84b1fc02fd49bf3d", "b31b2ef37c874437b0057c27bc7c2e1c", "3ed05739e002408886b6d1feb0683bd3", "30788f87e9be453cb6aa0b7125397b3c", "54e4cadeacb54b44896fed3cfc0f0d91", "a7865720d88e4414aaea2654d2910a90", "2330efda21e84ab396b5d1b0d2ff84de"]} executionInfo={"status": "ok", "timestamp": 1608615696721, "user_tz": -330, "elapsed": 118459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c22024a3-da25-49a1-8832-a15dbc3fb3c7"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load Model
pretrained = "sshleifer/distilbart-xsum-12-6"
model = BartForConditionalGeneration.from_pretrained(pretrained)
tokenizer = BartTokenizer.from_pretrained(pretrained)

# Switch to cuda, eval mode, and FP16 for faster inference
if device == "cuda":
    model = model.half()
model.to(device)
model.eval();
```

```python id="Z2EdDQbqpgy1"
# Define app
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Output Length (# Tokens)"),
                dcc.Slider(
                    id="max-length",
                    min=10,
                    max=50,
                    value=30,
                    marks={i: str(i) for i in range(10, 51, 10)},
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Beam Size"),
                dcc.Slider(
                    id="num-beams",
                    min=2,
                    max=6,
                    value=4,
                    marks={i: str(i) for i in [2, 4, 6]},
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Spinner(
                    [
                        dbc.Button("Summarize", id="button-run"),
                        html.Div(id="time-taken"),
                    ]
                )
            ]
        ),
    ],
    body=True,
    style={"height": "275px"},
)


# Define Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Dash Automatic Summarization (with DistilBART)"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    width=5,
                    children=[
                        controls,
                        dbc.Card(
                            body=True,
                            children=[
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Summarized Content"),
                                        dcc.Textarea(
                                            id="summarized-content",
                                            style={
                                                "width": "100%",
                                                "height": "calc(75vh - 275px)",
                                            },
                                        ),
                                    ]
                                )
                            ],
                        ),
                    ],
                ),
                dbc.Col(
                    width=7,
                    children=[
                        dbc.Card(
                            body=True,
                            children=[
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Original Text (Paste here)"),
                                        dcc.Textarea(
                                            id="original-text",
                                            style={"width": "100%", "height": "75vh"},
                                        ),
                                    ]
                                )
                            ],
                        )
                    ],
                ),
            ]
        ),
    ],
)
```

```python id="CPsXC0vnpi0a"
@app.callback(
    [Output("summarized-content", "value"), Output("time-taken", "children")],
    [
        Input("button-run", "n_clicks"),
        Input("max-length", "value"),
        Input("num-beams", "value"),
    ],
    [State("original-text", "value")],
)
def summarize(n_clicks, max_len, num_beams, original_text):
    if original_text is None or original_text == "":
        return "", "Did not run"

    t0 = time.time()

    inputs = tokenizer.batch_encode_plus(
        [original_text], max_length=1024, return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Generate Summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=num_beams,
        max_length=max_len,
        early_stopping=True,
    )
    out = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in summary_ids
    ]

    t1 = time.time()
    time_taken = f"Summarized on {device} in {t1-t0:.2f}s"

    return out[0], time_taken
```

<!-- #region id="HQUSxYkLsbvp" -->
Run the cell below to run your Jupyter Dash app. Click on the **temporary** URL to access the app.
<!-- #endregion -->

```python id="B_6lj6V3pk7I" colab={"base_uri": "https://localhost:8080/", "height": 671} executionInfo={"status": "ok", "timestamp": 1608615835367, "user_tz": -330, "elapsed": 1138, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="56b17e8f-6c8f-4683-f15a-d55bea7e809c"
app.run_server(mode='inline')
```

```python id="fbmR-8QC97Rc"

```
