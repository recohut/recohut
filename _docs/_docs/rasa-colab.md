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

<!-- #region id="fBP36hRuXl76" -->
# Running RASA Chatbot in Colab
<!-- #endregion -->

<!-- #region id="svM2GDQSXo_L" -->
## RASA Main App
<!-- #endregion -->

```python id="sIgLEkDmJdNm"
!pip install rasa
```

```python id="666GQ_ygcHfV" executionInfo={"status": "ok", "timestamp": 1601356535200, "user_tz": -330, "elapsed": 3263, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4664a4c9-6166-4c02-f0ba-e1ac00f6c9af" colab={"base_uri": "https://localhost:8080/", "height": 138}
!git clone https://github.com/RasaHQ/rasa-for-beginners.git
```

```python id="2CEXelioLMDh" executionInfo={"status": "ok", "timestamp": 1601356574862, "user_tz": -330, "elapsed": 1201, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b72617f6-5007-4cfd-fbc0-b2a2ff830ddb" colab={"base_uri": "https://localhost:8080/", "height": 34}
cd rasa-for-beginners
```

```python id="8EzYfNENSisg" executionInfo={"status": "ok", "timestamp": 1601356730327, "user_tz": -330, "elapsed": 145945, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d43e45c7-f37b-424e-9188-a8be20676a49" colab={"base_uri": "https://localhost:8080/", "height": 817}
!rasa train
```

```python id="CJirF9xSTJ5Z" executionInfo={"status": "ok", "timestamp": 1601357081304, "user_tz": -330, "elapsed": 22369, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="486b79f6-1ca5-4ae5-d678-b8068647d533" colab={"base_uri": "https://localhost:8080/", "height": 156}
# !pip install colab_ssh
from colab_ssh import launch_ssh
launch_ssh('<token>', '<pass>')
```

```python id="CCHqKj0VTXSf" executionInfo={"status": "ok", "timestamp": 1601365699519, "user_tz": -330, "elapsed": 8944, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a7150ab-aa64-47c0-f979-20888b19462b" colab={"base_uri": "https://localhost:8080/", "height": 52}
!pip install -q pyngrok
from pyngrok import ngrok
!ngrok authtoken <token>
```

```python id="y2qhNPDT1EXe" executionInfo={"status": "ok", "timestamp": 1601371988402, "user_tz": -330, "elapsed": 2455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="614a8644-02bb-48a9-a30d-7ddad315dc96" colab={"base_uri": "https://localhost:8080/", "height": 34}
ngrok.kill()
public_url = ngrok.connect(port='80'); public_url
```

```python id="qdBxCylu1ZEY"
!rasa run -p 80
```

```python id="yLzYBzWf31-n"
# Twilio Endpoint
# http://c41fc2bf36a0.ngrok.io/webhooks/twilio/webhook
```

```python id="UlUzJhhGPZTI"
!mv /content/*.py /content/rasa-for-beginners/files
!mv /content/*.ipynb /content/rasa-for-beginners/files
```

```python id="h1HbDOtLRFms" executionInfo={"status": "ok", "timestamp": 1601372977683, "user_tz": -330, "elapsed": 1626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5c0fad2d-e924-418a-896d-3bb1da273eb4" colab={"base_uri": "https://localhost:8080/", "height": 34}
%cd /content
```

```python id="hAn-zB4yRKbY"
# !mkdir -p /content/chatbots/wellnessTracker
# !mv  -v /content/rasa-for-beginners/* /content/chatbots/wellnessTracker/
# !rm -r /content/chatbots/wellnessTracker/.git
```

<!-- #region id="uuQPV-sfX5lA" -->
## RASA Action server
<!-- #endregion -->

```python id="PEVk_s57RAZf"
# !pip install rasa
# !pip3 install python-dotenv
```

```python id="kGHY6Do_RXC0"
!git clone https://github.com/RasaHQ/rasa-for-beginners.git
```

```python id="yVo-1GLmROoC" executionInfo={"status": "ok", "timestamp": 1601363760012, "user_tz": -330, "elapsed": 3749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d8829330-8621-4fc9-b009-78a07811b1b0" colab={"base_uri": "https://localhost:8080/", "height": 34}
%%writefile actions.py

from typing import Any, Text, Dict, List, Union
from dotenv import load_dotenv

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction

import requests
import json
import os

load_dotenv()

airtable_api_key='<api-key>' #os.getenv("AIRTABLE_API_KEY")
base_id='<id>' #os.getenv("BASE_ID")
table_name='TABLEX' #os.getenv("TABLE_NAME")

def create_health_log(confirm_exercise, exercise, sleep, diet, stress, goal):
    request_url=f"https://api.airtable.com/v0/{base_id}/{table_name}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {airtable_api_key}",
    }  
    data = {
        "fields": {
            "Exercised?": confirm_exercise,
            "Type of exercise": exercise,
            "Amount of sleep": sleep,
            "Stress": stress,
            "Diet": diet,
            "Goal": goal,
        }
    }
    try:
        response = requests.post(
            request_url, headers=headers, data=json.dumps(data)
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    
    return response
    print(response.status_code)

class HealthForm(FormAction):

    def name(self):
        return "health_form"

    @staticmethod
    def required_slots(tracker):

        if tracker.get_slot('confirm_exercise') == True:
            return ["confirm_exercise", "exercise", "sleep",
             "diet", "stress", "goal"]
        else:
            return ["confirm_exercise", "sleep",
             "diet", "stress", "goal"]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "confirm_exercise": [
                self.from_intent(intent="affirm", value=True),
                self.from_intent(intent="deny", value=False),
                self.from_intent(intent="inform", value=True),
            ],
            "sleep": [
                self.from_entity(entity="sleep"),
                self.from_intent(intent="deny", value="None"),
            ],
            "diet": [
                self.from_text(intent="inform"),
                self.from_text(intent="affirm"),
                self.from_text(intent="deny"),
            ],
            "goal": [
                self.from_text(intent="inform"),
            ],
        }

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:

        confirm_exercise = tracker.get_slot("confirm_exercise")
        exercise = tracker.get_slot("exercise")
        sleep = tracker.get_slot("sleep")
        stress = tracker.get_slot("stress")
        diet = tracker.get_slot("diet")
        goal = tracker.get_slot("goal")

        response = create_health_log(
                confirm_exercise=confirm_exercise,
                exercise=exercise,
                sleep=sleep,
                stress=stress,
                diet=diet,
                goal=goal
            )

        dispatcher.utter_message("Thanks, your answers have been recorded!")
        return []
```

```python id="l8HpM34cQ94Q" executionInfo={"status": "ok", "timestamp": 1601363765937, "user_tz": -330, "elapsed": 2654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29a0ff19-65c4-4da9-c0f9-35d587e03f96" colab={"base_uri": "https://localhost:8080/", "height": 34}
# !pip install -q pyngrok
# from pyngrok import ngrok
# !ngrok authtoken <token>
ngrok.kill()
public_url = ngrok.connect(port='5055'); public_url
```

```python id="CJC5-67zRw2l" executionInfo={"status": "ok", "timestamp": 1601373399992, "user_tz": -330, "elapsed": 7284826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9cb32b23-f095-4a9e-88a3-5e4422d92e55" colab={"base_uri": "https://localhost:8080/", "height": 138}
!rasa run actions
```
