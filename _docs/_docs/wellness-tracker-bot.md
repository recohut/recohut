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

<!-- #region id="V9SYHOEILWHU" -->
# How to Build a Wellness Tracker Bot
> A bot that logs daily wellness data to a spreadsheet (using the Airtable API), to help the user keep track of their health goals. Connect the assistant to a messaging channel—Twilio—so users can talk to the assistant via text message and Whatsapp.

- toc: true
- badges: false
- comments: true
- categories: [Chatbot, NLP]
- image:
<!-- #endregion -->

<!-- #region id="jBegqyUXVHeE" -->
## Introduction
<!-- #endregion -->

<!-- #region id="hBVy_wQ3VhIM" -->
<!-- #endregion -->

<!-- #region id="R8Al3ihzVMf2" -->
A bot that logs daily wellness data to a spreadsheet (using the Airtable API), to help the user keep track of their health goals. Connect the assistant to a messaging channel—Twilio—so users can talk to the assistant via text message and Whatsapp.

### What you'll learn?

- RASA chatbot with Forms and Custom actions
- Connect with Airtable API to log records in the table database
- Connect with Whatsapp for user interaction

### Why is this important?

- Bots are the future
- More and more users are starting using bots to get recommendations

### How it will work?

- Buid chatbot in RASA
- Test the functionality using command line
- Connect to Twilio
- Connect to Whatsapp via Twilio
- Store responses in Airtable

### Who is this for?

- People who are new in chatbots
- People looking to learn how chatbots work and suggest/assist users

### Important resources

1. [Udemy course](https://www.udemy.com/course/rasa-for-beginners/learn/lecture/20746878#overview)
2. [GitHub code repo](https://github.com/sparsh-ai/chatbots/tree/master/wellnessTracker)

<!---------------------------->

## Command-line chat

Duration: 10
<!-- #endregion -->

<!-- #region id="KMxp8dzkVjZU" -->
<!-- #endregion -->

<!-- #region id="EG2ozuv-Vl_E" -->
<!-- #endregion -->

<!-- #region id="_IwOw4UaVQIo" -->
## Twilio

Duration: 10

<!-- #endregion -->

<!-- #region id="2XhaLqTQVoRp" -->
<!-- #endregion -->

<!-- #region id="RyMbUbJCVR9R" -->
## Whatsapp

Duration: 5

<!-- #endregion -->

<!-- #region id="B8EuSfICVp5Q" -->
<!-- #endregion -->

<!-- #region id="3jkwtEs9VTuy" -->
## Airtable

Duration: 5


<!-- #endregion -->

<!-- #region id="8yJEidSbVrWE" -->
<!-- #endregion -->

<!-- #region id="mlcDoPl3VtMd" -->
<!-- #endregion -->

<!-- #region id="bReOYjUvT__0" -->
## Running the assistant

1. Download the Airtable template and generate an [Airtable API token](https://support.airtable.com/hc/en-us/articles/219046777-How-do-I-get-my-API-key-). You'll also need to locate your Table Name and Base ID, which can be found in the [Airtable API docs](https://airtable.com/api).
2. Make a copy of the `.example-env` and rename it `.env`. Add your Airtable API token, Base ID, and Table Name to the file.
3. Install Rasa Open Source: [https://rasa.com/docs/rasa/user-guide/installation/](https://rasa.com/docs/rasa/user-guide/installation/)
4. Install the action server dependencies: `pip install -r requirements-actions.txt`
5. Train the model: `rasa train`
6. Open a second terminal window and start the action server: `rasa run actions`
7. Return to the first terminal window and start the assistant on the command line: `rasa shell`

## Links and References

1. [Udemy course](https://www.udemy.com/course/rasa-for-beginners/learn/lecture/20746878#overview)
2. [GitHub code repo](https://github.com/sparsh-ai/chatbots/tree/master/wellnessTracker)
<!-- #endregion -->
