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
    language: python
    name: python3
---

<!-- #region pycharm={"name": "#%% md\n"} id="qUU7wy-brl_H" -->
# Hugging Face
> Leverage State-of-the-art NLP models with only one line of code

- toc: true
- badges: true
- comments: true
- categories: [nlp, huggingface]
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} id="-HLOHXuArl_L" -->
Newly introduced in transformers v2.3.0, **pipelines** provides a high-level, easy to use,
API for doing inference over a variety of downstream-tasks, including: 

- ***Sentence Classification _(Sentiment Analysis)_***: Indicate if the overall sentence is either positive or negative, i.e. *binary classification task* or *logitic regression task*.
- ***Token Classification (Named Entity Recognition, Part-of-Speech tagging)***: For each sub-entities _(*tokens*)_ in the input, assign them a label, i.e. classification task.
- ***Question-Answering***: Provided a tuple (`question`, `context`) the model should find the span of text in `content` answering the `question`.
- ***Mask-Filling***: Suggests possible word(s) to fill the masked input with respect to the provided `context`.
- ***Summarization***: Summarizes the ``input`` article to a shorter article.
- ***Translation***: Translates the input from a language to another language.
- ***Feature Extraction***: Maps the input to a higher, multi-dimensional space learned from the data.

Pipelines encapsulate the overall process of every NLP process:
 
 1. *Tokenization*: Split the initial input into multiple sub-entities with ... properties (i.e. tokens).
 2. *Inference*: Maps every tokens into a more meaningful representation. 
 3. *Decoding*: Use the above representation to generate and/or extract the final output for the underlying task.

The overall API is exposed to the end-user through the `pipeline()` method with the following 
structure:

```python
from transformers import pipeline

# Using default model and tokenizer for the task
pipeline("<task-name>")

# Using a user-specified model
pipeline("<task-name>", model="<model_name>")

# Using custom model/tokenizer as str
pipeline('<task-name>', model='<model name>', tokenizer='<tokenizer_name>')
```
<!-- #endregion -->

```python pycharm={"name": "#%% code\n"} id="4maAknWNrl_N" outputId="467e3cc8-a069-47da-8029-86e4142c7dde" colab={"base_uri": "https://localhost:8080/", "height": 102}
!pip install -q transformers
```

```python pycharm={"is_executing": false, "name": "#%% code \n"} id="uKaqzCh6rl_V"
from __future__ import print_function
import ipywidgets as widgets
from transformers import pipeline
```

<!-- #region pycharm={"name": "#%% md\n"} id="uDPZ42Uerl_b" -->
## 1. Sentence Classification - Sentiment Analysis
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="AMRXHQw9rl_d" outputId="a7a10851-b71e-4553-9afc-04066120410d" colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["4bab5df43b3c46caadf48e264344ab42", "9b426c68631f4bb288e2ca79aad9f9d9", "6902104f7ec143519fb1a6ab9363d4a0", "c133fb34fe2a4aba8a6b233671af8b04", "e3f72d443a74414ca62c2b848d34b125", "5462b581976e47048642aa6bc12435bd", "ad84da685cf44abb90d17d9d2e023b48", "a246f9eea2d7440cb979e728741d2e32"]}
nlp_sentence_classif = pipeline('sentiment-analysis')
nlp_sentence_classif('Such a nice weather outside !')
```

<!-- #region pycharm={"name": "#%% md\n"} id="RY8aUJTvrl_k" -->
## 2. Token Classification - Named Entity Recognition
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="B3BDRX_Krl_n" outputId="a6b90b11-a272-4ecb-960d-4c682551b399" colab={"base_uri": "https://localhost:8080/", "height": 185, "referenced_widgets": ["451464c936444ba5a652b46c1b4f9931", "279291efd2c14a9eb2c3b98efbf152ad", "b6e1a2e57f4948a39283f1370352612c", "9d4941ebdfa64978b47232f6e5908d97", "1006cc0fab1e4139bb7b135486261c92", "691c0bae60364890ab74934261207d4d", "405afa5bb8b840d8bc0850e02f593ce4", "78c718e3d5fa4cb892217260bea6d540"]}
nlp_token_class = pipeline('ner')
nlp_token_class('Hugging Face is a French company based in New-York.')
```

<!-- #region id="qIvUFEVarl_s" -->
## 3. Question Answering
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="ND_8LzQKrl_u" outputId="c59ae695-c465-4de6-fa6e-181d8f1a3992" colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["7d66a4534c164d2f9493fc0467abebbd", "7a15588f85b14f2b93e32b4c0442fa1b", "213567d815894ca08041f6d682ced3c9", "ee6c95e700e64d0a9ebec2c1545dd083", "3e556abf5c4a4ee69d52366fd59471b2", "876b2eba73fa46a6a941d2e3a8a975ad", "cd64e3f20b23483daa79712bde6622ea", "67cbaa1f55d24e62ad6b022af36bca56"]}
nlp_qa = pipeline('question-answering')
nlp_qa(context='Hugging Face is a French company based in New-York.', question='Where is based Hugging Face ?')
```

<!-- #region id="9W_CnP5Zrl_2" -->
## 4. Text Generation - Mask Filling
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="zpJQ2HXNrl_4" outputId="3fb62e7a-25a6-4b06-ced8-51eb8aa6bf33" colab={"base_uri": "https://localhost:8080/", "height": 321, "referenced_widgets": ["58669943d3064f309436157270544c08", "3eff293c2b554d85aefaea863e29b678", "d0b9925f3dde46008bf186cf5ef7722d", "427e07ce24a442af84ddc71f9463fdff", "1eb2fa080ec44f8c8d5f6f52900277ab", "23377596349e40a89ea57c8558660073", "a35703cc8ff44e93a8c0eb413caddc40", "9df7014c99b343f3b178fa020ff56010"]}
nlp_fill = pipeline('fill-mask')
nlp_fill('Hugging Face is a French company based in ' + nlp_fill.tokenizer.mask_token)
```

<!-- #region id="Fbs9t1KvrzDy" -->
## 5. Summarization

Summarization is currently supported by `Bart` and `T5`.
<!-- #endregion -->

```python id="8BaOgzi1u1Yc" outputId="2168e437-cfba-4247-a38c-07f02f555c6e" colab={"base_uri": "https://localhost:8080/", "height": 88}
TEXT_TO_SUMMARIZE = """ 
New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. 
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. 
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. 
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. 
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 
2010 marriage license application, according to court documents. 
Prosecutors said the marriages were part of an immigration scam. 
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. 
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective 
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. 
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. 
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages. 
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. 
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s 
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. 
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. 
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

summarizer = pipeline('summarization')
summarizer(TEXT_TO_SUMMARIZE)
```

<!-- #region id="u5JA6IJsr-G0" -->
## 6. Translation

Translation is currently supported by `T5` for the language mappings English-to-French (`translation_en_to_fr`), English-to-German (`translation_en_to_de`) and English-to-Romanian (`translation_en_to_ro`).
<!-- #endregion -->

```python id="8FwayP4nwV3Z" outputId="66956816-c924-4718-fe58-cabef7d51974" colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["57e8c36594d043c581c766b434037771", "82760185d5c14a808cbf6639b589f249", "f2a1b430594b4736879cdff4ec532098", "c81338551e60474fab9e9950fe5df294", "98563b405bd043a9a301a43909e43157", "8c0e1b7fb6ac4ee7bbbaf6020b40cc77", "ad78042ee71a41fd989e4b4ce9d2e3c1", "40c8d2617f3d4c84b923b140456fa5da"]}
# English to French
translator = pipeline('translation_en_to_fr')
translator("HuggingFace is a French company that is based in New York City. HuggingFace's mission is to solve NLP one commit at a time")
```

```python id="ra0-WfznwoIW" outputId="278a3d5f-cc42-40bc-a9db-c92ec5a3a2f0" colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["311a65b811964ebfa2c064eb348b3ce9", "5a2032c44d0e4f8cbaf512e6c29214cd", "54d1ff55e0094a4fa2b62ecdfb428328", "2e45f2d7d65246ecb8d6e666d026ac13", "e05c0ec3b49e4d4990a943d428532fb0", "39721262fc1e4456966d92fabe0f54ea", "4486f8a2efc34b9aab3864eb5ad2ba48", "d6228324f3444aa6bd1323d65ae4ff75"]}
# English to German
translator = pipeline('translation_en_to_de')
translator("The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods.")
```

<!-- #region id="qPUpg0M8hCtB" -->
## 7. Text Generation

Text generation is currently supported by GPT-2, OpenAi-GPT, TransfoXL, XLNet, CTRL and Reformer.
<!-- #endregion -->

```python id="5pKfxTxohXuZ" outputId="8705f6b4-2413-4ac6-f72d-e5ecce160662" colab={"base_uri": "https://localhost:8080/", "height": 120, "referenced_widgets": ["3c86415352574190b71e1fe5a15d36f1", "dd2c9dd935754cf2802233053554c21c", "8ae3be32d9c845e59fdb1c47884d48aa", "4dea0031f3554752ad5aad01fe516a60", "1efb96d931a446de92f1930b973ae846", "6a4f5aab5ba949fd860b5a35bba7db9c", "4b02b2e964ad49af9f7ce7023131ceb8", "0ae8a68c3668401da8d8a6d5ec9cac8f"]}
text_generator = pipeline("text-generation")
text_generator("Today is a beautiful day and I will")
```

<!-- #region id="Utmldmetrl_9" -->
## 8. Projection - Features Extraction 
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="O4SjR1QQrl__" outputId="2ce966d5-7a89-4488-d48f-626d1c2a8222" colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["fd44cf6ab17e4b768b2e1d5cb8ce5af9", "b8c0ea31578d4eaaa69251d0004fd8c6", "2015cd9c1da9467290ecd9019af231eb", "17bacdaee55b43e8977c4dfe4f7245bb", "879ef9e1a0e94f3d96ed56fb4bae64b8", "7ab70324d42647acac5020b387955caf", "31d97ecf78fa412c99e6659196d82828", "c6be5d48ec3c4c799d1445607e5f1ac6"]}
import numpy as np
nlp_features = pipeline('feature-extraction')
output = nlp_features('Hugging Face is a French company based in Paris')
np.array(output).shape   # (Samples, Tokens, Vector Size)

```

<!-- #region pycharm={"name": "#%% md\n"} id="02j8km8YrmAE" -->
Alright ! Now you have a nice picture of what is possible through transformers' pipelines, and there is more
to come in future releases. 

In the meantime, you can try the different pipelines with your own inputs
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%% code\n"} id="yFlBPQHtrmAH" outputId="03cc3207-a7e8-49fd-904a-63a7a1d0eb7a" colab={"base_uri": "https://localhost:8080/", "height": 116, "referenced_widgets": ["0bd407b4975f49c3827aede14c59501c", "3f5406df699e44f5b60678c1c13500f5", "17768469581445b68246ed308ce69326", "74cbcbae5cac4f12abf080a38390f05c", "62b10ca525cc4ac68f3a006434eb7416", "211109537fbe4e60b89a238c89db1346"]}
task = widgets.Dropdown(
    options=['sentiment-analysis', 'ner', 'fill_mask'],
    value='ner',
    description='Task:',
    disabled=False
)

input = widgets.Text(
    value='',
    placeholder='Enter something',
    description='Your input:',
    disabled=False
)

def forward(_):
    if len(input.value) > 0: 
        if task.value == 'ner':
            output = nlp_token_class(input.value)
        elif task.value == 'sentiment-analysis':
            output = nlp_sentence_classif(input.value)
        else:
            if input.value.find('<mask>') == -1:
                output = nlp_fill(input.value + ' <mask>')
            else:
                output = nlp_fill(input.value)                
        print(output)

input.on_submit(forward)
display(task, input)
```

```python pycharm={"is_executing": false, "name": "#%% Question Answering\n"} id="GCoKbBTYrmAN" outputId="57c3a647-160a-4b3a-e852-e7a1daf1294a" colab={"base_uri": "https://localhost:8080/", "height": 143, "referenced_widgets": ["d79946ac16ea4855a0bbe2ca2a4d4bf5", "ab5774ac19f84ab18ddf09a63433df00", "a02164204f0f43668bc36a907e720af7", "3b12aec414b14221ad2a11dfd975faa0", "d305ba1662e3466c93ab5cca7ebf8f33", "879f7a3747ad455d810c7a29918648ee"]}
context = widgets.Textarea(
    value='Einstein is famous for the general theory of relativity',
    placeholder='Enter something',
    description='Context:',
    disabled=False
)

query = widgets.Text(
    value='Why is Einstein famous for ?',
    placeholder='Enter something',
    description='Question:',
    disabled=False
)

def forward(_):
    if len(context.value) > 0 and len(query.value) > 0: 
        output = nlp_qa(question=query.value, context=context.value)            
        print(output)

query.on_submit(forward)
display(context, query)
```
