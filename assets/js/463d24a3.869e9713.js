"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7342],{3905:function(t,e,a){a.d(e,{Zo:function(){return p},kt:function(){return f}});var i=a(67294);function n(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function o(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(t);e&&(i=i.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,i)}return a}function r(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?o(Object(a),!0).forEach((function(e){n(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function l(t,e){if(null==t)return{};var a,i,n=function(t,e){if(null==t)return{};var a,i,n={},o=Object.keys(t);for(i=0;i<o.length;i++)a=o[i],e.indexOf(a)>=0||(n[a]=t[a]);return n}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(i=0;i<o.length;i++)a=o[i],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(n[a]=t[a])}return n}var s=i.createContext({}),c=function(t){var e=i.useContext(s),a=e;return t&&(a="function"==typeof t?t(e):r(r({},e),t)),a},p=function(t){var e=c(t.components);return i.createElement(s.Provider,{value:e},t.children)},m={inlineCode:"code",wrapper:function(t){var e=t.children;return i.createElement(i.Fragment,{},e)}},u=i.forwardRef((function(t,e){var a=t.components,n=t.mdxType,o=t.originalType,s=t.parentName,p=l(t,["components","mdxType","originalType","parentName"]),u=c(a),f=n,h=u["".concat(s,".").concat(f)]||u[f]||m[f]||o;return a?i.createElement(h,r(r({ref:e},p),{},{components:a})):i.createElement(h,r({ref:e},p))}));function f(t,e){var a=arguments,n=e&&e.mdxType;if("string"==typeof t||n){var o=a.length,r=new Array(o);r[0]=u;var l={};for(var s in e)hasOwnProperty.call(e,s)&&(l[s]=e[s]);l.originalType=t,l.mdxType="string"==typeof t?t:n,r[1]=l;for(var c=2;c<o;c++)r[c]=a[c];return i.createElement.apply(null,r)}return i.createElement.apply(null,a)}u.displayName="MDXCreateElement"},17883:function(t,e,a){a.r(e),a.d(e,{assets:function(){return p},contentTitle:function(){return s},default:function(){return f},frontMatter:function(){return l},metadata:function(){return c},toc:function(){return m}});var i=a(87462),n=a(63366),o=(a(67294),a(3905)),r=["components"],l={},s="Text Classification",c={unversionedId:"concept-extras/nlp/text-classification",id:"concept-extras/nlp/text-classification",title:"Text Classification",description:"/img/content-concepts-raw-text-classification-img.png",source:"@site/docs/concept-extras/nlp/text-classification.mdx",sourceDirName:"concept-extras/nlp",slug:"/concept-extras/nlp/text-classification",permalink:"/ai/docs/concept-extras/nlp/text-classification",editUrl:"https://github.com/sparsh-ai/ai/docs/concept-extras/nlp/text-classification.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Text Analysis",permalink:"/ai/docs/concept-extras/nlp/text-analysis"},next:{title:"Text Generation",permalink:"/ai/docs/concept-extras/nlp/text-generation"}},p={},m=[{value:"Introduction",id:"introduction",level:2},{value:"Models",id:"models",level:2},{value:"FastText",id:"fasttext",level:3},{value:"XLNet",id:"xlnet",level:3},{value:"BERT",id:"bert",level:3},{value:"TextCNN",id:"textcnn",level:3},{value:"Embedding",id:"embedding",level:3},{value:"Bag-of-words",id:"bag-of-words",level:3},{value:"Process flow",id:"process-flow",level:2},{value:"Use Cases",id:"use-cases",level:2},{value:"Email Classification",id:"email-classification",level:3},{value:"User Sentiment towards Vaccine",id:"user-sentiment-towards-vaccine",level:3},{value:"ServiceNow IT Ticket Classification",id:"servicenow-it-ticket-classification",level:3},{value:"Toxic Comment Classification",id:"toxic-comment-classification",level:3},{value:"Pre-trained Transformer Experiments",id:"pre-trained-transformer-experiments",level:3},{value:"Long Docs Classification",id:"long-docs-classification",level:3},{value:"BERT Sentiment Classification",id:"bert-sentiment-classification",level:3},{value:"Libraries",id:"libraries",level:2},{value:"Common applications",id:"common-applications",level:2},{value:"Links",id:"links",level:2}],u={toc:m};function f(t){var e=t.components,l=(0,n.Z)(t,r);return(0,o.kt)("wrapper",(0,i.Z)({},u,l,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"text-classification"},"Text Classification"),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"/img/content-concepts-raw-text-classification-img.png",src:a(18737).Z,width:"960",height:"720"})),(0,o.kt)("h2",{id:"introduction"},"Introduction"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Definition"),": Text classification is a supervised learning method for learning and predicting the category or the class of a document given its text content. The state-of-the-art methods are based on neural networks of different architectures as well as pre-trained language models or word embeddings."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Applications"),": Spam classification, sentiment analysis, email classification, service ticket classification, question and comment classification"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Scope"),": Muticlass and Multilabel classification"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Tools"),": TorchText, Spacy, NLTK, FastText, HuggingFace, pyss3")),(0,o.kt)("h2",{id:"models"},"Models"),(0,o.kt)("h3",{id:"fasttext"},"FastText"),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},(0,o.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1607.01759"},"Bag of Tricks for Efficient Text Classification. arXiv, 2016."))),(0,o.kt)("p",null,"fastText is an open-source library, developed by the Facebook AI Research lab. Its main focus is on achieving scalable solutions for the tasks of text classification and representation while processing large datasets quickly and accurately."),(0,o.kt)("h3",{id:"xlnet"},"XLNet"),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},(0,o.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1906.08237"},"XLNet: Generalized Autoregressive Pretraining for Language Understanding. arXiv, 2019."))),(0,o.kt)("h3",{id:"bert"},"BERT"),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},(0,o.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1810.04805"},"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv, 2018."))),(0,o.kt)("h3",{id:"textcnn"},"TextCNN"),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},(0,o.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1801.06287"},"What Does a TextCNN Learn?. arXiv, 2018."))),(0,o.kt)("h3",{id:"embedding"},"Embedding"),(0,o.kt)("p",null,"Feature extraction using either any pre-trained embedding models (e.g. Glove, FastText embedding) or custom-trained embedding model (e.g. using Doc2Vec) and then training an ML classifier (e.g. SVM, Logistic regression) on these extracted features."),(0,o.kt)("h3",{id:"bag-of-words"},"Bag-of-words"),(0,o.kt)("p",null,"Feature extraction using methods (CountVectorizer, TF-IDF) and then training an ML classifier (e.g. SVM, Logistic regression) on these extracted features."),(0,o.kt)("h2",{id:"process-flow"},"Process flow"),(0,o.kt)("p",null,"Step 1: Collect text data"),(0,o.kt)("p",null,"Collect via surveys, scrap from the internet or use public datasets"),(0,o.kt)("p",null,"Step 2: Create Labels"),(0,o.kt)("p",null,"In-house labeling or via outsourcing e.g. amazon mechanical turk"),(0,o.kt)("p",null,"Step 3: Data Acquisition"),(0,o.kt)("p",null,"Setup the database connection and fetch the data into python environment"),(0,o.kt)("p",null,"Step 4: Data Exploration"),(0,o.kt)("p",null,"Explore the data, validate it and create preprocessing strategy"),(0,o.kt)("p",null,"Step 5: Data Preparation"),(0,o.kt)("p",null,"Clean the data and make it ready for modeling"),(0,o.kt)("p",null,"Step 6: Model Building"),(0,o.kt)("p",null,"Create the model architecture in python and perform a sanity check"),(0,o.kt)("p",null,"Step 7: Model Training"),(0,o.kt)("p",null,"Start the training process and track the progress and experiments"),(0,o.kt)("p",null,"Step 8: Model Validation"),(0,o.kt)("p",null,"Validate the final set of models and select/assemble the final model"),(0,o.kt)("p",null,"Step 9: UAT Testing"),(0,o.kt)("p",null,"Wrap the model inference engine in API for client testing"),(0,o.kt)("p",null,"Step 10: Deployment"),(0,o.kt)("p",null,"Deploy the model on cloud or edge as per the requirement"),(0,o.kt)("p",null,"Step 11: Documentation"),(0,o.kt)("p",null,"Prepare the documentation and transfer all assets to the client  "),(0,o.kt)("h2",{id:"use-cases"},"Use Cases"),(0,o.kt)("h3",{id:"email-classification"},"Email Classification"),(0,o.kt)("p",null,"The objective is to build an email classifier, trained on 700K emails and 300+ categories. Preprocessing pipeline to handle HTML and template-based content. Ensemble of FastText and BERT classifier. Check out ",(0,o.kt)("a",{parentName:"p",href:"https://www.notion.so/MOFSL-Email-Classification-309EC-7a451afa54d446b7b8f2f656450c6167"},"this")," notion."),(0,o.kt)("h3",{id:"user-sentiment-towards-vaccine"},"User Sentiment towards Vaccine"),(0,o.kt)("p",null,"Based on the tweets of the users, and manually annotated labels (label 0 means against vaccines and label 1 means in-favor of vaccine), build a binary text classifier. 1D-CNN was trained on the training dataset. Check out ",(0,o.kt)("a",{parentName:"p",href:"https://www.notion.so/Twitter-Sentiment-Analysis-18d2d4ca41314b88a18db5c93f9eb2b2"},"this")," notion"),(0,o.kt)("h3",{id:"servicenow-it-ticket-classification"},"ServiceNow IT Ticket Classification"),(0,o.kt)("p",null,'Based on the short description, along with a long description if available for that particular ticket, identify the subject of the incident ticket in order to automatically classify it into a set of pre-defined categories. e.g. If custom wrote "Oracle connection giving error", this ticket type should be labeled as "Database". Check out ',(0,o.kt)("a",{parentName:"p",href:"https://www.notion.so/ESMCafe-IT-Support-Ticket-Management-69965830d39d486194f9a2f1222a81d8"},"this")," notion."),(0,o.kt)("h3",{id:"toxic-comment-classification"},"Toxic Comment Classification"),(0,o.kt)("p",null,"Check out ",(0,o.kt)("a",{parentName:"p",href:"https://www.notion.so/Toxic-Comment-Classification-Challenge-E5207-affac70cc5614f6dad38ab11ac15e1ab"},"this")," notion."),(0,o.kt)("h3",{id:"pre-trained-transformer-experiments"},"Pre-trained Transformer Experiments"),(0,o.kt)("p",null,"Experiment with different types of text classification models that are available in the HuggingFace Transformer library. Wrapped experiment based inference as a streamlit app."),(0,o.kt)("h3",{id:"long-docs-classification"},"Long Docs Classification"),(0,o.kt)("p",null,"Check out this ",(0,o.kt)("a",{parentName:"p",href:"https://colab.research.google.com/github/ArmandDS/bert_for_long_text/blob/master/final_bert_long_docs.ipynb"},"colab"),"."),(0,o.kt)("h3",{id:"bert-sentiment-classification"},"BERT Sentiment Classification"),(0,o.kt)("p",null,"Scrap App reviews data from Android playstore. Fine-tune a BERT model to classify the review as positive, neutral or negative. And then deploy the model as an API using FastAPI. Check out ",(0,o.kt)("a",{parentName:"p",href:"https://www.notion.so/BERT-Sentiment-Analysis-and-FastAPI-Deployment-43175-d0d19b234561445a84517538ad211405"},"this")," notion."),(0,o.kt)("h2",{id:"libraries"},"Libraries"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"pySS3"),(0,o.kt)("li",{parentName:"ul"},"FastText"),(0,o.kt)("li",{parentName:"ul"},"TextBrewer"),(0,o.kt)("li",{parentName:"ul"},"HuggingFace"),(0,o.kt)("li",{parentName:"ul"},"QNLP"),(0,o.kt)("li",{parentName:"ul"},"RMDL"),(0,o.kt)("li",{parentName:"ul"},"Spacy")),(0,o.kt)("h2",{id:"common-applications"},"Common applications"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Sentiment analysis."),(0,o.kt)("li",{parentName:"ul"},"Hate speech detection."),(0,o.kt)("li",{parentName:"ul"},"Document indexing in digital libraries."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Forum data"),": Find out how people feel about various products and features."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Restaurant and movie reviews"),": What are people raving about? What do people hate?"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Social media"),": What is the sentiment about a hashtag, e.g. for a company, politician, etc?"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Call center transcripts"),": Are callers praising or complaining about particular topics?"),(0,o.kt)("li",{parentName:"ul"},"General-purpose categorization in medical, academic, legal, and many other domains.")),(0,o.kt)("h2",{id:"links"},"Links"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html"},"Text Classification - PyTorch Official")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://youtu.be/bEOiYF1a6Ak"},"Building a News Classifier ML App with Streamlit and Python")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/brightmart/text_classification"},"https://github.com/brightmart/text_classification")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://www.tensorflow.org/tutorials/keras/text_classification"},"IMDB Sentiment. Tensorflow.")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/multi-label-text-classification-with-xlnet-b5f5755302df"},"XLNet Fine-tuning. Toxic Comment Multilabel.")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://mccormickml.com/2019/09/19/XLNet-fine-tuning/"},"XLNet Fine-tuning. CoLA.")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://notebooks.quantumstat.com/"},"Classification Demo Notebooks")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/microsoft/nlp-recipes/tree/master/examples/text_classification"},"Microsoft NLP Recipes")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f"},"Report on Text Classification using CNN, RNN & HAN")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"},"Implementing a CNN for Text Classification in TensorFlow")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/prakashpandey9/Text-Classification-Pytorch"},"prakashpandey9/Text-Classification-Pytorch")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://colab.research.google.com/github/georgianpartners/Multimodal-Toolkit/blob/master/notebooks/text_w_tabular_classification.ipynb#scrollTo=QZR8kqmfRssU"},"Google Colaboratory")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://www.tensorflow.org/tutorials/text/classify_text_with_bert"},"Classify text with BERT | TensorFlow Core")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://arxiv.org/pdf/2004.03705v1.pdf"},"Deep Learning Based Text Classification: A Comprehensive Review")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/"},"https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb"},"https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://blog.valohai.com/machine-learning-pipeline-classifying-reddit-posts"},"https://blog.valohai.com/machine-learning-pipeline-classifying-reddit-posts")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://www.kdnuggets.com/2018/03/simple-text-classifier-google-colaboratory.html"},"https://www.kdnuggets.com/2018/03/simple-text-classifier-google-colaboratory.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/AbeerAbuZayed/Quora-Insincere-Questions-Classification"},"https://github.com/AbeerAbuZayed/Quora-Insincere-Questions-Classification")),(0,o.kt)("li",{parentName:"ul"},"Document Classification: 7 pragmatic approaches for small datasets | Neptune's Blog"),(0,o.kt)("li",{parentName:"ul"},"Multi-label Text Classification using BERT - The Mighty Transformer"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb"},"https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb")),(0,o.kt)("li",{parentName:"ul"},"XLNet Fine-Tuning Tutorial with PyTorch"),(0,o.kt)("li",{parentName:"ul"},"Text Classification with XLNet in Action"),(0,o.kt)("li",{parentName:"ul"},"AchintyaX/XLNet_Classification_tuning"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/Dkreitzer/Text_ML_Classification_UMN"},"https://github.com/Dkreitzer/Text_ML_Classification_UMN")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://colab.research.google.com/github/markdaoust/models/blob/basic-text-classification/samples/core/get_started/basic_text_classification.ipynb"},"https://colab.research.google.com/github/markdaoust/models/blob/basic-text-classification/samples/core/get_started/basic_text_classification.ipynb")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/how-to-do-text-binary-classification-with-bert-f1348a25d905"},"https://towardsdatascience.com/how-to-do-text-binary-classification-with-bert-f1348a25d905")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://colab.research.google.com/github/tensorflow/hub/blob/master/docs/tutorials/text_classification_with_tf_hub.ipynb"},"https://colab.research.google.com/github/tensorflow/hub/blob/master/docs/tutorials/text_classification_with_tf_hub.ipynb")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/thomas-chauvet/kaggle_toxic_comment_classification"},"https://github.com/thomas-chauvet/kaggle_toxic_comment_classification")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/scionoftech/TextClassification-Vectorization-DL"},"https://github.com/scionoftech/TextClassification-Vectorization-DL")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/netik1020/Concise-iPython-Notebooks-for-Deep-learning/blob/master/Text_Classification/classification_imdb.ipynb"},"https://github.com/netik1020/Concise-iPython-Notebooks-for-Deep-learning/blob/master/Text_Classification/classification_imdb.ipynb")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/getmrinal/ML-Notebook/tree/master/17.%20textClassificationProject"},"https://github.com/getmrinal/ML-Notebook/tree/master/17. textClassificationProject")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/mpuig/textclassification"},"https://github.com/mpuig/textclassification")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/netik1020/Concise-iPython-Notebooks-for-Deep-learning/blob/master/Text_Classification/self_Attn_on_seperate_fets_of_2embds.ipynb"},"https://github.com/netik1020/Concise-iPython-Notebooks-for-Deep-learning/blob/master/Text_Classification/self_Attn_on_seperate_fets_of_2embds.ipynb")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb"},"https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb")),(0,o.kt)("li",{parentName:"ul"},"sgrvinod/a-PyTorch-Tutorial-to-Text-Classification")))}f.isMDXComponent=!0},18737:function(t,e,a){e.Z=a.p+"assets/images/content-concepts-raw-text-classification-img-eb03940957e68d21cdb50689827b00ad.png"}}]);