"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5467],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return h}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),c=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=c(e.components);return r.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),d=c(n),h=a,f=d["".concat(l,".").concat(h)]||d[h]||p[h]||o;return n?r.createElement(f,i(i({ref:t},u),{},{components:n})):r.createElement(f,i({ref:t},u))}));function h(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=d;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var c=2;c<o;c++)i[c]=n[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},8653:function(e,t,n){n.r(t),n.d(t,{assets:function(){return u},contentTitle:function(){return l},default:function(){return h},frontMatter:function(){return s},metadata:function(){return c},toc:function(){return p}});var r=n(87462),a=n(63366),o=(n(67294),n(3905)),i=["components"],s={},l="Text Style Transfer",c={unversionedId:"concept-extras/nlp/text-style-transfer",id:"concept-extras/nlp/text-style-transfer",title:"Text Style Transfer",description:"How to adapt the text to different situations, audiences and purposes by making some changes? The style of the text usually includes many aspects such as morphology, grammar, emotion, complexity, fluency, tense, tone and so on.",source:"@site/docs/concept-extras/nlp/text-style-transfer.mdx",sourceDirName:"concept-extras/nlp",slug:"/concept-extras/nlp/text-style-transfer",permalink:"/ai/docs/concept-extras/nlp/text-style-transfer",editUrl:"https://github.com/sparsh-ai/ai/docs/concept-extras/nlp/text-style-transfer.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Text Similarity",permalink:"/ai/docs/concept-extras/nlp/text-similarity"},next:{title:"Text Summarization",permalink:"/ai/docs/concept-extras/nlp/text-summarization"}},u={},p=[{value:"Literature review",id:"literature-review",level:3},{value:"Papers",id:"papers",level:3}],d={toc:p};function h(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"text-style-transfer"},"Text Style Transfer"),(0,o.kt)("p",null,"How to adapt the text to different situations, audiences and purposes by making some changes? The style of the text usually includes many aspects such as morphology, grammar, emotion, complexity, fluency, tense, tone and so on."),(0,o.kt)("p",null,"With this technology, the sentences can be converted according to the required style, such as the emotion from positive to negative, writing style from normal to shakepeare style and tone style from formal to informal. "),(0,o.kt)("p",null,"Challenges:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"Lack of a large number of parallel corpora"),(0,o.kt)("li",{parentName:"ol"},"Uncertainty of the evaluation indicators")),(0,o.kt)("h3",{id:"literature-review"},"Literature review"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Jhamtani (2017) used parallel data to train a seq2seq model to convert style of text to Shakespeare style. The data set used in this study is a line-by-line modern interpretation of 16 of 36 Shakespeare\u2019s plays, of which the training set contains 18,395 sentences in 14 plays, and the last play, \u201cRomeo and Juliet,\u201d which contains 1,462 sentences, is the test set.  In order to solve the problem of insufficient parallel data, the author combined with additional text to pre-train an external dictionary representation (embeddings) from Shakespeare English to modern English and explained that the shared dictionary table on the source language side and the target language side is beneficial to improve performance. After experiments, the method achieved a BLEU score of 31+, which is about 6 points higher than the strongest benchmark MOSES."),(0,o.kt)("li",{parentName:"ul"},"Carlson (2018) pointed out that the style of an article that can be perceived is composed of many characteristics, including the length of the sentence, the active or passive voice, the level of vocabulary, the degree of tone, and the formality of the language. He collected a large, undeveloped parallel text data set - the Bible translation corpus. The study shows that its data set is conducive to the model's generalization of style learning, because each version of the Bible reflects A unique style. Each version exceeds 31,000 sessions and can generate more than 1.5 million unique parallel training data."),(0,o.kt)("li",{parentName:"ul"},"Harnessing (2019) mainly focused on the study of the formality of text (Formality), and proposed a method to transfer the style from informal text to formal text. The study shows that when the current parallel corpus is very small, using a large neural network model that has been pre-trained on a large-scale corpus and has learned general language knowledge will be effective, and the introduction of rules effectively reduces the complexity of the data."),(0,o.kt)("li",{parentName:"ul"},"Shang (2019) pointed out that the method based on the standard S2S (sequence-to-sequence) proposed latent space cross prediction method (Cross Projection in Latent Space) realized the function of style conversion between different style data sets. By inputting from the Encoder module of Style A and outputting from the Decoder module of Style B through the cross prediction function, the text style transfer is realized."),(0,o.kt)("li",{parentName:"ul"},"Keith (2018) proposed Zero-Shot Style Transfer in which the zero-sample style transfer is converted into a single machine translation problem, and based on this, a recurrent neural network (RNN) model based on S2S (sequence-to-sequence) architecture is created.")),(0,o.kt)("h3",{id:"papers"},"Papers"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://arxiv.org/ftp/arxiv/papers/2005/2005.02914.pdf"},"Survey Paper (2020)"))))}h.isMDXComponent=!0}}]);