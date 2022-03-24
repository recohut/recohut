"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2497],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return p}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),m=l(n),p=o,h=m["".concat(c,".").concat(p)]||m[p]||d[p]||a;return n?r.createElement(h,i(i({ref:t},u),{},{components:n})):r.createElement(h,i({ref:t},u))}));function p(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=m;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var l=2;l<a;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7973:function(e,t,n){n.r(t),n.d(t,{assets:function(){return u},contentTitle:function(){return c},default:function(){return p},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return d}});var r=n(87462),o=n(63366),a=(n(67294),n(3905)),i=["components"],s={title:"Short-video Background Music Recommender",authors:"sparsh",tags:["recsys"]},c=void 0,l={permalink:"/ai/blog/2021/10/01/short-video-background-music-recommender",editUrl:"https://github.com/sparsh-ai/ai/blog/blog/2021-10-01-short-video-background-music-recommender.mdx",source:"@site/blog/2021-10-01-short-video-background-music-recommender.mdx",title:"Short-video Background Music Recommender",description:"Matching micro-videos with suitable background music can help uploaders better convey their contents and emotions, and increase the click-through rate of their uploaded videos. However, manually selecting the background music becomes a painstaking task due to the voluminous and ever-growing pool of candidate music. Therefore, automatically recommending background music to videos becomes an important task.",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"recsys",permalink:"/ai/blog/tags/recsys"}],readingTime:2.17,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],frontMatter:{title:"Short-video Background Music Recommender",authors:"sparsh",tags:["recsys"]},prevItem:{title:"Semantic Similarity",permalink:"/ai/blog/2021/10/01/semantic-similarity"},nextItem:{title:"The progression of analytics in enterprises",permalink:"/ai/blog/2021/10/01/the-progression-of-analytics-in-enterprises"}},u={authorsImageUrls:[void 0]},d=[],m={toc:d};function p(e){var t=e.components,s=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},m,s,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"Matching micro-videos with suitable background music can help uploaders better convey their contents and emotions, and increase the click-through rate of their uploaded videos. However, manually selecting the background music becomes a painstaking task due to the voluminous and ever-growing pool of candidate music. Therefore, automatically recommending background music to videos becomes an important task."),(0,a.kt)("p",null,"In ",(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/2107.07268.pdf"},"this")," paper, Zhu et. al. shared their approach to solve this task. They first collected ~3,000 background music from popular TikTok videos and also ~150,000 video clips that used some kind of background music. They named this dataset ",(0,a.kt)("inlineCode",{parentName:"p"},"TT-150K"),"."),(0,a.kt)("p",null,(0,a.kt)("img",{loading:"lazy",alt:"An exemplar subset of videos and their matched background music in the established TT-150k dataset",src:n(26296).Z,width:"902",height:"688"})),(0,a.kt)("p",null,"An exemplar subset of videos and their matched background music in the established TT-150k dataset"),(0,a.kt)("p",null,"After building the dataset, they worked on modeling and proposed the following architecture:"),(0,a.kt)("p",null,(0,a.kt)("img",{loading:"lazy",alt:"Proposed CMVAE (Cross-modal Variational Auto-encoder) framework",src:n(75409).Z,width:"1041",height:"532"})),(0,a.kt)("p",null,"Proposed CMVAE (Cross-modal Variational Auto-encoder) framework"),(0,a.kt)("p",null,"The goal is to represent videos (",(0,a.kt)("inlineCode",{parentName:"p"},"users")," in recsys terminology) and music (",(0,a.kt)("inlineCode",{parentName:"p"},"items"),") in a shared latent space. To achieve this, CMVAE use pre-trained models to extract features from unstructured data - ",(0,a.kt)("inlineCode",{parentName:"p"},"vggish")," model for audio2vec, ",(0,a.kt)("inlineCode",{parentName:"p"},"resnet")," for video2vec and ",(0,a.kt)("inlineCode",{parentName:"p"},"bert-multilingual")," for text2vec.  Text and video vectors are then fused using product-of-expert approach. "),(0,a.kt)("p",null,"It uses the reconstruction power of variational autoencoders to 1) reconstruct video from music latent vector and, 2) reconstruct music from video latent vector. In layman terms, we are training a neural network that will try to guess the video activity just by listening background music, and also try to guess the background music just by seeing the video activities. "),(0,a.kt)("p",null,"The joint training objective is $\\mathcal{L}",(0,a.kt)("em",{parentName:"p"},"{(z_m,z_v)} = \\beta \\cdot\\mathcal{L}"),"{cross","_","recon} - \\mathcal{L}",(0,a.kt)("em",{parentName:"p"},"{KL} + \\gamma \\cdot \\mathcal{L}"),"{matching}$, where $\\beta$ and $\\gamma$ control the weight of the cross reconstruction loss and the matching loss, respectively."),(0,a.kt)("p",null,"After training the model, they compared the model's performance with existing baselines and the results are as follows:"),(0,a.kt)("p",null,(0,a.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-short-video-background-music-recommender-untitled-2.png",src:n(79667).Z,width:"1240",height:"450"})),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Conclusion"),": I don't make short videos myself but can easily imagine the difficulty in finding the right background music. If I have to do this task manually, I will try out 5-6 videos and select one that I like. But here, I will be assuming that my audience would also like this music. Moreover, feedback is not actionable because it will create kind of an implicit sub-conscious effect (because when I see a video, I mostly judge it at overall level and rarely notice that background music is the problem). So, this kind of recommender system will definitely help me in selecting a better background music. Excited to see this feature soon in TikTok, Youtube Shorts and other similar services."))}p.isMDXComponent=!0},75409:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-short-video-background-music-recommender-untitled-1-81086e3ba53f8bc9648682e6f1b367ed.png"},79667:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-short-video-background-music-recommender-untitled-2-0002224cfc03822d344479f3766cd1ec.png"},26296:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-short-video-background-music-recommender-untitled-8e4334cad8c9ec206ebdba19b1163b61.png"}}]);