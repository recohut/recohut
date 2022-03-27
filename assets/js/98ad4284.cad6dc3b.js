"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9040],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return f}});var a=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=a.createContext({}),s=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=s(e.components);return a.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,r=e.originalType,l=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),d=s(n),f=i,m=d["".concat(l,".").concat(f)]||d[f]||u[f]||r;return n?a.createElement(m,o(o({ref:t},p),{},{components:n})):a.createElement(m,o({ref:t},p))}));function f(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var r=n.length,o=new Array(r);o[0]=d;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:i,o[1]=c;for(var s=2;s<r;s++)o[s]=n[s];return a.createElement.apply(null,o)}return a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},41500:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return l},default:function(){return f},frontMatter:function(){return c},metadata:function(){return s},toc:function(){return u}});var a=n(87462),i=n(63366),r=(n(67294),n(3905)),o=["components"],c={},l="Facial Analytics",s={unversionedId:"concept-extras/vision/facial-analytics",id:"concept-extras/vision/facial-analytics",title:"Facial Analytics",description:"/img/content-concepts-raw-computer-vision-facial-analytics-img.png",source:"@site/docs/concept-extras/vision/facial-analytics.mdx",sourceDirName:"concept-extras/vision",slug:"/concept-extras/vision/facial-analytics",permalink:"/recohut/docs/concept-extras/vision/facial-analytics",editUrl:"https://github.com/sparsh-ai/recohut/docs/concept-extras/vision/facial-analytics.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Walmart Model Selection",permalink:"/recohut/docs/concept-extras/success-stories/walmart-model-selection"},next:{title:"Image Segmentation",permalink:"/recohut/docs/concept-extras/vision/image-segmentation"}},p={},u=[{value:"Introduction",id:"introduction",level:2},{value:"Models",id:"models",level:2},{value:"FaceNet",id:"facenet",level:3},{value:"RetinaFace",id:"retinaface",level:3},{value:"FER+",id:"fer",level:3},{value:"Process flow",id:"process-flow",level:2},{value:"Use Cases",id:"use-cases",level:2},{value:"Automatic Attendance System via Webcam",id:"automatic-attendance-system-via-webcam",level:3},{value:"Detectron2 Fine-tuning for face detection",id:"detectron2-fine-tuning-for-face-detection",level:3}],d={toc:u};function f(e){var t=e.components,c=(0,i.Z)(e,o);return(0,r.kt)("wrapper",(0,a.Z)({},d,c,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"facial-analytics"},"Facial Analytics"),(0,r.kt)("p",null,(0,r.kt)("img",{loading:"lazy",alt:"/img/content-concepts-raw-computer-vision-facial-analytics-img.png",src:n(56562).Z,width:"960",height:"720"})),(0,r.kt)("h2",{id:"introduction"},"Introduction"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Definition:")," Analyze the facial features like age, gender, emotion, and identity."),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Applications:")," Identity verification, emotion detection"),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Scope:")," Human faces only, Real-time"),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Tools:")," OpenCV, dlib")),(0,r.kt)("h2",{id:"models"},"Models"),(0,r.kt)("h3",{id:"facenet"},"FaceNet"),(0,r.kt)("p",null,(0,r.kt)("em",{parentName:"p"},(0,r.kt)("a",{parentName:"em",href:"https://openaccess.thecvf.com/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf"},"FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR, 2015."))),(0,r.kt)("h3",{id:"retinaface"},"RetinaFace"),(0,r.kt)("p",null,(0,r.kt)("em",{parentName:"p"},(0,r.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1905.00641v2"},"RetinaFace: Single-stage Dense Face Localisation in the Wild. arXiv, 2019."))),(0,r.kt)("h3",{id:"fer"},"FER+"),(0,r.kt)("p",null,(0,r.kt)("em",{parentName:"p"},(0,r.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1608.01041v2"},"Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution. arXiv, 2016."))),(0,r.kt)("h2",{id:"process-flow"},"Process flow"),(0,r.kt)("p",null,"Step 1: Collect Images"),(0,r.kt)("p",null,"Capture via camera, scrap from the internet or use public datasets"),(0,r.kt)("p",null,"Step 2: Create Labels"),(0,r.kt)("p",null,"Compile a metadata table containing a unique id (preferably the same as the image name) for each face id."),(0,r.kt)("p",null,"Step 3: Data Preparation"),(0,r.kt)("p",null,"Setup the database connection and fetch the data into the environment. Explore the data, validate it, and create a preprocessing strategy. Clean the data and make it ready for modeling"),(0,r.kt)("p",null,"Step 4: Model Building"),(0,r.kt)("p",null,"Create the model architecture in python and perform a sanity check. Start the training process and track the progress and experiments. Validate the final set of models and select/assemble the final model"),(0,r.kt)("p",null,"Step 5: UAT Testing"),(0,r.kt)("p",null,"Wrap the model inference engine in API for client testing"),(0,r.kt)("p",null,"Step 6: Deployment"),(0,r.kt)("p",null,"Deploy the model on cloud or edge as per the requirement"),(0,r.kt)("p",null,"Step 7: Documentation"),(0,r.kt)("p",null,"Prepare the documentation and transfer all assets to the client  "),(0,r.kt)("h2",{id:"use-cases"},"Use Cases"),(0,r.kt)("h3",{id:"automatic-attendance-system-via-webcam"},"Automatic Attendance System via Webcam"),(0,r.kt)("p",null,"We use Face Recognition library and OpenCV to create a real-time webcam-based attendance system that will automatically recognizes the face and log an attendance into the excel sheet. Check out ",(0,r.kt)("a",{parentName:"p",href:"https://www.notion.so/Face-Recognition-based-Automated-Attendance-System-dfb6f70527994ea4be11caf69b054350"},"this")," notion."),(0,r.kt)("h3",{id:"detectron2-fine-tuning-for-face-detection"},"Detectron2 Fine-tuning for face detection"),(0,r.kt)("p",null,"Fine-tuned detectron2 on human face dataset to detect the faces in images and videos. Check out ",(0,r.kt)("a",{parentName:"p",href:"https://www.notion.so/Detectron-2-D281D-bb7f769860fa434d923feef3a99f9cbb"},"this")," notion."))}f.isMDXComponent=!0},56562:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-computer-vision-facial-analytics-img-ed8bce5e22ffa937180af34383b50152.png"}}]);