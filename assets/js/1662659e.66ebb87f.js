"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9956],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return f}});var o=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,o,i=function(e,t){if(null==e)return{};var n,o,i={},a=Object.keys(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=o.createContext({}),s=function(e){var t=o.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},p=function(e){var t=s(e.components);return o.createElement(l.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},u=o.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,l=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),u=s(n),f=i,m=u["".concat(l,".").concat(f)]||u[f]||d[f]||a;return n?o.createElement(m,r(r({ref:t},p),{},{components:n})):o.createElement(m,r({ref:t},p))}));function f(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,r=new Array(a);r[0]=u;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:i,r[1]=c;for(var s=2;s<a;s++)r[s]=n[s];return o.createElement.apply(null,r)}return o.createElement.apply(null,n)}u.displayName="MDXCreateElement"},92694:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return l},default:function(){return f},frontMatter:function(){return c},metadata:function(){return s},toc:function(){return d}});var o=n(87462),i=n(63366),a=(n(67294),n(3905)),r=["components"],c={},l="Object Detection",s={unversionedId:"concept-extras/vision/object-detection",id:"concept-extras/vision/object-detection",title:"Object Detection",description:"/img/content-concepts-raw-computer-vision-object-detection-slide29.png",source:"@site/docs/concept-extras/vision/object-detection.mdx",sourceDirName:"concept-extras/vision",slug:"/concept-extras/vision/object-detection",permalink:"/ai/docs/concept-extras/vision/object-detection",editUrl:"https://github.com/sparsh-ai/ai/docs/concept-extras/vision/object-detection.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Image Similarity",permalink:"/ai/docs/concept-extras/vision/image-similarity"},next:{title:"Object Tracking",permalink:"/ai/docs/concept-extras/vision/object-tracking"}},p={},d=[{value:"Introduction",id:"introduction",level:2},{value:"Models",id:"models",level:2},{value:"Faster R-CNN",id:"faster-r-cnn",level:3},{value:"SSD (Single Shot Detector)",id:"ssd-single-shot-detector",level:3},{value:"YOLO (You Only Look Once)",id:"yolo-you-only-look-once",level:3},{value:"EfficientDet",id:"efficientdet",level:3},{value:"Process flow",id:"process-flow",level:2},{value:"Use Cases",id:"use-cases",level:2},{value:"Automatic License Plate Recognition",id:"automatic-license-plate-recognition",level:3},{value:"Object Detection App",id:"object-detection-app",level:3},{value:"Logo Detector",id:"logo-detector",level:3},{value:"TF Object Detection API Experiments",id:"tf-object-detection-api-experiments",level:3},{value:"Pre-trained Inference Experiments",id:"pre-trained-inference-experiments",level:3},{value:"Object Detection App",id:"object-detection-app-1",level:3},{value:"Real-time Object Detector in OpenCV",id:"real-time-object-detector-in-opencv",level:3},{value:"EfficientDet Fine-tuning",id:"efficientdet-fine-tuning",level:3},{value:"YOLO4 Fine-tuning",id:"yolo4-fine-tuning",level:3},{value:"Detectron2 Fine-tuning",id:"detectron2-fine-tuning",level:3}],u={toc:d};function f(e){var t=e.components,c=(0,i.Z)(e,r);return(0,a.kt)("wrapper",(0,o.Z)({},u,c,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"object-detection"},"Object Detection"),(0,a.kt)("p",null,(0,a.kt)("img",{loading:"lazy",alt:"/img/content-concepts-raw-computer-vision-object-detection-slide29.png",src:n(92879).Z,width:"960",height:"720"})),(0,a.kt)("h2",{id:"introduction"},"Introduction"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Definition:")," Object detection is a computer vision technique that allows us to identify and locate objects in an image or video."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Applications:")," Crowd counting, Self-driving cars, Video surveillance, Face detection, Anomaly detection"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Scope:")," Detect objects in images and videos, 2-dimensional bounding boxes, Real-time"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"Tools:")," Detectron2, TF Object Detection API, OpenCV, TFHub, TorchVision")),(0,a.kt)("h2",{id:"models"},"Models"),(0,a.kt)("h3",{id:"faster-r-cnn"},"Faster R-CNN"),(0,a.kt)("p",null,(0,a.kt)("em",{parentName:"p"},(0,a.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1506.01497"},"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. arXiv, 2016."))),(0,a.kt)("h3",{id:"ssd-single-shot-detector"},"SSD (Single Shot Detector)"),(0,a.kt)("p",null,(0,a.kt)("em",{parentName:"p"},(0,a.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1512.02325"},"SSD: Single Shot MultiBox Detector. CVPR, 2016."))),(0,a.kt)("h3",{id:"yolo-you-only-look-once"},"YOLO (You Only Look Once)"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1804.02767"},(0,a.kt)("em",{parentName:"a"},"YOLOv3: An Incremental Improvement. arXiv, 2018."))),(0,a.kt)("h3",{id:"efficientdet"},"EfficientDet"),(0,a.kt)("p",null,(0,a.kt)("em",{parentName:"p"},(0,a.kt)("a",{parentName:"em",href:"https://arxiv.org/abs/1911.09070"},"EfficientDet: Scalable and Efficient Object Detection. CVPR, 2020."))),(0,a.kt)("p",null,"It achieved 55.1 AP on COCO test-dev with 77M parameters."),(0,a.kt)("h2",{id:"process-flow"},"Process flow"),(0,a.kt)("p",null,"Step 1: Collect Images"),(0,a.kt)("p",null,"Capture via camera, scrap from the internet or use public datasets"),(0,a.kt)("p",null,"Step 2: Create Labels"),(0,a.kt)("p",null,"This step is required only if the object category is not available in any pre-trained model or labels are not freely available on the web. To create the labels (bounding boxes) using either open-source tools like Labelme or any other professional tool."),(0,a.kt)("p",null,"Step 3: Data Acquisition"),(0,a.kt)("p",null,"Setup the database connection and fetch the data into python environment"),(0,a.kt)("p",null,"Step 4: Data Exploration"),(0,a.kt)("p",null,"Explore the data, validate it and create preprocessing strategy"),(0,a.kt)("p",null,"Step 5: Data Preparation"),(0,a.kt)("p",null,"Clean the data and make it ready for modeling"),(0,a.kt)("p",null,"Step 6: Model Building"),(0,a.kt)("p",null,"Create the model architecture in python and perform a sanity check"),(0,a.kt)("p",null,"Step 7: Model Training"),(0,a.kt)("p",null,"Start the training process and track the progress and experiments"),(0,a.kt)("p",null,"Step 8: Model Validation"),(0,a.kt)("p",null,"Validate the final set of models and select/assemble the final model"),(0,a.kt)("p",null,"Step 9: UAT Testing"),(0,a.kt)("p",null,"Wrap the model inference engine in API for client testing"),(0,a.kt)("p",null,"Step 10: Deployment"),(0,a.kt)("p",null,"Deploy the model on cloud or edge as per the requirement"),(0,a.kt)("p",null,"Step 11: Documentation"),(0,a.kt)("p",null,"Prepare the documentation and transfer all assets to the client  "),(0,a.kt)("h2",{id:"use-cases"},"Use Cases"),(0,a.kt)("h3",{id:"automatic-license-plate-recognition"},"Automatic License Plate Recognition"),(0,a.kt)("p",null,"Recognition of vehicle license plate number using various methods including YOLO4 object detector and Tesseract OCR. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Automatic-License-Plate-Recognition-10ec22181b454b1facc99abdeadbf78f"},"here"),"."),(0,a.kt)("h3",{id:"object-detection-app"},"Object Detection App"),(0,a.kt)("p",null,"This is available as a streamlit app. It detects common objects. 3 models are available for this task - Caffe MobileNet-SSD, Darknet YOLO3-tiny, and Darknet YOLO3. Along with common objects, this app also detects human faces and fire. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Object-Detector-App-c60fddae2fcd426ab763261436fb15d8"},"here"),". "),(0,a.kt)("h3",{id:"logo-detector"},"Logo Detector"),(0,a.kt)("p",null,"Build a REST API to detect logos in images. API will receive 2 zip files - 1) a set of images in which we have to find the logo and 2) an image of the logo. Deployed the model in AWS Elastic Beanstalk. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Logo-Detection-91bfe4953dcf4558807b342efe05a9ff"},"here"),"."),(0,a.kt)("h3",{id:"tf-object-detection-api-experiments"},"TF Object Detection API Experiments"),(0,a.kt)("p",null,"The TensorFlow Object Detection API is an open-source framework built on top of TensorFlow that makes it easy to construct, train, and deploy object detection models. We did inference on pre-trained models, few-shot training on single class, few-shot training on multiple classes and conversion to TFLite model. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Tensorflow-Object-Detection-API-499b017e502d4950a9d448fb35a41d58"},"here"),"."),(0,a.kt)("h3",{id:"pre-trained-inference-experiments"},"Pre-trained Inference Experiments"),(0,a.kt)("p",null,"Inference on 6 pre-trained models - Inception-ResNet (TFHub), SSD-MobileNet (TFHub), PyTorch YOLO3, PyTorch SSD, PyTorch Mask R-CNN, and EfficientDet. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Object-Detection-Inference-Experiments-568fa092b1d34471b676fd43a42974b2"},"here")," and ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Object-Detection-Inference-with-Pre-trained-models-da9e2e5bfab944bc90f568f6bc4b3e1f"},"here"),"."),(0,a.kt)("h3",{id:"object-detection-app-1"},"Object Detection App"),(0,a.kt)("p",null,"TorchVision Mask R-CNN model Gradio App. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/MaskRCNN-TorchVision-Object-Detection-Gradio-App-c22f2a13ab63493b9b38720b20c50051"},"here"),"."),(0,a.kt)("h3",{id:"real-time-object-detector-in-opencv"},"Real-time Object Detector in OpenCV"),(0,a.kt)("p",null,"Build a model to detect common objects like scissors, cups, bottles, etc. using the MobileNet SSD model in the OpenCV toolkit. It will task input from the camera and detect objects in real-time. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/Object-Detection-with-OpenCV-MobileNet-SSD-38ff496d2f0d427185a9c51cebc1ddf2"},"here"),". Available as a Streamlit app also (this app is not real-time)."),(0,a.kt)("h3",{id:"efficientdet-fine-tuning"},"EfficientDet Fine-tuning"),(0,a.kt)("p",null,"Fine-tune YOLO4 model on new classes. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/EfficientDet-fine-tuning-01a6ffd1e11f4dc1941073aff4b9b486"},"here"),"."),(0,a.kt)("h3",{id:"yolo4-fine-tuning"},"YOLO4 Fine-tuning"),(0,a.kt)("p",null,"Fine-tune YOLO4 model on new classes. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/YOLO-4-b32c2d2a4b8644b59f1c05e6887ffcca"},"here"),"."),(0,a.kt)("h3",{id:"detectron2-fine-tuning"},"Detectron2 Fine-tuning"),(0,a.kt)("p",null,"Fine-tune Detectron2 Mask R-CNN (with PointRend) model on new classes. Checkout the notion ",(0,a.kt)("a",{parentName:"p",href:"https://www.notion.so/YOLO-4-b32c2d2a4b8644b59f1c05e6887ffcca"},"here"),"."))}f.isMDXComponent=!0},92879:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-computer-vision-object-detection-slide29-981cc9dc9dbccf90a7a22f93eeb43f22.png"}}]);