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

<!-- #region id="iZULaGX7_H1u" -->
# Running a YOLOv4 Object Detector with Darknet in the Cloud! (GPU ENABLED)
This tutorial will help you build YOLOv4 easily in the cloud with GPU enabled so that you can run object detections in milliseconds!
<!-- #endregion -->

<!-- #region id="O2w9w1Ye_nk1" -->
# Step 1: Enabling GPU within your notebook
You will want to enable GPU acceleration within your Colab notebook so that your YOLOv4 system will be able to process detections over 100 times faster than CPU.

### Steps:

### i) Click **Edit** at top left of your notebook
### ii) Click **Notebook Settings** within dropdown
### iii) Under 'Hardware Accelerator' select **GPU** and then hit **Save**

Your notebook should now have GPU enabled!
<!-- #endregion -->

<!-- #region id="gE3azXoR_pWi" -->
# Step 2: Cloning and Building Darknet
The following cells will clone darknet from AlexeyAB's famous repository, adjust the Makefile to enable OPENCV and GPU for darknet and then build darknet.

Do not worry about any warnings when you run the '!make' cell!
<!-- #endregion -->

```python id="68eMertgIxaB" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1593043398702, "user_tz": 240, "elapsed": 2333, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="9e7cd7bf-45e8-478e-e52b-bec9919e7952"
# clone darknet repo
!git clone https://github.com/AlexeyAB/darknet
```

```python id="xym8_m8CIyXK" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1593043404789, "user_tz": 240, "elapsed": 8412, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="480d2c73-847d-45cf-d2df-f863b94d30c7"
# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

```python id="5uloUwmUKF05" colab={"base_uri": "https://localhost:8080/", "height": 87} executionInfo={"status": "ok", "timestamp": 1593043406541, "user_tz": 240, "elapsed": 10160, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="459408c6-93b7-4b4f-c2fc-c71e1d8f6552"
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```

```python id="q2Jjv0yRKLPe" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593043483743, "user_tz": 240, "elapsed": 87360, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="f02743d8-8cd0-46e6-89cd-94f6461ebbc8"
# make darknet (builds darknet so that you can then use the darknet executable file to run or train object detectors)
!make
```

<!-- #region id="ikgv38AdKldo" -->
# Step 3: Download pre-trained YOLOv4 weights
YOLOv4 has been trained already on the coco dataset which has 80 classes that it can predict. We will grab these pretrained weights so that we can run YOLOv4 on these pretrained classes and get detections.
<!-- #endregion -->

```python id="vVtgyY_ELoSf" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1593043494810, "user_tz": 240, "elapsed": 98424, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="4af76309-0e35-406a-cf67-8f3d45933dce"
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

<!-- #region id="xn7TpB-UMBWf" -->
# Step 4: Define Helper Functions

These three functions are helper functions that will allow you to show the image in your Colab Notebook after running your detections, as well as upload and download images to and from your Cloud VM.
<!-- #endregion -->

```python id="G9Fv0wjCMPYY"
# define helper functions
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

# use this to upload files
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

# use this to download a file  
def download(path):
  from google.colab import files
  files.download(path)
```

<!-- #region id="zROXpfwcMcJW" -->
# Step 5: Run Your Detections with Darknet and YOLOv4!
Darknet is now built and ready to run detections using YOLOv4 in the cloud! You can find out which sorts of classes the pre-trained YOLOv4 weights can detect by clicking here. [COCO CLASSES](http://cocodataset.org/#explore)

The object detector can be run using the following command
```bash
!./darknet detector test <path to .data file> <path to config> <path to weights> <path to image>
```
Darknet comes with a few images already installed in the darknet/data/ folder.

**Note:** After running detections OpenCV can't open the image instantly in the cloud so we must run: 
```bash
imShow('predictions.jpg')
```
This will output the image with the detections shown. The most recent detections are always saved to 'predictions.jpg'

Try out the examples below for yourself!
<!-- #endregion -->

```python id="jESyRhd4Nd38" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593043500726, "user_tz": 240, "elapsed": 104330, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="8a9156b3-2983-4387-fa06-6c9310f1aef7"
# run darknet detection on test images
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg
```

```python id="GQQrAMdXN22a" colab={"base_uri": "https://localhost:8080/", "height": 575} executionInfo={"status": "ok", "timestamp": 1593043504299, "user_tz": 240, "elapsed": 107900, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="3c062bad-6099-4a8f-d9bc-1c96653e8442"
# show image using our helper function
imShow('predictions.jpg')
```

```python id="32MX4xIkUyKM" colab={"base_uri": "https://localhost:8080/", "height": 185} executionInfo={"status": "error", "timestamp": 1593043504326, "user_tz": 240, "elapsed": 107923, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="1d480c92-c162-46c8-9982-9a94eceff655"
# This stops 'Run all' at this cell by causing an error
assert False
```

<!-- #region id="F6W6MlHRVEfA" -->
# Step 6: Uploading Local or Google Drive Files to Use
You may want to run detections on more than just the images within the darknet/data/ folder. This step will show you how to upload local or Google Drive files to the cloud VM and run detections on them!
<!-- #endregion -->

<!-- #region id="yqDN3aOWWCzE" -->
### Method 1: Local Files
To upload local files just use our helper function by running 'upload()' as seen below. Click **Choose Files** and select the image from your local machine that you want to upload to the cloud VM.

If this function doesn't work for you then click the **Upload** button in the File Explorer on the left side of your notebook.

The image should save to the root directory of the cloud VM so that you can access it from the darknet command by running.
```
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights ../<your image name>
```
<!-- #endregion -->

```python id="EZouo3ZDWU8q"
# try out the upload helper function! (I uploaded an image called highway.jpg, upload whatever you want!)
%cd ..
upload()
%cd darknet
```

```python id="uRvm2x8OYZd3"
# run darknet with YOLOv4 on your personal image! (note yours will not be called highway.jpg so change the name)
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights ../highway.jpg
imShow('predictions.jpg')
```

<!-- #region id="fGKcGR_kZAU7" -->
### Method 2: Google Drive
Images can also be uploaded from your Google Drive and easily have YOLOv4 detections run on them.

You will want to run the below cell to mount your google drive into the cloud VM so that you can access its contents. It is that easy!

**NOTE:** We will be creating a symbolic link between '/content/gdrive/My\ Drive/' and '/mydrive.

This means we are just creating a shortcut '/mydrive' to map to the contents within the folder '/content/gdrive/My\ Drive/'.

The reason for this is that sometime having the space in 'My Drive' folder path can cause issues when running certain commands. This symbolic link will stop this from happening!

Now you can run YOLOv4 with images from Google Drive using the darknet command:
```
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /mydrive/<path to image>
```
I recommend saving images within a folder called 'images' at the root level of your Google Drive.
<!-- #endregion -->

```python id="uMLpWp-DZSZY" colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"status": "ok", "timestamp": 1593043736420, "user_tz": 240, "elapsed": 27427, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="6c8a4d31-a38e-4036-c883-76b1e0fc5114"
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')
```

```python id="bBFiu7rFZdmU" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1593043743745, "user_tz": 240, "elapsed": 3507, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="24c52003-9d19-4227-eb8f-e55d4ce8dec7"
# this creates a symbolic link so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
```

```python id="JafIRIU0Z6UQ" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1593043760243, "user_tz": 240, "elapsed": 768, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="b1640ae1-f848-4756-a5f1-748f2265848b"
# cd back into the darknet folder to run detections
%cd darknet
```

```python id="g3yVsww8aCn8"
# run detections on image within your Google Drive!
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /mydrive/images/street.jpg
imShow('predictions.jpg')
```

<!-- #region id="5FU3Pz9ja8gP" -->
### Download Files to Local Machine or Google Drive from Cloud VM
You can also easily download images from your cloud VM to save to your local machine or Google Drive. 

#### Method 1: Local Machine

You can do it easily by using our helper function 'download()' or by right clicking the image in the File Explorer on the left side of your notebook and hitting **Download**. Files will be saved to your *Downloads* folder.

This is useful if you want to download the **'predictions.jpg'** images that the object detector outputs. 

#### Method 2: Google Drive
A simple copy command can copy over any file to your Google Drive as it is already mounted. (you must run the mount command above if you have not already)

```
!cp <file to download> <destination to save file>
```
See example of each below!
<!-- #endregion -->

```python id="zEJPj5HQa901"
# LOCAL MACHINE DOWNLOAD
# if you get an error first run then run it again and it should work
download('predictions.jpg')
```

```python id="R_HooCprbDKZ"
# GOOGLE DRIVE DOWNLOAD
# note that I can change what the image name is saved as (I am saving it as detection1.jpg)
!cp predictions.jpg /mydrive/images/detection1.jpg
```

<!-- #region id="AoxKr73dgd3i" -->
# Step 7: Running YOLOv4 on Video in the Cloud!

You can also run YOLOv4 object detections on video in your Google Colab VM. Follow the cells below to see how to run videos from your local machine or from your Google Drive! 
<!-- #endregion -->

<!-- #region id="RSVop1HHkFOo" -->
## Local Machine Video
Here is how to upload video from local machine, run detector and then download video showing detections.
<!-- #endregion -->

```python id="EmKMPcmDfF7X"
# upload the video of your choosing! (Feel free to use the same video I do, it is in the Github repository)
upload()
```

```python id="Lgf-ZUbqbb4K"
!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show test.mp4 -i 0 -out_filename results.avi
```

```python id="fkPm6t6yj6Ky"
# download the video with detections shown
download('results.avi')
```

<!-- #region id="nPskTMswkVmS" -->
## Google Drive Video
Here is how to run detector on video stored in Google Drive and save video straight to Google drive as well!

**Note:** You will have to change the paths to where your video is stored within your Google Drive and to where you want the resulting video stored. I have a videos folder in the home directory of my Google Drive.
<!-- #endregion -->

```python id="rUSlB1hRkjhX"
!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show /mydrive/videos/test.mp4 -i 0 -out_filename /mydrive/videos/results.avi
```

<!-- #region id="SSs6pVUwlDyj" -->
# Step 8: Customize YOLOv4 with the different command line flags.
Darknet and YOLOv4 have a lot of command line flags you can add to your '!./darknet detector ...' to allow it to be customizeable and flexible.

I will show a few examples of these flags that you can take advantage of! Feel free to mix and match them together to customize your detections in any way you want.
<!-- #endregion -->

<!-- #region id="kRr0itg5sJFc" -->
## Threshold Flag
There is a flag '-thresh' you can use to add a threshold for confidences on the detections.  Only detections with a confidence level above the threshold you set will be returned.

In the example below we run darknet with YOLOv4 without a threshold on the test image dog.jpg. The image returns four detections, the lowest confidence being on the pottedplant with 33%.

If we add the '-thresh 0.5' flag this will only output three detections as now pottedplant falls below the threshold and is ignored. 

Check it out below!
<!-- #endregion -->

```python id="bK2UB9h2st99"
# this is ran without the threshold flag set
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
imShow('predictions.jpg')
```

```python id="nhPyF5wNtph7"
# same detections but ran with the threshold flag set to 0.5 (pottedplant is no longer detected!)
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg -thresh 0.5
imShow('predictions.jpg')
```

<!-- #region id="VJVZjOlbuAU4" -->
## Output Bounding Box Coordinates
You can output bounding box coordinates for each detection with the flag '-ext_output'. This external outputs flag will give you a few extra details about each detection within an image.

Check it out below!
<!-- #endregion -->

```python id="ZljsbwXnubZc"
# darknet run with external output flag to print bounding box coordinates
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -ext_output
imShow('predictions.jpg')
```

<!-- #region id="ifqddnZ0yjCd" -->
## Don't Show Image
You can add the flag '-dont_show' to not have the image outputted after running darknet. This doesn't really affect anything when running in Colab as the image is unable to output properly straight from darknet anyways. However, by adding the -dont_show flag you will get rid of the following warning from showing.
```
Unable to init server: Could not connect: Connection refused

(predictions:1850): Gtk-WARNING **: 17:01:00.687: cannot open display:
```

This is an important flag to have when running darknet with YOLOv4 on video as it will suppress having the video shown.
<!-- #endregion -->

```python id="HjHJX_Md1YdU"
# running darknet with dont show flag set (no longer get warnings)
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -dont_show
```

<!-- #region id="LPpfxAGA2Ldb" -->
# Step 9: Multiple Images at Once
YOLOv4 object detections can be run on multiple images at once. This is done through having a text file which has the paths to several images that you want to have the detector run on.


The .txt file should be in this format. One path to an image per line.

This file is stored to my Google Drive root directory and holds the path to three images within my Google Drive images folder.



<!-- #endregion -->

<!-- #region id="7jKaUSus6OPr" -->
## Save Results to .JSON File
Here is an example of saving the multiple image detections to a .JSON file.
<!-- #endregion -->

```python id="BjbebMZZ50Zg"
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < /mydrive/images.txt
```

```python id="CWPWB_yICGlL"
download('result.json')
```

<!-- #region id="xhh-N0GF8GC4" -->
## Saving Results to a .txt File
You can also save the results of running multiple images to a text file.
<!-- #endregion -->

```python id="Fn3wzgCA8mv8"
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < /mydrive/images.txt > result.txt
```

```python id="YLKxikH6CN22"
download('result.txt')
```

<!-- #region id="qaONUTI2l4Sf" -->
# How to Train Your Own YOLOv4 Custom Object Detector!
Now comes the time to create your own custom YOLOv4 object detector to recognize any classes/objects you want!

This requires a couple tricks and tips so make sure to follow along closely with the rest of the tutorial.

In order to create a custom YOLOv4 detector we will need the following:

*   Labeled Custom Dataset
*   Custom .cfg file
*   obj.data and obj.names files
*   train.txt file (test.txt is optional here as well)
<!-- #endregion -->

<!-- #region id="ulw-TMsaBSUq" -->
# Step 1: Gathering and Labeling a Custom Dataset
In order to create a custom object detector you need a good dataset of images and labels so that the detector can be efficiently trained to detect objects.

This can be done in two ways. through  or through using Google images or creating your own dataset and using an annotation tool to manually draw labels. **(I recommend the first way!)**
<!-- #endregion -->

<!-- #region id="_5oMbqCBBlRP" -->
### Method 1: Using Google's Open Images Dataset (RECOMMENDED)
This method is the method I recommend as you can gather thousands of images and auto-generate their labels within minutes! Gathering a dataset from Google's Open Images Dataset and using OIDv4 toolkit to generate labels is easy and time efficient. The dataset contains labeled images for over 600 classes! [Explore the Dataset Here!](https://storage.googleapis.com/openimages/web/index.html)

Watch one of my previous youtube videos on how to do use this approach to create your own custom dataset!

[Creating a Custom YOLOv3 Dataset (Video)](https://www.youtube.com/watch?v=_4A9inxGqRM) **(images and labels are same format for YOLOv4 so video still works)**

Here is a link to my github repository for the OIDv4 toolkit! [Github Repo](https://github.com/theAIGuysCode/OIDv4_ToolKit)

For this tutorial I will be creating a license plate object detector using data from Google's Open Images Dataset. 
<!-- #endregion -->

<!-- #region id="-XD-je3J15Ud" -->
#### Training Dataset
I ran the following commands within the toolkit to create my custom training dataset. I chose to use 1500 images within my dataset but the general rule is the more images the better for higher accuracy of your model. 
```
python main.py downloader --classes 'Vehicle registration plate' --type_csv train --limit 1500
```
**Note: If you are training a detector for multiple classes add '--multiclasses 1' to the command above to have all your training images saved together in one folder.**

This creates a folder with 1500 images of 'Vehicle registration plates' (license plates) for us to train our custom object detector with.
<!-- #endregion -->

<!-- #region id="c7BeRTgCopdg" -->
#### Validation Dataset
**Note:** Having a validation dataset is optional but highly recommended.

We want to have a validation dataset in order to properly test our custom object detector after it is trained. We can take advantage of the OIDv4_Toolkit as it has seperate images for validation to make sure we have different images and not ones that are in our training dataset.

We can run the same commands above but for validation. I recommend having a validation dataset size that is roughly 20-30% of your training dataset size.
Since I chose 1500 images for my training data this means I will have a validation size of 300 images. (20% of 1500 = 300)

```
python main.py downloader --classes 'Vehicle registration plate' --type_csv validation --limit 300
```
#### Now you have a proper validation dataset to test your custom object detector after training!

<!-- #endregion -->

<!-- #region id="h8YlJovaTg7C" -->
#### Converting Labels to YOLOv4 Format
The labels that we get from the toolkit are not in the proper YOLOv4 format. Luckily, with a few simple commands we can convert them to the proper format.

Within the root OIDv4_ToolKit folder open the file classes.txt and edit it to have the classes you just downloaded, one per line.


Example of classes.txt for multiclass object detector:


Now run this command to convert the labels generated by OIDv4_Toolkit into YOLOv4 labels (this should convert both train and validation dataset labels):
```
python convert_annotations.py
```

This converts all labels to YOLOv4 format which can now be used by darknet to properly train our custom object detector.

Remove the old 'Label' folder in the train and validation folders which contains the non YOLOv4 formatted labels by running the following: (your file path will have a different name for 'Vehicle registration plate'... depending on which classes you downloaded.
```
rm -r OID/Dataset/train/'Vehicle registration plate'/Label/
```
And finally delete the old labels for validation as well.
```
rm -r OID/Dataset/validation/'Vehicle registration plate'/Label/
```
If this command doesn't work on your machine then just go to the folder with 'Label' and right click and hit Delete to manually delete it.

The folders with all your images and annotations should now look like this. Each image should have a text file with the same name beside (or below) it.


## You now have your Custom YOLOv4 Training and Validation Datasets with proper labels!
<!-- #endregion -->

<!-- #region id="xQvuhPPlkzeL" -->
### Method 2: Manually Labeling Images with Annotation Tool
If you can't find the proper images or classes within Google's Open Images Dataset then you will have to use an annotation tool to manually draw your labels which can be a tiresome process.

I have created a previous video where I walk through how to mass download images from Google Images along with how to use LabelImg, an annotation tool, in order to create a custom dataset for YOLOv4. Hit link below to learn how.

[Create Labels and Annotations for Custom YOLOv3 Dataset (Video)](https://www.youtube.com/watch?v=EGQyDla8JNU) **(same format for YOLOv4)**

After following the tutorial video you should be able to have a folder with images and text files as your training dataset as well as one for your validation dataset like the one above.

## You now have your Custom YOLOv4 Datasets!
<!-- #endregion -->

<!-- #region id="9ket44SxkdBu" -->
# Step 2: Moving Your Custom Datasets Into Your Cloud VM
So now that you have your datasets properly formatted to be used for training and validation, we need to move them into this cloud VM so that when it comes the time we can actually train and validate our model.

I recommend renaming the trainin dataset folder with your images and text files on your local machine to be called '**obj**' and then creating a .zip folder of the 'obj' folder. Then I recommend uploading the zip to your Google Drive. So you should now have obj.zip someplace in your Google drive. 

Do the same with your validation dataset but name it '**test**'. So you should now have **test.zip** also uploaded to your Google Drive.

This will **greatly reduce** the time it takes to transfer our dataset into our cloud VM.

Now we can copy in the zips and unzip them in your cloud VM.
<!-- #endregion -->

```python id="YL--CsG0Vevs" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1593044410415, "user_tz": 240, "elapsed": 2581, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="55f2d4ad-23de-4bab-9f3d-09ae8a50c88a"
# this is where my datasets are stored within my Google Drive (I created a yolov4 folder to store all important files for custom training) 
!ls /mydrive/yolov4
```

```python id="zqLQF0fNWJro"
# copy over both datasets into the root directory of the Colab VM (comment out test.zip if you are not using a validation dataset)
!cp /mydrive/yolov4/obj.zip ../
!cp /mydrive/yolov4/test.zip ../
```

```python id="A9mYUoKOWWlR" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593044540834, "user_tz": 240, "elapsed": 7710, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="a6451ef6-a34c-4dd6-8c0a-7806766acfb3"
# unzip the datasets and their contents so that they are now in /darknet/data/ folder
!unzip ../obj.zip -d data/
!unzip ../test.zip -d data/
```

<!-- #region id="SWjNcjfcXKgx" -->
# Step 3: Configuring Files for Training
This step involves properly configuring your custom .cfg, obj.data, obj.names, train.txt and test.txt files.

It is important to configure all these files with extreme caution as typos or small errors can cause major problems with your custom training.
<!-- #endregion -->

<!-- #region id="nWZqyTvxdZW1" -->
## i) Cfg File
Copy over the yolov4.cfg to your Google Drive by running the cell below. This will allow us to edit it in a text editor.
<!-- #endregion -->

```python id="WpI1Y_EzdtYx"
# download cfg to google drive and change its name
!cp cfg/yolov4-custom.cfg /mydrive/yolov4/yolov4-obj.cfg
```

```python id="vFJ0-dkueG-F"
# to download to local machine (change its name to yolov4-obj.cfg once you download)
download('cfg/yolov4-custom.cfg')
```

<!-- #region id="YQUDXrhJekxl" -->
Now you need to edit the .cfg to fit your needs based on your object detector. Open it up in a code or text editor to do so.

If you downloaded cfg to google drive you can use the built in  **Text Editor** by going to your google drive and double clicking on yolov4-obj.cfg and then clicking on the **Open with** drop down and selectin **Text Editor**.

**(Image from previous tutorial so don't mind different file name)**


I recommend having **batch = 64** and **subdivisions = 16** for ultimate results. If you run into any issues then up subdivisions to 32.

Make the rest of the changes to the cfg based on how many classes you are training your detector on.

**Note:** 
I set my **max_batches = 6000**, **steps = 4800, 5400**, I changed the **classes = 1** in the three YOLO layers and **filters = 18** in the three convolutional layers before the YOLO layers.

How to Configure Your Variables:

width = 416

height = 416
**(these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)**

max_batches = (# of classes) * 2000
**(but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)**

steps = (80% of max_batches), (90% of max_batches)
**(so if your max_batches = 10000, then steps = 8000, 9000)**

filters = (# of classes + 5) * 3
**(so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)**


**Optional:** If you run into memory issues or find the training taking a super long time. In each of the three yolo layers in the cfg, change one line from random = 1 to **random = 0** to speed up training but slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

<!-- #endregion -->

```python id="kXkXb1tlLCnE"
# upload the custom .cfg back to cloud VM from Google Drive
!cp /mydrive/yolov4/yolov4-obj.cfg ./cfg
```

<!-- #region id="a4b33o9EiHUD" -->
## ii) obj.names and obj.data 
Create a new file within a code or text editor called **obj.names** where you will have one class name per line in the same order as your classes.txt from the dataset generation step.

**NOTE:** You do not want to have spaces in your class name. For this reason I changed "Vehicle Registration Plate" to **license_plate**.


Example for multiclass obj.names file:


You will also create a **obj.data** file and fill it in like this (change your number of classes accordingly, as well as your backup location)


This backup path is where we will save the weights to of our model throughout training. Create a backup folder in your google drive and put its correct path in this file.
<!-- #endregion -->

```python id="1vuqvzvLUtw2"
# upload the obj.names and obj.data files to cloud VM from Google Drive
!cp /mydrive/yolov4/obj.names ./data
!cp /mydrive/yolov4/obj.data  ./data
```

<!-- #region id="zIA38VpCU1y9" -->
## iii) Generating train.txt and test.txt
The last configuration files needed before we can begin to train our custom detector are the train.txt and test.txt files which hold the relative paths to all our training images and valdidation images.

Luckily I have created scripts that eaily generate these two files withe proper paths to all images.

The scripts can be accessed from the [Github Repo](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)

Just download the two files to your local machine and upload them to your Google Drive so we can use them in the Colab Notebook.
<!-- #endregion -->

```python id="4mgiK3C2ZF0k"
# upload the generate_train.py and generate_test.py script to cloud VM from Google Drive
!cp /mydrive/yolov4/generate_train.py ./
!cp /mydrive/yolov4/generate_test.py ./
```

<!-- #region id="eq9gDuAaJWM6" -->
Now simply run both scripts to do the work for you of generating the two txt files.
<!-- #endregion -->

```python id="DAKjJSJwJb5R"
!python generate_train.py
!python generate_test.py
```

```python id="nv5yDQptaJRE" colab={"base_uri": "https://localhost:8080/", "height": 105} executionInfo={"status": "ok", "timestamp": 1593045663616, "user_tz": 240, "elapsed": 1956, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="2f0b3a55-8f1a-4695-dccf-ce6cfc275368"
# verify that the newly generated train.txt and test.txt can be seen in our darknet/data folder
!ls data/
```

<!-- #region id="5mFOT8pC6Qbg" -->
If you are uncertain whether generating the files worked and want to double check that everything went as planned, double click on **train.txt** on the left side File Explorer and it should look like this.


It will contain one line for each training image path.
<!-- #endregion -->

<!-- #region id="nrkC4l-PbmNS" -->
# Step 4: Download pre-trained weights for the convolutional layers.
This step downloads the weights for the convolutional layers of the YOLOv4 network. By using these weights it helps your custom object detector to be way more accurate and not have to train as long. You don't have to use these weights but trust me it will help your modle converge and be accurate way faster. USE IT!
<!-- #endregion -->

```python id="KcUKbtO-byOE" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1593045721021, "user_tz": 240, "elapsed": 7876, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="586123c5-a973-481c-b3e2-7813a800d471"
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

<!-- #region id="POozxsvFdXTu" -->
# Step 5: Train Your Custom Object Detector!
The time has finally come! You have made it to the moment of truth! You are now ready to train your custom YOLOv4 object detector on whatever crazy classes you have decided on. So run the following command. (-dont_show flag stops chart from popping up since Colab Notebook can't open images on the spot, -map flag overlays mean average precision on chart to see how accuracy of your model is, only add map flag if you have a validation dataset)
```
!./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
```
**TIP:** This training could take several hours depending on how many iterations you chose in the .cfg file. You will want to let this run as you sleep or go to work for the day, etc. However, Colab Cloud Service kicks you off it's VMs if you are idle for too long (30-90 mins).

To avoid this hold (CTRL + SHIFT + i) at the same time to open up the inspector view on your browser.

Paste the following code into your console window and hit **Enter**
```
function ClickConnect(){
console.log("Working"); 
document
  .querySelector('#top-toolbar > colab-connect-button')
  .shadowRoot.querySelector('#connect')
  .click() 
}
setInterval(ClickConnect,60000)
```
Looks like this, it will click the screen every 10 minutes so that you don't get kicked off for being idle! HACKS!

<!-- #endregion -->

```python id="imc0NP19hLuq" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593046910520, "user_tz": 240, "elapsed": 29137, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="0192fde4-fc25-4a41-da87-8c33b44db3b2"
# train your custom detector! (uncomment %%capture below if you run into memory issues or your Colab is crashing)
# %%capture
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
```

<!-- #region id="5GjIkYFjhnYL" -->
After training, you can observe a chart of how your model did throughout the training process by running the below command. It shows a chart of your average loss vs. iterations. For your model to be 'accurate' you should aim for a loss under 2.
<!-- #endregion -->

```python id="uXPGYKmchv7Z" colab={"base_uri": "https://localhost:8080/", "height": 575} executionInfo={"status": "ok", "timestamp": 1593047083731, "user_tz": 240, "elapsed": 1967, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="c6522130-ad70-4fcd-a89f-6c66840b8331"
# show chart.png of how custom object detector did with training
imShow('chart.png')
```

<!-- #region id="28HHdjQOG1QL" -->
Here is what the chart.png should look like after an uninterrupted training! If you stop training or it crashes during training your chart will look like the above one but don't worry you can still check accuracy of your model in the next steps.

<!-- #endregion -->

<!-- #region id="8q2gyKSujCxr" -->
**TRICK**: If for some reason you get an error or your Colab goes idle during training, you have not lost your partially trained model and weights! Every 100 iterations a weights file called **yolov4-obj_last.weights** is saved to **mydrive/yolov4/backup/** folder (wherever your backup folder is). This is why we created this folder in our Google drive and not on the cloud VM. If your runtime crashes and your backup folder was in your cloud VM you would lose your weights and your training progress.

We can kick off training from our last saved weights file so that we don't have to restart! WOOHOO! Just run the following command but with your backup location.
```
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show
```
<!-- #endregion -->

```python id="Fl7PsmikjCBW" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593047371230, "user_tz": 240, "elapsed": 106711, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="095d317c-9960-4fe7-da21-7616e475e5c5"
# kick off training from where it last saved
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show
```

<!-- #region id="voia4UtxlPMa" -->
# Step 6: Checking the Mean Average Precision (mAP) of Your Model
If you didn't run the training with the '-map- flag added then you can still find out the mAP of your model after training. Run the following command on any of the saved weights from the training to see the mAP value for that specific weight's file. I would suggest to run it on multiple of the saved weights to compare and find the weights with the highest mAP as that is the most accurate one!

**NOTE:** If you think your final weights file has overfitted then it is important to run these mAP commands to see if one of the previously saved weights is a more accurate model for your classes.
<!-- #endregion -->

```python id="2UoSBmTFlNep" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593047910418, "user_tz": 240, "elapsed": 15410, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="974cd56d-7afa-45be-9af3-27f1d0c56d55"
!./darknet detector map data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_1000.weights
```

<!-- #region id="8dfPY2h39m-T" -->
# Step 7: Run Your Custom Object Detector!!!
You have done it! You now have a custom object detector to make your very own detections. Time to test it out and have some fun!
<!-- #endregion -->

```python id="xXkZNeA09o2f" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1593047989463, "user_tz": 240, "elapsed": 3586, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="7adfc9a3-a5cb-4833-a04e-45e336c8b282"
# need to set our custom cfg to test mode 
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
%cd ..
```

```python id="o7IIzEal90dA" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1593048049896, "user_tz": 240, "elapsed": 10898, "user": {"displayName": "The AI Guy", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiO9tsnUOEllOY95FgLi5LzK822WgvCyeFfzvGcBw=s64", "userId": "17800843676226924807"}} outputId="26a870ca-fc36-474c-b503-7f214fdb4db2"
# run your custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights /mydrive/images/car2.jpg -thresh 0.3
imShow('predictions.jpg')
```

<!-- #region id="H8DYUTHiEeGZ" -->
# Thanks!
Thanks for following along this tutorial, I hope it worked well for all of you!

If you enjoy YOLOv3 or YOLOv4 check out my other videos and subscribe to my channel so you don't miss out on any other amazing YOLO content! 

[The AI Guy](https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q)
<!-- #endregion -->
