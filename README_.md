# Welcome to MOJO sperm detection challenge


## Problem

#### Concentration

In order to perform semen analysis, we capture images by means of optics and a camera (controlled and standardized hardware). Several positions are captured for the same patient sample. For each position, we grab several frames to be converted into a video. Our objective is to generate meaningful metrics out of these videos.

A metric of high importance for doctor is `spermatozoon concentration`. How many spermatozoa per volume unit for a given sample ? It's usually expressed in number of million of sperms per milliliter (M/ml)

Given the size of the image and the pixel pitch of the camera, we can compute the number of sperms per m2 (in a plan given by two X, Y axis). To infer the number of sperm per m3 we can multiply the surface analyzed by the height of our sample volume. The height is called depth of focus and corresponds to an interval of the Z axes where sperms are in the focal plan. Meaning that we only consider a part of the all volume.

Have look below at the image. Some sperms are blurry because outside of the focus of the camera. Some are really sharp in the focus of the camera.

The yellow annotation correspond to in focus sperm and purple annotations out of the focus sperm.

![alt text for screen readers](example.png "Image of in focus and out of focus spermatozoon")

In this case we are only interested in the "yellow" sperms, no more, no less.

The challenge here for a detection algorithm is to class in and class out these two types of sperms.

You may have noticed that sperms are not the only cells, nor objects in the image. Cells that are not sperm should not be counted as well as little convex objects called debris as well as bubble edges, etc..

Note: It's possible that some annotations are incorrect because of the relative difficulty of the task.

### Motility

The detection algorithm plays another role. That is to provide motility metrics. Doctors need to know if a sperm is motile or immotile which gives a great indicator of its health.
To provide such metrics, we track the sperms from frame to frame. Meaning that the detection algorithm will be then followed by a tracking algorithm that takes detection outputs as its input.
The goal here is not for you to perform tracking but rather know that the quality of the detections has an impact of subsequent processes.

Along with the dataset, we provide a couple of videos for you.
Feel free to use these to visualize the detection quality at the scale of a video of your algorithm and reflect on potential issues this can cause.

## Challenge

Given a set of images, you will design an algorithm able to count the number of in focus sperm per image.

## Dataset

Images are in a .jpg format and are grayscale (images folder).
Annotations are in a .txt format (labels folder)
Each image file name in the images folder corresponds to a unique file name in the labels folder and the other way round.

images / 0.jpg
labels / 0.txt

#### Images

Images are 1920x1200 pixels. Don't hesitate to resize them to speed up the algorithm. Because of our optic limitations, going above 960x600 is useless because information is redundant.

#### Annotations
We provide you a set of images with annotations. 

Important: The annotations only corresponds to "yellow" sperms, not "purple" sperms (out of focus) meaning that purple sperms will be considered as something to not detect like any object.

Also important: The annotations are square and not rectangle

The format of annotations are: c,x,y,w,h 
c is the class
x,y is the center of the square
w,h is the height and the width of the square

Notice that:
- there is only one class (c=0). 
- w, h are constant and does not bring additional information.
- x, y, w, h are all included in the interval ]0, 1[ because divided by the width and the height of the image


0 0.094271 0.057500 0.041667 0.066667

0 0.170313 0.038333 0.041667 0.066667

0 0.201042 0.993333 0.041667 0.066667

## Advice

Since you don't have much time to address this problem, we would advise you to focus on only one algorithm/model/framework. It could be interesting to compare several but it really isn't the goal.
Also don't spend too much time to tune or obtain best scores. It's not the goal either.

The primary goal is too see how you approach this problem and what techniques/tools you make use of to address it.
If you run out of time just write down what you would have liked to try and why.

Your code should be shareable with us and ideally we should be able to read it and launch it without your help.

## Questions/discussion

After you finished, think about some of the following questions:
 
What would you change ? What would you do if you had more time ? How to improve the quality of the detections ?

Do you know other algorithms and/or did you read papers you think would fit with the problem ? 

What do you think of our dataset (particularity, difficulties, etc) ?

What do you think of our capture and analysis pipeline ?

