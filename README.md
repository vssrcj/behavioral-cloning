# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

* made by [CJ](https://github.com/vssrcj)
Overview
---
In this project, deep neural networks and convolutional neural networks are used to clone driving behavior.
Keras is used to create the model.

Modal Used
---
NVIDEA's Convolutional Neural Network is used, as described below:
<src img="/nvidea-architecture.png" height="300" />

In addition to that architecture, dropout (50%) layers are added between the *Dense* layers, to reduce overfitting.

The Adam optimizer is used.

Image Capturing Process
---
I drove three times around the track (twice counterclockwize, and once clockwize).

Pre processing of images
---
The images were cropped, 60 pixels from the top, 20 from the bottom.
For each instance, there are 3 images recorded (1 from the center of the car, and 2 from the sides)
The side images were corrected for the steering angles.

For each of these 3 images, their flipped images were also used, for a total of 6 images per instance.

The training process
---
Firstly, the images were cropped
After 4 epochs using the model used above, an impressive 0.0075 validation MSE is achieved.
Any more epochs causes overfitting, and the MSE rises.
<src img="/graph.png" height="240" />

The result
---
The car follows the track safely around, never veer off track, or make unexpected moves.
The result is shown in *video.mp4*.  The car completed the course at a speed of 14.

The car can even complete a track at a speed of 25, but its movements get erratic, and as a passanger, you won't feel safe.

Thoughts on the *advanced* track
---
I recored 3 laps around the advanced track (2 clockwize, 1 counterclockwize).  It also achieved a final MSE loss of 0.01, but only managed to complete a quarter of a lap.

How to improve?
---
The reason it failed on the more difficult track, is because of:
* There isn't enough training data that indicate how to correct from a bad position.
* The images aren't calibrated / pre-processed sufficiently (especially when shadows appear on the road)
* The training model may need some tweaking.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
