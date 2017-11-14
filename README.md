# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*made by [CJ](https://github.com/vssrcj)*

Overview
---
In this project, deep neural networks and convolutional neural networks are used with Keras to clone driving behavior.

Modal Architecture and Training Strategy
---
nVidea's Convolutional Neural Network is used, as described below:
<img src="/nvidea-architecture.png" height="300" />

In addition to that architecture (that uses the Adam optimizer), 3 dropout (50%) layers are added between the *Dense* layers, to reduce overfitting.

Image Capturing Process
---
The training images consists of driving three times around the track (twice counterclockwize, and once clockwize).

Image Preprocessing
---
* The images were cropped, 60 pixels from the top, 20 from the bottom.
* For each instance / frame, there are 3 images recorded, 1 from the center of the car, and 2 from the sides (which were corrected for the steering angles.)

* For each of these 3 images, their horizontally flipped images were also used, for a total of 6 images per instance.

The training process
---
80% of the images were used for training, and 20% were used for validation.

After 4 epochs using the model used above, an impressive 0.0075 validation MSE is achieved.
Any more epochs causes overfitting, and the MSE rises.
<img src="/graph.png" height="240" />

The result
---
The car follows the track safely around, never veer off track, or make unexpected moves.
The result can be seen in *video.mp4*.  The car completed the course at a speed of 14.

The car can even complete a track at a speed of 25, but its movements get erratic, and as a passanger, you won't feel safe.

Thoughts on the *advanced* track
---
I recored 3 laps around the advanced track (2 clockwize, 1 counterclockwize).  It achieved a final MSE loss of 0.01, but only managed to complete a quarter of a lap before plummeting off a cliff.

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
