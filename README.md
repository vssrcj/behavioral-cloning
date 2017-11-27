# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*made by [CJ](https://github.com/vssrcj)*

Overview
---
In this project, deep neural networks and convolutional neural networks are used with Keras to clone driving behavior.

Modal architecture and training strategy
---
LeNet's architecture was used, but it didn't perform optimally.

nVidea's Convolutional Neural Network, which was designed for this exact problem, is used instead, as described [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/):
<div>
   <img src="/images/nvidea-architecture.png" height="400" />
</div>

In addition to that architecture (that uses the Adam optimizer), 3 dropout (50%) layers are added between the *Dense* layers, to reduce overfitting.

The summary is as follows:
<table>
    <tr>
        <th>Layer</th><th>Output Shape</th>
    </tr>
    <tr><td>Normalization</td><td>(160, 320, 3)</td></tr>
    <tr><td>Cropping (top 60, bottom 20)</td><td>(100, 320, 3)</td></tr>
    <tr><td>5x5 Convolution (24 filters)</td><td>(38, 158, 24)</td></tr>
    <tr><td>5x5 Convolution (36 filters)</td><td>(17, 77, 36)</td></tr>
    <tr><td>5x5 Convolution (48 filters)</td><td>(7, 37, 48)</td></tr>
    <tr><td>3x3 Convolution (64 filters)</td><td>(5, 35, 64)</td></tr>
    <tr><td>3x3 Convolution (64 filters)</td><td>(3, 33, 64)</td></tr>
    <tr><td>Flatten</td><td>(6336)</td></tr>
    <tr><td>Linear Activation (100 Output)</td><td>(100)</td></tr>
    <tr><td>50% Dropout</td><td></td></tr>
    <tr><td>Linear Activation (50 Output)</td><td>(50)</td></tr>
    <tr><td>50% Dropout</td><td></td></tr>
    <tr><td>Linear Activation (10 Output)</td><td>(10)</td></tr>
    <tr><td>50% Dropout</td><td></td></tr>
    <tr><td>Linear Activation (1 Output)</td><td>(1)</td></tr>
</table>

The model uses an adam optimizer, so the learning rate is not tuned manually.

Image capturing process
---
The training images consists of driving three times around the track (twice counterclockwize, and once clockwize).

Image preprocessing
---
* The images were cropped, 60 pixels from the top, 20 from the bottom.
* For each instance / frame, there are 3 images recorded, 1 from the center of the car, and 2 from the sides (which were corrected for the steering angles.)

* For each of these 3 images, their horizontally flipped images were also used, for a total of 6 images per instance.

The following is one of the training images (taken from the center):
<div>
    <img src="/images/center.jpg" height="240">
</div>
The following images are the center, left, and right preprocessed images of the above image:
<div>
    <img src="/images/left_aug.jpg" height="120">
</div>
<div>
    <img src="/images/center_aug.jpg" height="120">
</div>
<div>
    <img src="/images/right_aug.jpg" height="120">
</div>

The training process
---
80% of the images were used for training, and 20% were used for validation.

The training was performed on a GeForce GTX 1070, and it took about 300 seconds per epoch on ~3000 images.

After 4 epochs using the model used above, an impressive 0.0075 validation MSE is achieved.
Any more epochs causes overfitting, and the MSE rises.
<div>
   <img src="/images/graph.png" height="240" />
</div>

The result
---
The car follows the track safely around, never veer off track, or make unexpected moves.
The result can be seen in <a href="/video.mp4">video.mp4</a>.  The car completed the course at a speed of 14.

The car can even complete a track at a speed of 25, but its movements get erratic, and as a passanger, you won't feel safe.

Thoughts on the *advanced* track
---
I recored 3 laps around the advanced track (2 clockwize, 1 counterclockwize).  It achieved a final MSE loss of 0.01, but only managed to complete a quarter of a lap before plummeting off a cliff.

Other thoughs
---
I included the throttle in the training process as well.  It was simple:
* A y_train item consisted of a (steering, throttle) tuple, instead of just a steering value.
* The final layer was *Dense(2)* instead of *Dense(1)*.
* In **drive.py** the throttle was obtained from the prediction, and wasn't fixed.

The reason why it isn't used in the submission of the project, is because it didn't add to the accuracy of the model,
and the speed was typically too high for the steering to adjust in time.

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
