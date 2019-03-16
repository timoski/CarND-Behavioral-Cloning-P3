# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./inages/figure_1.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on InterceptionV3.

First layer of my moded is a Lambda Layer to normalize the images. The second Layer crops the upper 70 pixel which do not contain any useful information.
Then the InterceptionV3 model is used. I used pretrained weight from imagenet to speed up the (training weights='imagenet'). No fully connected top layer at the top of the network is used (include_top=False). Global max pooling is applied (pooling=max).
To reduce overfitting a dropout layer with a dropout rate of 0.5 is added.
Finally a flatten layer and dense layer is added to the model. The output space of dens is set to 1, which represents the steering angle.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. Early stopping is used to stop the training as soon as the validation loss is increasing. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
As most curves in the training data are left curves, I although flipped all images to get more training data for right corner driving. These flipped images are used for training, too.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the InterceptionV3 I thought this model might be appropriate because it is known to achive very good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that InterceptionV3 works very well on driving in the simulator.

To combat overfitting, I added a dropout layer and added early stoppin.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model is based on InterceptionV3.

First layer of my moded is a Lambda Layer to normalize the images. The second Layer crops the upper 70 pixel which do not contain any useful information.
Then the InterceptionV3 model is used. I used pretrained weight from imagenet to speed up the (training weights='imagenet'). No fully connected top layer at the top of the network is used (include_top=False). Global max pooling is applied (pooling=max).
To reduce overfitting a dropout layer with a dropout rate of 0.5 is added.
Finally a flatten layer and dense layer is added to the model. The output space of dens is set to 1, which represents the steering angle.

#### 3. Creation of the Training Set & Training Process

I used the provided training data.

To augment the data sat, I also flipped images and angles thinking that this would help right curve dring as in the training data are not many right curves.

After the collection process, I had 48216 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
![Image1][image1]
