# **Traffic Sign Recognition** 



**GOAL: Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./submission_images/histogram.png "Data Histogram"
[image2]: ./submission_images/mean_images.png "Mean Images"
[image3]: ./submission_images/sample_images.png "Sample Images"
[image4]: ./submission_images/preprocessing.png "Preprocessing"
[image5]: ./submission_images/data_augmentation.png "Data Augmentation"
[image6]: ./submission_images/loss_vs_accuracy.png "Loss vs. Accuracy"
[image7]: ./submission_images/test_images.png "Test Images"
[image8]: ./submission_images/1.jpg   "Test Image Class 1  - Speed limit (30km/h)"
[image9]: ./submission_images/4.jpg   "Test Image Class 4  - Speed limit (70km/h)"
[image10]: ./submission_images/9.jpg  "Test Image Class 9  - No passing"
[image11]: ./submission_images/11.jpg "Test Image Class 11 - Right-of-way at the next intersection"
[image12]: ./submission_images/17.jpg "Test Image Class 17 - No entry"
[image13]: ./submission_images/intermediate_output_visualization_conv1.png "Network output at conv1"
[image14]: ./submission_images/intermediate_output_visualization_conv2.png "Network output at conv2"

## Rubric Requirements:
### Below are the requirements according to the [rubric file](https://review.udacity.com/#!/rubrics/481/view)

---
### Submission files
The project submission includes all required files.

#### 1. Ipython notebook with code
This is the completed [notebook](https://github.com/mythreyajl/SDCND-P2-TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb) with code.

#### 2. HTML output of the code
Downloaded [HTML](https://github.com/mythreyajl/SDCND-P2-TrafficSignClassification/blob/master/Traffic_Sign_Classifier.html)

#### 3. A writeup report (either pdf or markdown)
Let this markdown file serve as the writeup report.

### Dataset Exploration

#### 1.Dataset Summary
I calculated the statistics of the provided dataset using numpy
* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory Visualization
##### 1. Samples per class for train, test and validation datasets.
The following shows that while some classes have a lot of samples, others are deprived. I take some steps to augment data for these deprived classes which I will explain later in the report.
![alt text][image1]

##### 2. Mean training image for each class 
The following collage contains mean image per class after brightness normalization of each image.
![alt text][image2]

##### 3. Sample training image for each class 
The following collage contains a sample image per class after brightness normalization.
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Preprocessing:
The steps I took for preprocessing are as follows:
* Converted from RGB to Grayscale (Y of YUV) using OpenCV - The reason being, often times, most of the information is contained in the intensity (Y) channel of an image. The U and V channels, athough very informative, doesn't help as much in recognition as the intensity channel does. Further, due to this, memory, network size and CPU usage reduces significantly due to eliminating the other two channels
* Normalized the image by subtracting 128 and dividing by 128 - the mean value of an image pixel is 128 for an 8-bit image. By doing this operation, the image is guaranteed to be in the interval [-1. 1] which effectively normalizes the data to bring all pixels of all images to the same frame of reference.
* Brightness normalization -  In many cases, the image contrast and brightness is poor. This step attempts to fix the range of intensities in an image.

![alt text][image4]

#### 2. Data Augmentation:
I augment the data as suggested in the paper - [Traffic Sign Recognition with Multi-Scale Convolutional Networks](Traffic Sign Recognition with Multi-Scale Convolutional Networks). The two things that I hope to achieve with this are:
* To increase the number of samples for data deprived classes to aid its training
* To increase variation in the existing samples to make it robust during testing time.

For all images the following steps are taken:
* Random translations between -2 and 2 pixels
* Random scaling  between 0.9 and 1.1 times the base image
* Random rotation between -15 and 15 degrees

For images of classes that are deficient, these additional steps are taken:
* Random translations between -10 and 10 pixels
* Random scaling  between 0.5 and 1.5 times the base image
* Random rotation between -20 and 20 degrees
This step 'robustifies' the deficient classes more.

![alt text][image5]

#### 3. Model Architecture:
Initially I considered the vanilla-LeNet model. But adding a convolutional layer and a fully connected layer greatly improved the performance.

My final model consisted of the following layers:


| Layer         		      |     Description	        					               | 
| --------------------- | ------------------------------------------- | 
| Input         		      | 32x32x1 Grayscale image   							           | 
| Convolution 3x3     	 | 1x1 stride, valid padding, outputs 28x28x20 |
| RELU					             |												                                 |
| Max pooling	      	   | 2x2 stride,  outputs 14x14x20 			           |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 10x10x40 |
| RELU					             |												                                 |
| Max pooling	      	   | 2x2 stride,  outputs 5x5x40 				            |
| Convolution 3x3	      | 1x1 stride, same padding, outputs 5x5x80    |
| RELU					             |											                                  |
| Fully connected		     | outputs 2000        									               |
| RELU					             |											                                  |
| Fully connected		     | outputs 1000        									               |
| RELU					             |											                                  |
| Fully connected		     | outputs 400         									               |
| RELU					             |											                                  |
| Fully connected		     | outputs 200         									               |
| RELU					             |											                                  |
| Fully connected		     | outputs 43        									                 |
| Softmax				           |         									                           |


#### 4. Model Training:
##### Training
The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
* Cross-entropy - Following one-hot encoding, a softmax cross entropy is calculated
* Loss operation - This is then fed to the loss operation that calculates the mean of cross entropy for the batch
* Optimizer - The Adam optimizer is used since this helps with decaying the learning rate naturally. 
* Training operation - The Adam optimizer is used to calculate the descent required to minimize loss
* Prediction - Prediction is the argmax of logits calculated by the network
* Learning rate - 0.0005, Batch size - 128 and Epochs - 50: These numbers were decided following various experiments described in the following subsection

##### Evaluation
* Correct prediction - If the argmax of logits is the same as the one-hot encoded label for the image, the prediction is correct
* Accuracy - The correct prediction for the each image of the batch converted to a float and average is calculated to get batch accuracy

#### 5. Solution Approach
My final model results were:
* Training set accuracy   : 99.941%
* Validation set accuracy : 97.506%
* Test set accuracy of    : 95.511%

I chose an iterative approach to the solution with the following justifications:
* What was the first architecture that was tried and why was it chosen?
I started off with the vanilla LeNet architecture. The motivation for this is that LeNet is a successful choice for handwritten digits and the Traffic Sign Classification is a very similar undertaking.
* What were some problems with the initial architecture?
The initial architecture was not big enough network to handle the Traffic Sign Classification which has significantly more classes and a lot more intra class variation and more inter class similarity. 
* How was the architecture adjusted and why was it adjusted? 
The architecture was adjusted to have an additional convolution layer, bigger layer depth per convolution layer and and additional fully connected layer to gradually increase the ROI in the top end of the network considering the increased depth in the convolutional layers. The reason for this choice is that I noticed that the network wasn't accurate enough with vanilla LeNet and it was underfitting quite a bit. 
* Which parameters were tuned? How were they adjusted and why?
Several parameters were tuned. Network depth in the convolutional layers determined the size of the network. This was the first thing that was tuned. The learning rate was tuned to make gradually decrease loss while increasing accuracy.

![alt text][image6]

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?
- Convolutional layers were chosen in the front-end of the network. This is useful in extracting several features from the images while keeping the same weight for the convolutional filter layer. 
- A ReLU was added after each layer to add non-linearities to the network which mimics the boundaries between the classes in the 32x32 dimensional space
- Max-pooling is added after the convolutional layers to intelligently choose the best of the features without losing much information. This is what can be called a compression operation.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet was the baseline for Traffic Sign Classification
* Why did you believe it would be relevant to the traffic sign application?
I believe it is relevant for this project because of the similarities between the hand written digit classification - which is what it was originally designed for - and this project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The dataset has a lot of intraclass similarities. A lot of the signs have similar shapes and very similar contents inside of the sign boundaries. Despite this the accuracies in training, validation and testing (reported in the beginning of the section) indicate the the model works.
 

### Test a Model on New Images

#### 1.Acquiring New Images

I found several traffic signs on the web. As a preprocessing step, I cropped the images so that the relative aspect ratio between the final image and the traffic sign is reasonable enough to mimic the training set. A challenge with the internet images is that a lot of them contain watermarks which aren't present in the dataset presented to us. This might throw the classifier off on occasions. Also, this project attempts to build a classifier and not a detector which is a whole other undertaking involving regression on 3 values(bounding box center, width and height). Below is a collage of all images downloaded:

![alt_text][image7]

Among those I chose the following as the five images for the test set.

![alt text][image8] 
![alt text][image9] 
![alt text][image10] 
![alt text][image11] 
![alt text][image12]

Comments on chosen images:
* Image 1 - Class 1  - Speed limit (30km/h) - Class 1 and Class 4 are very similar and could be challenging
* Image 2 - Class 4  - Speed limit (70km/h) - Class 1 and Class 4 are very similar and could be challenging 
* Image 3 - Class 9  - No passing - This might be an easier image to classify
* Image 4 - Class 11 - Right-of-way at the next intersection - Might be easier to classify
* Image 5 - Class 17 - No entry - This image is very similar to the Stop sign (Class 14) and might be challenging

#### 2. Performance on New Images

Subset performance: The model was able to correctly guess 4 among the 5 traffic signs, which gives an accuracy of 80%. This pales in comparison to the accuracy on the test set of 95.511%. 


| Image			                              | Prediction	            					          | 
| ------------------------------------- | ------------------------------------- | 
| Speed limit (30km/h)      		          | Speed limit (30km/h)  									       | 
| Speed limit (70km/h)    			           | Speed limit (70km/h) 										       |
| No passing					                       | End of no passing											          |
| Right-of-way at the next intersection | Right-of-way at the next intersection	|
| No entry 			                          | No entry       							                |


Overall performance on the 43 downloaded images: 60.465% accurate. One reason for the inaccuracies is that many of the images that I downloaded are 'computer generated model images' which don't look anything like real images. This might have thrown off the classifier, however I believe the performance on real world images might be better.

#### 3. Model Certainty - Softmax Probabilities
These are the Softmax probabilities obtained for the test predictions. From the below table, it is clear that the model is very certain about the predictions for these classes.

| Input class | c1 | c2 | c3 | c4 | c5 | SM1 | SM2 | SM3 | SM4 | SM5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 1  | 5  | 2  | 0  | 6  | 1.00000e+00  | 3.58568e-14 | 5.69922e-15 | 4.61392e-16 | 4.56762e-16 |
| 4  | 4  | 0  | 29 | 24 | 1  | 1.00000e+00  | 5.41072e-08 | 2.08294e-10 | 7.76585e-11 | 5.15568e-11 |
| 9  | 41 | 42 | 3  | 19 | 9  | 0.32976      | 0.2326      | 0.17779     | 0.13344     | 0.07231     |
| 11 | 11 | 21 | 31 | 2  | 30 | 1.00000e+00  | 1.37943e-14 | 5.41940e-18 | 4.02765e-18 | 3.55944e-18 |
| 17 | 17 | 32 | 29 | 0  | 42 | 1.00000e+00  | 2.96605e-16 | 3.16787e-17 | 9.17878e-19 | 4.55823e-20 |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
The conv1 and conv2 layers have the following outputs as shown in the image below. The first layer shows that the outputs are fairly similar to the traffic sign provided to the input. However, the second layer seems to have broken it down into finer features.

![alt text][image13]
![alt text][image14]


