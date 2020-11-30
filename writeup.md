# **Traffic Sign Recognition** 
## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/0001_labelHistogram.png "Visualization"
[image2]: ./writeup_images/0002_gray_conversion.png "Grayscaling"
[image3]: ./writeup_images/0003_pixelValuesNormalization.png "Normalization"
[image4]: ./webImages/01.png "Traffic Sign 1"
[image5]: ./webImages/02.png "Traffic Sign 2"
[image6]: ./webImages/03.png "Traffic Sign 3"
[image7]: ./webImages/04.png "Traffic Sign 4"
[image8]: ./webImages/05.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
- Link to my [project code](https://github.com/remichartier/011_selfDrivingCarND_TrafficSignClassifierProject/blob/master/Traffic_Sign_Classifier_v17.ipynb) (Jupyter Notebook)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

- This is done using code in chapter **"`Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas`"** from my notebook.
- I used the numpy library to calculate summary statistics of the traffic signs data set :
  - The size of training set is 34799
  - The size of the validation set is 4410
  - The size of test set is 12630
  - The shape of a traffic sign image is (32, 32, 3)
  - The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set (Notebook chapter **"`Histogram of image labels y_train`"**). It is a bar chart showing how the data is split across different German Traffic Sign labels.
We can see that repartition accross labels is inequal, therefore some Traffic signs have more images to train the classifier (like 1000 to 2000 images) while some Traffic signs have less images (like under 500/250). 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The normalization steps I took are coded in my notebook in chapter **"`Normalization for data 0 mean and equal variant`"**
- For preprocessing, the first take I took was to normalize the data/images.
  - Normalization goal is to transform the data to have 0 mean and equal variant.
  - It helps convolution algorithm and backpropagation of Gradient Descent to converge faster towards minimizing the loss and finding faster convolution + classifier weights and bias to minimize the loss/error function.
  - I applied simple normalization in a first step consisting on doing the following on each image pixel : (pixel -128)/128. This is a quick way to approximately normalize the data. 
  - Just to compare before normalization and after normalization, I printed out the pixel average, Image pixel standard deviations vs average, minimum and maximum values of pixels for the X_train dataset.
    - Before normalisation :
    ```
      - Values average/mean : 82.68
      - Values average standard deviation : 66.82
      - Mininum value : 0.00
      - Maximum value : 255.00
    ```
    - After normalisation :
    ```
      - Values average/mean : -0.35
      - Values average standard deviation : 0.52
      - Mininum value : -1.00
      - Maximum value : 0.99
    ```
    - So we see that the approximate normalization applied ((pixel -128)/128) still does a good job relatively over the X_train data  to reach a zero mean (82.68 --> -0.35). Not sure it really improves the equalization of the data variance, but at least we slide the dataset more towards a zero mean to help in future steps on gradient descent steps to minimize the loss function.
    - Below is a visualization of the X_train pixel values before and after normalization, to make sure pixel values after normalization are within [-1;+1] and more around mean Zero.
      - Note : this is not a view on the whole pixel values, this is only representing 200 000 pixel values. Visualizing all pixel values would be too long to compute and display.
  
![alt text][image3]  
  
  
- I did not take any further pre-processing steps as I wanted to see first the results with this normalization, and then come back if I needed more improvements in the classifier pipeline results.
- Just for testing, I also implemented a grayscale conversion and I modified the code towards that goal (notebook chapter **"`Try Gray Scale conversion`"**.
  - however I did not find any improvements in the classifier pipeline performance using grayscale conversion so this step is currently disabled via a flag in the code : `is_converted_gray`
  - Depending on this flag, I adjust the channels input of the convolution pipeline, setting 3 channels for color images and 1 for grayscale images.

Here is an example of a traffic sign image before and after grayscaling, which I am able to get whenever flag `is_converted_gray` is set to `True`.

![alt text][image2]

I have not explored other ways yet due to time constraints and the need to move forward in the course lectures.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

- My Model code is covered in my notebook in chapter : **"`Model Architecture`"**
- It is based on LeNet Model seen in the course, adapted to input images (3 channels when RGB or BGR, 1 channel if taking Grayscale images like in the course) and adapted to classify 43 classes (43 different German Traffic Sign) instead of 10 classes from LeNet MNIST model.
- It consists of the following layers:

| Layer         		    |     Description	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Input         		             | 32x32x3 RGB image (or 32x32x1 if grayscale). | 
| Layer 1 : Convolution 01 (5x5) | 1x1 stride, VALID padding, Input = 32x32x3 or x1 if Grayscale. Output = 28x28x6.	|
| Activation					           | RELU											|
| Max pooling      	             | 2x2 stride,  Input = 28x28x6. Output = 14x14x6. |
| Layer 2 : Convolution 02 (5x5) | 1x1 stride, VALID padding, outputs 10x10x16. |
| Activation 					           | RELU											|
| Max pooling	      	           | 2x2 stride,  Input = 10x10x16. Output = 5x5x16. |
| Flatten                        | Input = 5x5x16. Output = 400. |
| Layer 3: Fully Connected		   | Input = 400. Output = 120.  |
| Activation 					           | RELU											|
|	Dropout 					             | keep_prob = 0.75 for training, 1.0 for validation/test, to prevent overfitting. |
| Layer 4: Fully Connected		   | Input = 120. Output = 84. |
| Activation 					           | RELU											|
|	Dropout					               | keep_prob = 0.75 for training, 1.0 for validation/test, to prevent overfitting. |
| Layer 5: Fully Connected		   | Input = 84. Output = 43.       									|
|	Dropout					               | keep_prob = 0.75 for training, 1.0 for validation/test, to prevent overfitting. |

- Note : 
  1. The final Softmax function to finalize the classification and provide probability of each image for the 43 classes (German Traffic Signs) is called later via TensorFlow function `tf.nn.softmax_cross_entropy_with_logits()`
  2. LeNet() modified to add input parameter `in_channels` to specify number of channels of input images, like 3 for RGB/BRG images, 1 for grayscale images, or it could be 4 for RGBA images.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
- This code is covered in my notebook chapter : **"`Training Pipeline`"** as well as **"`Train the Model`"**.
- To train my model, I took LeNet TensorFlow algorithm previously applied to MNIST, and re-used it entirely for this German Traffic Sign Classifier.
- I ran it "as is" to check the first validation accuracy I would reach with this reference model, ie I kept : 
  - The learning rate "'rate = 0.001'"
  - Using cross entropy to calculate loss.
  - Kept the Adam optimizer for Gradient Descent.
  - Batch size of 128.
  - Number of epochs at 10.
  - Other hyperparameters like for initializing Weights before running Gradient Descent kept at `mean = 0, stddev = 0.1`. 
  
  I recognize that I could have played with all those parameters/choices to improve validation accuracy performance. However, so far, the only parameter I played with in order to reach the minimul test accuracy target of 0.93 was to change the number of epochs from 10 to 20.
  
  But I recognize that if I want to explore further improvements, all the other choices/parameters would be available for me to try getting more accuracy improvements, especially optimizing the learning rate, exploring the optimizer other options, changing the Batch size, doing more testing with number of epochs and weight initialization parameters, extensive range of options on convolution networks kernel sizes, number of convolutions, activation functions choices, pooling functions, number of fully connected layers and sizes. List of other options is very long and opening for many avenues to either improve and worsen the classifier performance results, so I chose to limit myself to changing parameters I could logically see would improve results and reach the minimum passing test accuracy of 0.93.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

- This code is covered in my notebook chapters **"`Model Evaluation`"** as well as **"`Train the Model`"** as well as **"`Evaluate the Model`"**.
- Training Set accuracy was not yet implemented in reference LeNet MNIST model I took as a reference, so I added few more lines to include it for the writeup.

My final model results were :
* training set accuracy : 0.934
* validation set accuracy of 0.947 
* test set accuracy of 0.939

Those numbers reflect numbers output from my notebook, for which I can give a snapshot exemple : 
```
Training...

EPOCH 1 ...Train Accuracy = 0.702...Validation Accuracy = 0.723
EPOCH 2 ...Train Accuracy = 0.849...Validation Accuracy = 0.845
EPOCH 3 ...Train Accuracy = 0.883...Validation Accuracy = 0.882
EPOCH 4 ...Train Accuracy = 0.893...Validation Accuracy = 0.891
EPOCH 5 ...Train Accuracy = 0.909...Validation Accuracy = 0.905
EPOCH 6 ...Train Accuracy = 0.915...Validation Accuracy = 0.918
EPOCH 7 ...Train Accuracy = 0.919...Validation Accuracy = 0.916
EPOCH 8 ...Train Accuracy = 0.925...Validation Accuracy = 0.921
EPOCH 9 ...Train Accuracy = 0.920...Validation Accuracy = 0.928
EPOCH 10 ...Train Accuracy = 0.929...Validation Accuracy = 0.926
EPOCH 11 ...Train Accuracy = 0.932...Validation Accuracy = 0.934
EPOCH 12 ...Train Accuracy = 0.933...Validation Accuracy = 0.933
EPOCH 13 ...Train Accuracy = 0.931...Validation Accuracy = 0.927
EPOCH 14 ...Train Accuracy = 0.936...Validation Accuracy = 0.934
EPOCH 15 ...Train Accuracy = 0.940...Validation Accuracy = 0.946
EPOCH 16 ...Train Accuracy = 0.935...Validation Accuracy = 0.936
EPOCH 17 ...Train Accuracy = 0.937...Validation Accuracy = 0.941
EPOCH 18 ...Train Accuracy = 0.940...Validation Accuracy = 0.939
EPOCH 19 ...Train Accuracy = 0.933...Validation Accuracy = 0.935
EPOCH 20 ...Train Accuracy = 0.934...Validation Accuracy = 0.947

Test Accuracy = 0.934
```

Iterative approach followed :
* I started with the TensorFlow course LeNet implementation for MNIST classification, just adapting input images channels (1-->3) and classes output (10-->43).
* I chose to keep with this LeNet implementation as it worked well for number images, and thought that it could be a good base to start classifying Traffic Sign images, with some adaptations if necessary.
* This reference model could only reach a Validation Accuracy of 0.87
* I then looked at further improvement paths, one of which being to try to regularize data in order to avoid overfitting. One of the most promizing approach is to use Dropout strategy, consisting of dropping out some logits in fully connected layers, to prevent neural network to fit systematically on those logits run after run and optimize over a different set of logits at every step.
  - I only activated this Dropout Regularization technique during trainings, not for validation or testing phases ie `keep_drop rate < 1` for trainings, and `keep_drop rate = 1` for validation or test.
    - I tested starting from `keep_drop = 0.5`, first after Layer 5 Fully Connected, then also after Layer 4 and Layer 3 as I saw consistent improvements in accuracy validation results.
    - I then increase the `keep_drop` rate from 0.5 to 0.75 as I also saw consistent improvements doing so on accuracy validation performance.
    - Those steps were enough to go from an accuracy validaton of 0.87 to 0.94/0.95.
* I also increased number of Epochs from 10 to 20 to keep accuracy validation around 0.94/0.95.

* At the end, I can see that train_accuracy and validation_accuracy numbers are pretty similar over all epochs starting Epoch # 2, so overfitting does not see to be a big issue with the model and parameters chosen. And test_accuracy stays as well in close range of validation and train accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

- All those images found on the web are pretty clear, in day light, with bright colors, so the model should be pretty good at classifying them.
- Last image (Pedestrian sign) is slightly oriented and rotated so it may be a challenge to classify.
- Second image (Priority Way) may also be challenging because it has the back of a square sign in its background, it may be a source of mis-classification.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


