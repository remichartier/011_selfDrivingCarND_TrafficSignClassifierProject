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
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
- Link to my [project code](https://github.com/remichartier/011_selfDrivingCarND_TrafficSignClassifierProject/blob/master/Traffic_Sign_Classifier_v13.ipynb)

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
    
  
  
  
  
- I did not take any further pre-processing steps as I wanted to see first the results with this normalization, and then come back if I needed more improvements in the classifier pipeline results.
- Just for testing, I also implemented a grayscale conversion and I modified the code towards that goal (notebook chapter **"`Try Gray Scale conversion`"**.
  - however I did not find any improvements in the classifier pipeline performance using grayscale conversion so this step is currently disabled via a flag in the code : `is_converted_gray`
  - Depending on this flag, I adjust the channels input of the convolution pipeline, setting 3 channels for color images and 1 for grayscale images.

Here is an example of a traffic sign image before and after grayscaling, which I am able to get whenever flag `is_converted_gray` is set to `True`.

![alt text][image2]

I have not explored other ways yet due to time constraints and the need to move forward in the ND lectures.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

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


