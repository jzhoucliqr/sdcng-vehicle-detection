## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/box_example.png
[image5]: ./examples/combine_10_frame.png
[image10]: ./examples/box_example_hog.png
[image11]: ./examples/box_example_dl.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first cell .

I'm using the training dataset provided by the course, together with extracted images from the udacity published dataset. I extracted the non-car image from several frames of the project video, so in total I got 21652 car images and 19963 non-car images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choose the following parameters considering both the accuracy and speed: 

```
Color space: YCrCb
orientations: 9
pixels_per_cell: (16,16)
cells_per_block=(1, 1)
Hog_channel = 'ALL'
spatial_size = (16,16)
spatial_feat = True
hist_feat = False
hog_feat = True
```

I choose `pixels_per_cell=16` which speeds up the training and prediction quite a lot, at the same time does not lose too much accuracy. 

The color space `YCrCb ` actually I don't see too much difference from other spaces. 

I didn't use color histogram feature because I see that introduce a lot of false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I combine all the features extracted (All three channels Hog feature together with spatial bin features) and train a linear SVM classifier. 

I use the GridSearchCV() to find the optimal value for different parameters.

Initially I tried to search between both linear kernel and rbf kernel, together with different values for C and gamma. But the training takes too long to finish, so in the end I just choose the linear kernel, and only use GridSearchCV() to find the optimal value for C between [0.1, 1, 10]. This ends with a C=0.1 be the optimal value.

The code for training is in hog_train.py. It will save the final model in a file, then in the python notebook, it'll load the model from file. So this way I can start/stop my notebook without re-train my model.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search algorithm is inside the function `find_cars_half_hog()`.

I tried to follow the sliding window search solution from the lecture.  

I tried multiple scales combination, and in the end choose to use scales = [1, 1.2, 1.5, 2] consider both the accuracy and speed. 

I tried multiple window size, from 64 to 96, to 128, and in the end still use the window size of 64, and take `cell_size * 2` as step size.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  

Here are some example images:

![alt text][image4]

In order to speed up the search for the project video, I only search the right-lower part the image, with the search area like this: 

```
xstart = 640
xstop = 1280
ystart = 400
ystop = 656
```

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

I tried to use both SVM with hog features, and also transfer learning with imagenet as base model, here are links to both outputs:


Transfer learning: [https://youtu.be/bhp3WSNuMMQ](https://youtu.be/bhp3WSNuMMQ)

SVM with hog:  [https://youtu.be/dbUsXY2Wgz8](https://youtu.be/dbUsXY2Wgz8)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used two level of heatmap with threshold to do filtering. The fist level is within each frame for multiple matches by the sliding window search. I use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, and treat them as a car. 

Then I combine the results across 10 frames, and do the second level of heatmap with threshold. I get the final output as result of the last frame. 
 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps. 

The final row is the final output combine all the previous 10 frames.

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. speed vs accuracy: 
A small step size with a small hog cell increase the accuracy, with the cost of speed. A larger step size and hog cell speed up the training/predict of algorithm, but it'll decrease the accuracy. So the tradeoff between speed and accuracy need more tuning.


2. heatmap threshold:
Another tradeoff is the heatmap threshold. If the threshold too low, then too many false positive, on the other hand, if threshold too high, then recall is low. So again this parameter needs tuning.

3. feature vector:
I only used the hog feature together with space spin, without the color histogram. Because I found it to introduce way too many false positive.

4. sliding window search:
The sliding window search I used is a very basic one, with a fixed window size. Ideally the window size and step size can be automatically adjusted based on the position. The window size and step size can increase when it get closer to the bottom of the image. I tried multiple ways to get an adjusted window size and step size, including mannually construct rules, and fit a poly nomial based on the size and position from the udacity dataset. But neither of these techniques give a good result. So in the end I only use the very basic one.

	

---

### Using deep learning 

Since this is a typical image classification problem, I also tried a deep learning solution with transfer learning to train the classifier. 

The code is inside dl_train.py. I followed an [online tutorial](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recogni    tion-94b0b02444f2) to use transfer learning with keras to get a model from imagenet. 

I replaced the top tier of the original model with a new fully connected layer with only 2 classes. Then I also tried fine tuning with the last several layers. 

With the same training dataset, the classification accuracy is pretty high. So for each frame there is almost no false positive. 

But the problem is recall is a bit low. Part of the reason might be my sliding window algorithm does not do a good job. 

Here is an example with the same image using both svm and deep learning method, with scale of [1, 1.2, 1.5, 2]. 

And this is the final video using deep learning with only scale [1,5, 2]. (it takes a long time for the pipeline to finish, so I didn't use the four scale as the SVM classifier)

Here is the [link to video result with deep learing.](https://youtu.be/bhp3WSNuMMQ)


![alt text][image10]

![alt text][image11]


 



 

