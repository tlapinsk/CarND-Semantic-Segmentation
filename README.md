# Semantic Segmentation Project
In this project, I used Python to create an advanced neural network that classified road versus non-road sections of dataset.

## Project Info
To see the implementation please go to `main.py` in the CarND-Semantic-Segmentation repository.

## Overview
The goal of this project was to create a a fully convolutional neural network based on the VGG-16 image classifier architecture to perform semantic segmentation (road versus not road in images).

## Architecture
As recommended, I used a pre-trained VGG-16 network that was converted to a fully convolutional neural network. This was achieved by replacing the final fully connected layer with a 1x1 convolution and setting the depth to the desired number of classes.

The use of skip connections, 1x1 convolutions on prior layers, and finally upsampling (to output the same size image as the input) further helped me achieve great results. Lastly, as recommended, I also implemented a kernal initializer and regularizer. These helped tremendously.

The main section of the code where this was all achieved can be seen below:

```
# 1x1 convolution of vgg layer 7
layer7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# upsample
layer4a_in = tf.layers.conv2d_transpose(layer7_out, num_classes, 4, strides=(2, 2), padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# 1x1 convolution of vgg layer 4
layer4b_in = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# skip connection
layer4a_out = tf.add(layer4a_in, layer4b_in)

# upsample
layer3a_in = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4, strides=(2, 2), padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# 1x1 convolution of vgg layer 3
layer3b_in = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# skip connection
layer3a_out = tf.add(layer3a_in, layer3b_in)

# upsample
nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16, strides=(8, 8), padding= 'same', 
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
```

### Optimizer
Loss function: Cross-entropy
Optimizer: Adam optimizer

### Training
Hyperparameters below:

|  Input          |    MSE   |
|  -----          |  ------- |
|  keep_prob      |  0.5     |
|  learning_rate  |  0.0009  |
|  epochs         |  25      |
|  batch_size     |  5       |


### Results
Check out samples below:

![Sample 1](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000015.png?raw=true "Sample 1")
![Sample 2](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000008.png?raw=true "Sample 2")
![Sample 3](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000017.png?raw=true "Sample 3")
![Sample 4](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/umm_000034.png?raw=true "Sample 4")
![Sample 5](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/umm_000078.png?raw=true "Sample 5")
![Sample 6](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/uu_000023.png?raw=true "Sample 6")
![Sample 7](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/uu_000090.png?raw=true "Sample 7")
![Sample 8](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/uu_000006.png?raw=true "Sample 8")
![Sample 9](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000080.png?raw=true "Sample 9")
![Sample 10](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000063.png?raw=true "Sample 10")
![Sample 11](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000026.png?raw=true "Sample 11")
![Sample 12](https://github.com/tlapinsk/CarND-Semantic-Segmentation/blob/master/output/um_000005.png?raw=true "Sample 12")

## Resources
Shoutout to the tutorials provided by Udacity on fully convolutional neural networks. The Semantic Segmentation Walkthrough was especially helpful to get me started. It didn't take much to complete the project after watching the Walkthrough a few times. Below are resources and helpful links that I used to complete this project:

- [Semantic Segmentation Project Walkthrough](https://youtu.be/5g9sZIwGubk)
- [AWS AMI setup](https://discussions.udacity.com/t/aws-ami-setup-operation-timed-out-error/349017/12)
- [Setup and use Jupyter (IPython) Notebooks on AWS](https://towardsdatascience.com/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5)
- [Update TensorFlow](https://stackoverflow.com/questions/42574476/update-tensorflow)
- [Elapsed time in Python](https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module)
- [Round to nearest integer in Python](https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer)

# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
