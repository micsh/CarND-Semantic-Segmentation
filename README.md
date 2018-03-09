# Semantic Segmentation
## Introduction
The goal of this project is to construct a fully convolutional network (FCN) based on the VGG-16 image classifier architecture for performing semantic segmentation to identify drivable road area from a car's dashcam image on the KITTI data set.

## Architecture
A pre-trained VGG-16 network was converted to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution and setting the depth equal to the number of desired classes (in this case, two: road and not-road). Performance is improved through the use of skip connections, performing 1x1 convolutions on previous VGG layers (in this case, layers 3 and 4) and adding them element-wise to upsampled (through transposed convolution) lower-level layers (i.e. the 1x1-convolved layer 7 is upsampled before being added to the 1x1-convolved layer 4). Each convolution and transpose convolution layer includes a kernel initializer and regularizer.

This is work is based on the paper by [Shelhamer, Long
and Darrell](https://arxiv.org/pdf/1605.06211.pdf).

### Optimizer
The loss function for the network is cross-entropy, and an Adam optimizer is used.

### Training & Results
The hyperparameters used for training are:

    keep_prob: 0.8
    learning_rate: 0.0009

runs with increasing batch sizes:

run 1 (50 epochs):
    
    batch_size: 3, epochs: 15
    batch_size: 5, epochs: 5
    batch_size: 10, epochs: 10
    batch_size: 20, epochs: 20

    Final cost: 0.022

run 2 (60 epochs):

    batch_size: 3, epochs: 10
    batch_size: 3, epochs: 10
    batch_size: 5, epochs: 10
    batch_size: 10, epochs: 10
    batch_size: 20, epochs: 20

    Final cost: 0.019

run 3 (60 epochs):

    batch_size: 3, epochs: 10
    batch_size: 3, epochs: 10
    batch_size: 5, epochs: 10
    batch_size: 8, epochs: 10
    batch_size: 13, epochs: 10
    batch_size: 21, epochs: 10

    Final cost: 0.009
    (Image results for this run can be found in [runs](/runs/1520372357.4342237/))


## Examples
Here we can see a few example of the segmentation results:

![1](/runs/1520372357.4342237/umm_000002.png)
![1](/runs/1520372357.4342237/um_000003.png)
![1](/runs/1520372357.4342237/um_000005.png)
![1](/runs/1520372357.4342237/um_000007.png)
![1](/runs/1520372357.4342237/um_000013.png)
![1](/runs/1520372357.4342237/um_000015.png)
![1](/runs/1520372357.4342237/um_000032.png)
![1](/runs/1520372357.4342237/um_000040.png)
![1](/runs/1520372357.4342237/um_000062.png)
![1](/runs/1520372357.4342237/umm_000008.png)
![1](/runs/1520372357.4342237/umm_000014.png)
![1](/runs/1520372357.4342237/umm_000024.png)
![1](/runs/1520372357.4342237/umm_000028.png)
![1](/runs/1520372357.4342237/umm_000032.png)
![1](/runs/1520372357.4342237/umm_000035.png)
![1](/runs/1520372357.4342237/uu_000002.png)
![1](/runs/1520372357.4342237/uu_000004.png)
![1](/runs/1520372357.4342237/uu_000013.png)
![1](/runs/1520372357.4342237/uu_000017.png)
![1](/runs/1520372357.4342237/uu_000023.png)
![1](/runs/1520372357.4342237/uu_000027.png)
![1](/runs/1520372357.4342237/uu_000049.png)
![1](/runs/1520372357.4342237/uu_000067.png)
![1](/runs/1520372357.4342237/uu_000095.png)

---
Original README content

---

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
