# I. Introduction and Scope

This project hopes to give a practical example of what a simple but non-trivial deep learning computer vision project might look like from start to finish. We chose the topic of real-time face tracking. In the end we'll be able to overlay a video recording or webcam feed of one person with his or her face's bounding box.

- TODO: Finale Animation von Gesicht mit BBox (Endlosgif)

We avoid getting technical about mathematics and the mechanisms behind the training process of a neural network and, instead, focus on the macro steps for the sake of clarity.

# II. Data Preparation

## II.1 Collecting Image Data

Face detection (and object detection in general) is a supervised learning task. These types of problems are solved by providing an algorithm with input samples, paired with their respective desired labels. In our case, the sample set comprises images of our face and the labels correspond to the bounding boxes' coordinates. Given these inputs, the algorithm iteratively solves a proxy optimization problem, minimizing a measure of difference between the algorithm's current best guess of a face's bounding box and the image's label, i.e. the correct position of the bounding box.

For this process to work we need data, lots and lots of data. For simple computer vision projects and small models between 10k and 100k labelled samples are often enough to solve the problem sufficiently well. While there are plenty of free dataset of faces available online (e.g. [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [LFW](http://vis-www.cs.umass.edu/lfw/)) for this project we're taking the pedestrian's approach and collect and label the data manually.

To this end, I just walked around in the office, my laptop in my hands, repeatedly taking images of my face via the webcam every once in a while. Having pictures in different locations, at different distances, with different objects in the background and varying lighting conditions is helpful for the algorithm to pick up what is actually the common denominator in each of these images - the human face.

- TODO: Beispielbilder einfügen

In total I captured about 400 images at my webcam's native resolution of 1280x720 px. 300 of those images do contain my face while the remaining 100 show no face at all. With a little trickery these relatively small quantities of images can be inflated to be sufficient for our purposes.

<!--There are a lot of premade datasets available online that may or may not be applicaple to one's problem, in this we are gonna assume that there is no such dataset and we will create our own data "from scratch" so to speak.

In all of this, it can't be overstated how important a lot of high quality data is for Deep Learning projects in general. If faced with the decision between spending time to gather more data or custom engineering something in the Network or the Training process, the former is almost always the right choice. 

Now that we have our example inputs, we need to concern ourselves with the other part of the equation, the expected output for each input. This leads us directly to: -->


## II.2 Drawing Bounding Boxes

The ultimate goal of our project is to predict a bounding box around an image of a face and hence we need to provide our model with examples of exactly that. Thus, it is now the time to create the labels for our new dataset, i.e. to draw bounding boxes around the face we've captured.

The annotation of image data, such as drawing a bounding box, are done using dedicated tools. For our purposes, we'll use [LabelMe](https://github.com/wkentaro/labelme), a well established workhorse application. LabelMe loads a directory of images and allows to overlay each image with a rectangle, or, more generally, a polygon.

- TODO: Bilder von LabelMe bei der Arbeit einfügen

Note that the original images are not modified in any way. LabelMe parameterizes the bounding box rectangle as a tuple of its top left and bottom right corner coordinates and writes these four numbers to a JSON file.

- TODO: Side-by-side Vergleich von Bild mit BBox und JSON Annotation.

For the images without faces we choose a degenerate rectangle of zero width and height by convention.

Now that we have a few hundred labelled samples two final steps of preprocessing remain. Usually these are performed on-the-fly as part of the training pipeline, but for the sake of clarity we'll do them explicitly.

## II.3 Augmenting our Data

As mentioned above, for relatively simple problems a good rule of thumb to start with are some 10k samples. The 400 images and labels we have acquired might seem like a lot (and manually labelling them _is_ a lot of work) but we are still off by two orders of magnitude.

Here, we're going to employ a standard method, which we referred to as a "trick" earlier in this article: We are going to inflate the number of samples by the application of simple image transformations. Flips, rotations, crops, RGB shifts and gamma corrections can be used to derive "new" images from the originals. This practice is called *data augmentation*.

- TODO: Grund für Crops erklären
- TODO: Erläutern wie mit den Bounding Boxes verfahren wird.

In our project we're going to create 60 derivative images per original sample which leaves us with a respectable dataset of 24k samples.

- TODO: Beispielbilder zeigen

<!--This is where Augmentation comes in. We are going to take in one image at a time, apply random crops (to a size of 720x720px), flips, RGB shifts, gamma corrections etc. to it and in this way create many examples from a single one. If we do this for every Frame, let's say 60 times, we can create 24.400 training examples from our original 400! That should do the trick. It should be mentioned that data augmentation is in no way a magic bullet that can alleviate the need for large datasets, but it is a useful tool and often increases the robustness of neural networks trained with randomly augmented data. -->

## II.4 Downscaling the Images

In the final process step we rescale the dimensions of our images to one tenth of their original size. This step reduces the total volume of input data by a factor of 100 and is necessary to be able to perform the training process on regular commodity hardware (i.e. your laptop) in an acceptable amount of time.

- TODO: Skalierte Beispiele zeigen um zu verifizieren, dass kleine Auflösung nachwievor genug ist um Gesichter gut zu erkennen.

<!--The second step is to resize all of our frames. Our original Images were 1280x720px, 921.600 pixels in total. This would be way too much for everyday hardware to run a neural network on. We already cropped our images to be 720x720px and now we are going to resize these pictures to 120x120px, a far more managable 14.400 pixels total.-->

## II.5 Recap

Let's take a step back and have a look at what we have accomplished so far: We have
1. manually created a small dataset from scratch by capturing images and
2. labelling them and
3. we have augmented and
4. rescaled our dataset.

This is nothing to sneeze at! In practice, getting to this point and finishing the data preparation is often 80% of the total work. In machine learning, just as anywhere else, the principle of garbage in, garbage out applies, so all of the above efforts amount to time certainly well spent.


# III Training

# III.1 The Loss Function

The next step is the choice of a *loss function*, also called the *training criterion*, which defines how the difference between the network's guess of a bounding box and the image's actual label shall be expressed numerically. For this project we choose the squared error between the corner coordinates x and y values.

$$ L = (x_{TL}-\hat{x}_{TL})^2 + (y_{TL}-\hat{y}_{TL})^2 + (x_{BR}-\hat{x}_{BR})^2 + (y_{BR}- \hat{y}_{BR})^2$$

- TODO: Konfidenz / Klassifikationslos erläutern

<!--
In adition to that, our network will also predict a single number between 0 and 1, this number will indicate, whether or not the network thinks that there is a face in the frame at all.

For this we will use a run of the mill BinaryCrossEntropy Loss. The "Box Loss" and the "Class Loss" are added together and returned as the final loss value.
-->

# III.2 Choice of Network Architecture

In the past, convolutional neural networks (abbreviated *CNN* or *convnet*), a special kind of neural network architecture, have proven themselves to be very well suited for computer vision tasks. To solve our problem we're going to use a pretrained convnet and adjust it to our specific problem. This is a common practice referred to as *transfer learning* which we refer to in another blog post.

Specifically, we're going to use the [VGG16 model](https://arxiv.org/abs/1409.1556), pretrained on [ImageNet](https://www.image-net.org/), a large image classification dataset. This provides the added benefit of not having to run the training process for too long before acceptable results are achieved.

This has the advantage that we don't need to train the Network for as long and will likely still get pretty good results. Finally the Network has 2 Output Layers. One for our bounding box prediction and one for our binary "face present" truth value.

## III.2 Training our Neural Network

In practice, this step is often just a bunch of trial and error. Since this isn't that complex of a problem and our solution isn't required to be perfect, I just trained this Network once on my personal Laptop (M1 Macbook Air) for about 24 hours and got pretty good results.

- TODO: Trainingsverlauf einblenden

# IV Done

Now are done and can use our model to implement a realtime face tracking application.

- TODO: Finale BBox Animation

- TODO: "I" vs. "we"
