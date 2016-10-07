# udacity-SDC-baseline

Scripts to evaluate some baseline DNN models on [udacity/self-driving-car](https://github.com/udacity/self-driving-car)

## Dataset
- Download from data links in [github](https://github.com/udacity/self-driving-car)
- I have only evaluated on the smaller set

## Processing
### Reading from ROSBAG

Use [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader/blob/master/README.md). Successful extraction will generate a `dataset` folder structured like this:
```
camera.csv  center/  left/  right/  steering.csv
```

### Data Generator

The time series of camera images and steerings are divided into buckets of 0.1 second long. Within each time
bucket, there might be multiple images and steering recorded depending on hardware fps rate. For the same time bucket, I randomly select the images (which should be similiar) as inputs and the mean of steering angles (if more than one) as targets.

I haven't used any encoding such as [mu-law](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) or discretizatoin for outputs.

I didn't exclude any data, even though it might help to exclude scenarios like traffic-light-waiting and turning, for learning staying on the lane. 


### Train and Test Split

I hardcoded the last 30s (roughly) for testing. It may not be a good choice because there seems to be some long traffic-light-waiting in the last 30 seconds. This can be changed in `test_model.py`. 

## Models

I have been playing with different models, and their input/output processing, mainly using [Keras library](https://keras.io/), which supports both theano and tensorflow at backend. 

For VGG bottleneck features, please refer to this [article](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). For CNN for SDC, please refer to [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf).

RMSE is used to evaluate models' performances. It is useful in training models, but may be irrelevant to the real performances of a self driving car, (e.g, measured by autonomy value as discussed in the Nvidia paper) 

## Examples

### 1. ConvNet-like trained from scratch

```
python test_model.py --dataset dataset/ --model cnn --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

```
### 2. Pretrained VGG16 bottleneck + fully connected layers

```
python test_model.py --dataset dataset/ --model vgg16 --resized-image-width 224 --resized-image-height 224 --nb-epoch 4

```

## Observations

- Given the data, it is difficult to train a driving model that "stays on the lane". For example, when tested
on the last 30 seconds (with a traffic-light-waiting), the `cnn` model tends to drive wiggly to avoid the cars in the front? But actually the cars have stopped for the traffic light. Not sure if this is caused by the "turning examples" in the training data. It shows how hard driving could be without knowing the car speed or seeing any road marks.
- It looks like choosing the "right" training data is crucial.
- RMSE may not be a good choice even for training purpose - the ditribution of steering radians is not close to Gaussian.  

## Resources on SDC
- [Udacity open source SDC](https://github.com/udacity/self-driving-car)
- [commaai research](https://github.com/commaai/research)
- [Tensorflow Implementation of Nividia Paper](https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow)
- [deepdrive](http://deepdrive.io/)
- [DanDoesData SDC playlist](https://www.youtube.com/playlist?list=PLFxrZqbLojdJIMff4WzpXpk1G2pEa3jAb)
- [General resources](https://gist.github.com/bigsnarfdude/a0702a88b122be81cb1f4edb55229a7b)

## Disclaimer
It's a baseline to show how far it goes without any advanced data processing and model optimization. In a way it demostrates where the problem can be challenging. 

## TODO
Pick a clean set to train a specific model for staying on lanes, maybe using RNN?