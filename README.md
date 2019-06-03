# genetic-classifier

A vanilla genetic toy classifier which can be trained to classify images from for example the MNIST or Fashion-MNIST dataset. This project shows, that despite the fact that the model is really small (7840 weights) a relatively high accuracy can be achieved.

Genetic optimization is much slower compared to other optimization techniques such as gradient descent. However it is possible to achieve about 90 % test accuracy for the MNIST and about 70 % test accuracy for the Fashion-MNIST dataset.

Despite the poor performance, genetic algorithms are fun to play with and very easy to implement. This implementation comes with three key hyperparameters. `population` which represents the size of the population, `mut_loc` and `mut_glb` which represents the local and global mutation rate, respectively.
