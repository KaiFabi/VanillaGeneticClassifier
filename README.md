# VanillaGeneticClassifier

This is a vanilla genetic classifier which can be trained to classify images from for example the MNIST or Fashion-MNIST dataset. This project shows, that despite the fact that the model is really small (7840 weights) a relatively high accuracy can be achieved.

Genetic optimization is much slower compared to other optimization techniques such as gradient descent. However it is possible to achieve about 90 % test accuracy for the MNIST and about 80 % test accuracy for the Fashion-MNIST dataset.

Despite the poor performance, genetic algorithms are fun to play with and very easy to implement. This implementation comes with three key hyperparameters. `population` which represents the size of the population, `mut_loc` and `mut_glb` which represents the local and global mutation rate, respectively. The global mutation rate determine which proportion of the weights are being mutated. The local mutation rate determines the magnitued of change of the selected weight. 

This genetic optimizer takes the following parameters for training: population size `population`, `batch_size`, local mutation rate `mut_loc`, global mutation rate `mut_glb`, and number of `epochs`. Here, for the demonstration the following training parameters were used:

```python
# Training parameters
population = 8
batch_size = 64
mut_loc = 0.01
mut_glb = 0.01
epochs = 100
````

The genetic classifier was trained for 100 epochs, with a population size of 8, and a mini batch size of 64 images. Both, the local and global mutation rate was set to 0.01. The mutation rates are similar to the learning rate of any other recursive optimizer such as the gradient descent optimizer. The classifier trained on the MNIST dataset achieved a test accuracy of 88.05 % while the classifier trained on the more complex Fashion-MNIST dataset achieved a test accuracy of 79.94 %. Not bad for a genetic optimizer with only 7840 weights.

The resulting graphs show the loss and accuracy for the evaluation dataset of the MNIST and Fashion-MNIST dataset for differnt combinations of local and global mutation rates.

**MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/mnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/mnist_eval_accuracy.png" height="320">
</div>

**Fashion-MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/fmnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/fmnist_eval_accuracy.png" height="320">
</div>


In order to get a better understanding of what the genetic optimizer has learned during its training, the weights can be visualized. It turns out that weights are learned that look like handwritten digits themselves. Interestingly, the weights learned tend to be an average of the training data. The upper and lower rows show the weights learned and the mean values of the training data for each class, respectively.

<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/gml_weights_mnist.png" width="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/gml_weights_fmnist.png" width="320">
  
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/gml_mnist_mean.png" width="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/gml_fmnist_mean.png" width="320">
</div>

**Hyperparameter optimization**

There are several hyperparameters that influence the genetic classifier's performance. Here, the mutation rates will be considered more closely. A good choice of mutation rates is essential for a genetic algorithm. Good mutation rates can be found, for example, by trial and error, or by a brute force grid search approach. Here are two example for the MNIST dataset.

Alternatively, it is possible to build another genetic hyperparameter optimization algorithm on top of the genetic classifier algorithm.

Especially the last two approaches to find a good set of mutation rates can be very time consuming.

Using genetic algorithms to optimize a large number of weights is not very efficient. Nevertheless, these algorithms are great to play around with. 
