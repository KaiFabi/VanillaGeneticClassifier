# VanillaGeneticClassifier

This is a vanilla genetic classifier which can be trained to classify images from for example the MNIST or Fashion-MNIST dataset. This project shows that despite the fact that the model is really small (7840 weights), a relatively high accuracy can be achieved.

Genetic optimization is much slower compared to other optimization techniques such as gradient descent. However it is possible to achieve about 90 % test accuracy for the MNIST and about 80 % test accuracy for the Fashion-MNIST dataset.

Despite the poor performance, genetic algorithms are fun to play with and very easy to implement. This implementation comes with three key hyperparameters: `population` which represents the size of the population, `mut_loc` and `mut_glb` which represent the local and global mutation rate, respectively. The global mutation rate determines which proportion of the weights are being mutated. The local mutation rate determines the magnitude of change of the selected weight. 

This genetic optimizer takes the following parameters for training: population size `population`, `batch_size`, local mutation rate `mut_loc`, global mutation rate `mut_glb`, and number of `epochs`. Here, for the demonstration the following training parameters were used:

```python
# Training parameters
population = 8
batch_size = 64
mut_loc = 0.01
mut_glb = 0.01
epochs = 100
````

The genetic classifier was trained for 100 epochs, with a population size of 8, and a mini batch size of 64 images. Both, the local and global mutation rate were set to 0.01. The mutation rates are similar to the learning rate of any other recursive optimizer such as the gradient descent optimizer. The classifier trained on the MNIST dataset achieved a test accuracy of 88.05 % while the classifier trained on the more complex Fashion-MNIST dataset achieved a test accuracy of 79.94 %. Not bad for a genetic optimizer with only 7840 weights.

The resulting graphs show the loss and accuracy for the evaluation dataset of the MNIST and Fashion-MNIST dataset for different combinations of local and global mutation rates.

**MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/mnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/mnist_eval_accuracy.png" height="320">
</div>

**Fashion-MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/fmnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/fmnist_eval_accuracy.png" height="320">
</div>


In order to get a better understanding of what the genetic optimizer has learned during its training, the weights can be visualized. It turns out that weights that are learned look like handwritten digits themselves. Interestingly, the weights learned tend to be an average of the training data. The upper and lower rows show the weights learned and the mean values of the training data for each class, respectively.

<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/gml_weights_mnist.png" width="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/gml_weights_fmnist.png" width="320">
  
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/gml_mnist_mean.png" width="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/gml_fmnist_mean.png" width="320">
</div>

**Hyperparameter optimization**

There are several hyperparameters that influence the genetic classifier's performance. Here, the mutation rates will be considered more closely. A good choice of mutation rates is essential for a genetic algorithm. Good mutation rates can be found, for example, by trial and error, or by a brute force grid search approach. Here are two example for the MNIST dataset.

The result of such a grid search are well suited to be visualized. The results of a grid search for both mutation parameters from `[0.0001, 0.3]` are shown below:

<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/loss_accuracy_03.png" height="320">
</div>

It can be seen that the algorithm reacts more sensitively to changes in the global mutation rate. Since the search range is still too large, the search ranges `[0.0001, 0.05]` and `[0.0001, 0.025]` can be examined more closely.

<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/loss_accuracy_test_005.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/loss_accuracy_test_0025.png" height="320">
</div>

The traditional grid search approach delivers the values `0.0118` and `0.0098` for the local and global mutation rates, respectively.

Alternatively, it is possible to build another genetic hyperparameter optimization algorithm on top of the genetic classifier algorithm. The algorithm for this can be programmed in a few lines. Here is a draft of such an implementation:

```python
n_pop = 3
mut_loc_, mut_glb_ = 0.5*np.ones((n_pop)) + 1e-4*np.random.uniform(-1,1,n_pop), 0.5*np.ones((n_pop)) + 1e-4*np.random.uniform(-1,1,n_pop)
loss_, accuracy_ = np.zeros((n_pop)), np.zeros((n_pop))

while True:
    for k in range(n_pop):
        np.random.seed(seed=73214)
        loc, glb = mut_loc_[k], mut_glb_[k]
        gml = GeneticClassifier(n_classes, n_input, glb, loc, population=4)
        loss_[k], accuracy_[k] = gml.optimize(x_train, y_train, x_eval, y_eval, x_test, y_test, epochs=40, batch_size=128)

    # Determine best child
    fitness_ = accuracy_ / loss_
    best = np.argmax(fitness_, axis=0)

    # Pass genes to next generation
    for k in range(n_pop):
        mut_loc_[k] = mut_loc_[best]
        mut_glb_[k] = mut_glb_[best]

    # Mutate genes
    mut_loc_ = mut_loc_ + 0.05 * np.random.rand(n_pop) * np.random.uniform(-1,1,size=(n_pop))
    mut_glb_ = mut_glb_ + 0.05 * np.random.rand(n_pop) * np.random.uniform(-1,1,size=(n_pop))
````

<div align="center">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/hps_global_local.png" height="320">
<img src="https://github.com/KaiFabi/VanillaGeneticClassifier/blob/master/results/hps_loss_accuracy.png" height="320">
</div>

The results of the genetic hyperparameter search are comparable to those of the grid search method. After about 80 iterations, both hyperparameters converge and lie in the range between 0.01 and 0.03. While grid search took more than 48 hours, the genetic approach took only about 6 hours to find reasonable hyperparameters.

Using genetic algorithms to optimize a large number of weights is not very efficient. Nevertheless, these algorithms are great to play around with. 
