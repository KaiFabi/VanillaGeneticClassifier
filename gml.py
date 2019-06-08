import os
import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Seed
np.random.seed(seed=73214)

# Multilayer Perceptron Class
class GeneticClassifier(object):

    def __init__(self, n_classes, n_input, population, mut_glb, mut_loc):

        # Population size 
        self.population = population
        # Mutation parameter
        self.mut_glb = mut_glb
        self.mut_loc = mut_loc
        # Initialize weights
        self.W = [1e-4*np.random.uniform(-1,1,size=(n_classes, n_input)) for k in range(population)]

    def optimize(self, x_train, y_train, x_eval, y_eval, x_test, y_test, epochs, batch_size, mut_loc, mut_glb):
        
        loss = np.zeros((self.population))
        for epoch in range(epochs):
            idx_list = self.grouped_rand_idx(len(x_train), batch_size)

            # Train
            for idx in idx_list:
                # Get batch of random training samples
                x_batch, y_batch = x_train[idx], y_train[idx]

                for k in range(self.population):
                    # Compute loss
                    U = expit(x_batch.dot(self.W[k].T)) - y_batch
                    loss[k] = np.sum(U*U)

                # Compute best child
                best_idx = np.argmin(loss, axis=0)

                # Pass genes to next generation # TODO: Combine with next loop
                for k in range(self.population):
                    if k != best_idx:
                        self.W[k] = self.W[best_idx]

                # Mutate genes
                for k in range(self.population):
                    if k != best_idx:
                        self.W[k] = self.W[k] + self.mut_loc*np.random.uniform(-1,1,size=(n_classes, n_input))*(np.random.rand(n_classes, n_input)<self.mut_glb)
 
            if epoch % 1 == 0:
                self.prediction(x_eval, y_eval, best_idx, epoch, mode="eval")

            if epoch % 30 == 0:
                self.visualize_weights(best_idx, epoch)

        # Compute test accuracy and loss
        self.prediction(x_test, y_test, best_idx, epoch, mode="test")

    def pred(self, X, Y, best_idx):
        Y_ = expit(X.dot(self.W[best_idx].T))    # Predict
        loss = np.sum((Y-Y_)**2) / len(X)        # Compute loss
        accuracy = np.sum(np.argmax(Y_, axis=1) == np.argmax(Y, axis=1)) / len(X) # Compute accuracy
        return loss, accuracy

    def prediction(self, X, Y, best_idx, epoch, mode):
        loss, accuracy = self.pred(X, Y, best_idx)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)

    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]

    def visualize_weights(self, best_idx, epoch):
        nrow, ncol = 2, 5
        fig, axes = plt.subplots(nrows = nrow, ncols=ncol, figsize=(ncol, nrow))
        for k, ax in enumerate(axes.flatten()):
            ax.imshow(self.W[best_idx][k].reshape(28,28), cmap="viridis")
            ax.axis("off")
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        plt.savefig("gml_loc_01_glb_01_epoch_" + str(epoch) + ".png", dpi=200)
        plt.close()

# Load data
x_train, y_train, x_eval, y_eval, x_test, y_test, n_classes, n_input = data.get_data(dataset="mnist", norm=True, one_hot=True)

# Training parameters
population = 8
batch_size = 64
mut_loc = 0.01
mut_glb = 0.01
epochs = 100

# Run training
gml = GeneticClassifier(n_classes, n_input, population)
gml.optimize(x_train, y_train, x_eval, y_eval, x_test, y_test, epochs, batch_size, mut_loc, mut_glb)
