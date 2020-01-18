import os
import data
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.special import expit

# Seed
np.random.seed(seed=73214)

class Classifier(object):

    def __init__(self, n_classes, n_input, local_population_size, global_population_size, mutation_rate, update_probability, local_epochs, global_epochs):

        # Population size 
        self.local_population_size = local_population_size
        self.global_population_size = global_population_size

        # Mutation parameter
        self.update_probability = update_probability
        self.mutation_rate = mutation_rate

        # Initialize weights
        self.W = np.random.uniform(-1,1,size=(n_classes, n_input+1))/n_input

        # Training duration 
        self.local_epochs = local_epochs
        self.global_epochs = global_epochs 
 
    def optimize_genetic_parallel(self, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size):

        # Add bias term
        x_train = self.add_bias(x_train)
        x_test= self.add_bias(x_test)
        x_valid= self.add_bias(x_valid)

        n_cores = self.global_population_size 
        pool = mp.Pool(processes = n_cores)
        for epoch in range(global_epochs):

            if epoch % 1 == 0:
                valid_loss, valid_accuracy = self.pred(x_valid, y_valid)
                train_loss, train_accuracy = self.pred(x_train, y_train) 
                args = (epoch, "valid", "train", valid_loss, valid_accuracy, train_loss, train_accuracy)
                print('epoch {0} {1}_loss {3:.6f} {1}_accuracy {4:.4f} {2}_loss {5:.6f} {2}_accuracy {6:.4f}'.format(*args), flush=True)
                self.visualize_weights(epoch)

            # Every core gets a copy of the weight matrices and returns the best matrix
            rnd_seed = np.random.randint(np.iinfo(np.uint32).max, size=n_cores)
            W_tmp = pool.map(self.optimizer, ((self.W, x_train, y_train, rnd_seed[p]) for p in range(n_cores)))

            loss = self.prediction_loss(x_valid, y_valid, W_tmp)
            best_idx = np.argsort(loss, axis=0)

            # Pass best and second best individual to crossover operation
            self.W = self.crossover(W_tmp[best_idx[0]], W_tmp[best_idx[1]])

        # Compute test accuracy and loss
        test_loss, test_accuracy = self.pred(x_test, y_test)
        print('epoch {0} {1}_loss {2:.6f} {1}_accuracy {3:.4f}'.format(epoch, "test", test_loss, test_accuracy), flush=True)
        #self.prediction(x_test, y_test, epoch, mode="test")

    def optimizer(self, args): # no need to copy array of matrices, single matrix sufficient
        W_glb, x_train, y_train, rnd_seed = args
        np.random.seed(rnd_seed)
        W = [np.copy(W_glb) for k in range(self.local_population_size)]
        for _ in range(self.local_epochs):
            idx_list = self.grouped_rand_idx(len(x_train), batch_size)
            for idx in idx_list:
                W = self.mutation(W)
                # Get batch of random training samples
                x_batch, y_batch = x_train[idx], y_train[idx]
                # Compute loss for every agent
                loss = self.prediction_loss(x_batch, y_batch, W)
                # Determine best agent 
                best_idx = np.argmin(loss, axis=0)
                # Duplicate best genes
                for k in range(self.local_population_size):
                    if k != best_idx:
                        W[k] = W[best_idx]
        return W[best_idx]
    
    def prediction_loss(self, x_batch, y_batch, W):
        loss = [np.square(y_batch - self.softmax(x_batch.dot(W[k].T))).sum() for k in range(len(W))]
        return loss

    def mutation(self, W):
        W = [W[k] + self.mutation_rate*np.random.uniform(-1, 1, size=W[0].shape)* \
                (np.random.random(size=W[0].shape)<self.update_probability) for k in range(len(W))]
        return W

    def crossover(self, W_1, W_2):
        B_1 = np.random.randint(2, size=W_1.shape)
        return B_1*W_1 + (1-B_1)*W_2
    
    def softmax(self, x):
        y = expit(x)
        return y / np.sum(y, axis=-1, keepdims=True) 

    def pred(self, X, Y):
        Y_ = self.softmax(X.dot(self.W.T))         # Predict
        loss = np.sum((Y-Y_)**2) / len(X)   # Compute loss
        accuracy = np.sum(np.argmax(Y_, axis=1) == np.argmax(Y, axis=1)) / len(X) # Compute accuracy
        return loss, accuracy

    def prediction(self, X, Y, epoch, mode):
        loss, accuracy = self.pred(X, Y)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)
    
    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]

    def visualize_weights(self, epoch):
        nrow, ncol = 2, 5
        fig, axes = plt.subplots(nrows = nrow, ncols=ncol, figsize=(ncol, nrow))
        for k, ax in enumerate(axes.flatten()):
            ax.imshow(self.W[k][:-1].reshape(28,28), cmap="viridis")
            ax.axis("off")
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        plt.savefig("fmnist_" + str(epoch) + ".png", dpi=200)
        plt.close()

        fig, axes = plt.subplots(1, 1, tight_layout=True)
        axes.hist(self.W.reshape(10*(28**2+1),1), histtype="stepfilled", bins = int(np.sqrt(10*(28**2+1))), density=False, facecolor='g', alpha=0.4)
        axes.grid(linestyle="dashed", linewidth="0.5", color="gray")
        plt.xlabel('Weight value')
        plt.ylabel('Frequency')
        plt.savefig("hist_fmnist.png", dpi=200)
        plt.close()

    def add_bias(self, x):
        M, N = x.shape
        x_tmp = np.ones((M, N + 1))
        x_tmp[:,:-1] = x
        return x_tmp

# Load data
x_train, \
y_train, \
x_valid, \
y_valid, \
x_test, \
y_test, \
n_classes, \
n_input = data.get_data(dataset="fmnist", norm=True, one_hot=True)

# Training parameters
batch_size = 16
local_epochs = 10
global_epochs = 100
mutation_rate = 0.001
update_probability = 0.001
local_population_size = 4
global_population_size = 4

# Run training
gml = Classifier(n_classes, n_input, local_population_size, global_population_size, mutation_rate, update_probability, local_epochs, global_epochs)
gml.optimize_genetic_parallel(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size)
