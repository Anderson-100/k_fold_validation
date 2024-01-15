import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

datapath = "./"


def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=0):
    # set the random state if provided
    np.random.seed(rand_seed)

    # initialize the split index
    array_len = len(arrays[0].T)
    split_idx = int(array_len * (1 - test_size))

    # initialize indices to the default order
    indices = np.arange(array_len)

    # shuffle the arrays if shuffle is True
    if shuffle:
        np.random.shuffle(indices)

    # Split the arrays
    result = []
    for array in arrays:
        if shuffle:
            array = array[:, indices]
        train = array[:, :split_idx]
        test = array[:, split_idx:]
        result.extend([train, test])

    return result


def run_task1():
    csvname = datapath + 'new_gene_data.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))  # (1, 72)

    np.random.seed(0)  # fix randomness

    norm, inv_norm = standard_normalizer(x)
    # norm, inv_norm = PCA_sphereing(x)
    x = norm(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, rand_seed=1)

    # TODO
    num_folds = 4
    folds = split_data(y_train, num_folds)

    # Loop through different hyperparameters, saving the ones with lowest validation cost
    best_accuracy = 0
    best_hyperparams = []

    # Determined through testing that this is the best range of values to check through
    for exp1 in range(3, 5):
        alpha = 10**(-exp1)

        for exp2 in range(1, 5):
            lam = 10**(-exp2)
            total_val_accuracy = 0
            print(f"Testing alpha={alpha}, lambda={lam}")
            for i in range(num_folds):
                # run gradient descent with current alpha and lam
                # on every fold except i
                current_train_x = []
                current_train_y = []

                current_val_x = []
                current_val_y = []

                for j, f in enumerate(folds):
                    if int(f) - 1 == i:
                        current_val_x.append(x_train[:, j])
                        current_val_y.append(y_train[:, j])
                    else:
                        current_train_x.append(x_train[:, j])
                        current_train_y.append(y_train[:, j])
                
                current_train_x = np.array(current_train_x).T
                current_train_y = np.array(current_train_y).T

                current_val_x = np.array(current_val_x).T
                current_val_y = np.array(current_val_y).T

                # print(np.shape(current_train_x))
                # print(np.shape(current_train_y))

                weights, costs = gradient_descent(softmax, 0.001 * jnp.array(np.random.randn(7129, 1)), current_train_x, current_train_y, 100, alpha, lam)
                # print(f"Final Cost: {costs[-1]}")
                # calculate accuracy of fold i using weights determined from training
                total_val_accuracy += accuracy(weights[-1], current_val_x, current_val_y)
            
            # if better than current best, save
            avg_val_accuracy = total_val_accuracy / float(num_folds)
            print(f"Avg Validation Accuracy: {avg_val_accuracy}\n")
            if avg_val_accuracy > best_accuracy:
                best_accuracy = avg_val_accuracy
                best_hyperparams = [alpha, lam]

    print(f"Best Validation Accuracy: {best_accuracy}")
    print(f"Best Hyperparameters: {best_hyperparams}\n")

    weights, costs = gradient_descent(softmax, 0.001 * jnp.array(np.random.randn(7129, 1)), x_train, y_train, 100, best_hyperparams[0], best_hyperparams[1])
    print(f"Test Accuracy: {accuracy(weights[-1], x_test, y_test)}\n")
    plt.figure()
    plt.plot(np.arange(len(costs)), costs)
    plt.title("Cost History of Best Hyperparameters on Whole Training Set")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig('costs.png')

    flattened_weights = weights[-1].flatten()
    top_5_indexes = np.abs(flattened_weights).argsort()[-6:]

    # Ignore index 0
    print(f"Top 5 Most Influential Genes:\n{top_5_indexes}\n")
    print(f"Top 5 Most Influential Genes Values:\n{flattened_weights[top_5_indexes]}")

def gradient_descent(g, w, x, y, max_its, alpha, lam):
	# compute gradient module using jax
    gradient = grad(g)

	# run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [g(w, x, y, lam)]          # container for corresponding cost function history
    for k in range(1, max_its+1):

        # Gradient descent step
        # print(gradient(w, x, y, lam))
        w = w - alpha * gradient(w, x, y, lam)

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w, x, y, lam))

        # if k % 1000 == 0:
        # 	print(k, "Current Cost =", cost_history[-1])
    return weight_history, cost_history

def model(w, x):
    a = w[0] + jnp.dot(x.T, w[1:])
    return a.T

# Compute softmax losss
def softmax(w, x, y, lam):
    cost = jnp.sum(jnp.log(1 + jnp.exp(-y * model(w, x))))
    cost += lam * jnp.linalg.norm(w[1:], ord=1)**2
    return cost / float(np.size(y))

def accuracy(w, x, y):
    all_evals = model(w, x)
    results = all_evals * y >= 0

    unique, unique_count = np.unique(results, return_counts=True)
    d = dict(zip(unique, unique_count))

    correct = 0
    if True in d:
        correct = d[True]

    return correct / float(np.size(y))

def standard_normalizer(x):    
    # compute the mean and standard deviation of the input
    x_means = np.nanmean(x,axis = 1)[:,np.newaxis]
    x_stds = np.nanstd(x,axis = 1)[:,np.newaxis]   

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # fill in any nan values with means 
    ind = np.argwhere(np.isnan(x) == True)
    for i in ind:
        x[i[0],i[1]] = x_means[i[0]]

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means

    # return normalizer 
    return normalizer,inverse_normalizer

# function for splitting dataset into k folds
def split_data(y, folds):
    # split data into k equal (as possible) sized sets
    # np.random.seed(0)
    L = np.size(y)
    order = np.random.permutation(L)
    c = np.ones((L,1))
    L = int(np.round((1/folds)*L))
    for s in np.arange(0,folds-2):
        c[order[s*L:(s+1)*L]] = s + 2
    c[order[(folds-1)*L:]] = folds
    return c

if __name__ == '__main__':
    run_task1()
