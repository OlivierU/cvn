import numpy as np

from sample import Sample
from model import Model
from matplotlib import pyplot as plt

# ***********************************************************************
# This file contains the main logic of the modeling process.
#
# All necessary variables are defined and used to build the objects needed to do the actual calculations.
# The basic structure is:
#   - Generate a random sample based on a specified distribution and it's parameters
#   - Define a model and calculate likelihoods for a set range of parameters
#   - Generate plots to visualise the process
#
#
#   TODO: Goodnes-of-Fit, Variability calculations
#
# ***********************************************************************


########################
# Variable definitions #
########################

# Sample #

size = 50 # number of values to generate

parameters = [ # distribution parameters used
    3,
    0.13,
]

error = 0.64 # SD of the error added to each value

# Optimisation #

limits = [ # computational range for each parameter
    [1, 5],
    [-0.1, 0.15],
]

steps = 120 # number of optimisation cycles

k = 10 # number of splits for k-fold validation

bootstrap_cycles = 10 # number of time the bootstrapping is applied

#######################
# Object Construction #
#######################

# generate sample data based on given params
sample = Sample(size, parameters, error)

data = sample.get_sample()
sample.save_sample(data)

# generate the model used to find parameters
model = Model()


################
# Optimisation #
################

rng = np.random.default_rng()

# split the data into k folds to use as training and testing sets
splits = np.hsplit(rng.permutation(data, 1), k)

# initialise an array of length k to store the estimated parameter of each train & test iteration
max_likelihoods = np.empty((k, len(parameters)))

# use each fold once as a testing set while the other k-1 folds are used as training sets
for f in range(len(splits)):
    test = splits[f]
    train = np.sort(np.column_stack(np.delete(splits, f, 0)))

    # calculate ml parameters for each fold
    max_likelihood_f = model.optimise(train, steps, limits)

    max_likelihoods[f] = max_likelihood_f

    plt.plot(train[0], model.func(max_likelihood_f, train[0]), 'b--', alpha=0.2)

# calculate the mean of each parameter from all k parameter sets to use as final ml estimate
best_ml = np.mean(max_likelihoods, axis=0)

# plot the final ml estimate in green
plt.plot(data[0], model.func(best_ml, data[0]), 'g-')

############
# Plotting #
############

bootstrapped_samples = np.empty((bootstrap_cycles, size * 2))

bootstrap_params = np.empty((bootstrap_cycles, len(parameters)))

def get_sample_from_model():
    poly = np.polynomial.polynomial.Polynomial(best_ml)

    return np.sort(poly(rng.integers(low=-size, high=size, size=size*2)))


for i in range(bootstrap_cycles):
    bootstrapped_samples[i] = get_sample_from_model()

    indexed_sample = np.stack((np.arange(-size, size), bootstrapped_samples[i]), axis=0)

    bootstrap_params[i] = model.optimise(indexed_sample, steps, limits)

    plt.plot(data[0], model.func(bootstrap_params[i], data[0]), 'y--')


bootstrap_var = bootstrap_params.T.std(axis=1)

plt.plot(data[0], data[1], 'r.')

plt.show()