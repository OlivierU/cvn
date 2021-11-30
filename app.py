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

size = 10 # number of values to generate

parameters = [ # distribution parameters used
    3,
    0.13,
    0.25,
    0.44,
]

error = 15 # SD of the error added to each value

# Optimisation #

limits = [ # computational range for each parameter
    [1, 100],
    [0.1, 0.3],
    [0.2, 0.3],
    [0.3, 0.5],
]

steps = 20 # number of optimisation cycles


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

# find maximum likelihood function based on the defined variables
abm_max = model.optimise(data, steps, limits)


############
# Plotting #
############

# plot the sample data as well as the maximum likelihood model
plt.plot(data["x"], data["y"], 'r.')
plt.plot(data["x"], model.func(abm_max, data["x"]), 'b--')
plt.show()