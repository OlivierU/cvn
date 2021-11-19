import numpy as np

from sample import Sample
from model import Model
from matplotlib import pyplot as plt

# define parameters to generate a sample
size = 100
parameters = [0.44, 0.13, 41]
error = 65

# generate sample data based on given params
sample = Sample(size, parameters, error)

data = sample.get_sample()
sample.save_sample(data)

model = Model(size, parameters)

# define a computational range for each parameter
a_range = [0.3, 0.5]
b_range = [0.1, 0.3]
m_range = [1, 100]

# define the number of steps to use for optimisation
steps = 20

# combine the series of each parameter into a meshgrid and then flatten it out to get an array of all possible param combinations
A, B, M = np.meshgrid(np.linspace(a_range[0], a_range[1], steps), np.linspace(b_range[0], b_range[1], steps), np.linspace(m_range[0], m_range[1], steps))
ABM = np.c_[A.ravel(), B.ravel(), M.ravel()]

# calculate likelihoods
L = np.array([model.likelihood(data[0], data[1], lambda x: model.func(x, abm[0], abm[1], abm[2]), 65) for abm in ABM]).reshape(M.shape)

# select parameter with maximum likelihood
abm_max = np.array([A[np.unravel_index(L.argmax(),L.shape)],B[np.unravel_index(L.argmax(),L.shape)],M[np.unravel_index(L.argmax(),L.shape)]])

# plot the sample data as well as the maximum likelihood model
plt.plot(data[0], data[1], 'r.')
plt.plot(data[0], model.func(data[0], abm_max[0], abm_max[1], abm_max[2]), 'b--')
plt.show()