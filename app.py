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

model = Model()

# define a computational range for each parameter
a_range = [0.3, 0.5]
b_range = [0.1, 0.3]
m_range = [1, 100]

# define the number of steps to use for optimisation
steps = 20

abm_max = model.optimise(data, steps, [a_range, b_range, m_range])

# plot the sample data as well as the maximum likelihood model
plt.plot(data[0], data[1], 'r.')
plt.plot(data[0], model.func(data[0], abm_max[0], abm_max[1], abm_max[2]), 'b--')
plt.show()