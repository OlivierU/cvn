from sample import Sample
from model import Model
from matplotlib import pyplot as plt

# define parameters to generate a sample
size = 100
parameters = {
    "a": 0.44,
    "b": 0.13,
    "m": 41
}
error = 65

# generate sample data based on given params
sample = Sample(size, parameters, error)

data = sample.get_sample()
sample.save_sample(data)

model = Model()

# define a computational range for each parameter
limits = {
    "a": [0.3, 0.5],
    "b": [0.1, 0.3],
    "m": [1, 100]
}

# define the number of steps to use for optimisation
steps = 20

abm_max = model.optimise(data, steps, limits)

# plot the sample data as well as the maximum likelihood model
plt.plot(data["x"], data["y"], 'r.')
plt.plot(data["x"], model.func(data["x"], abm_max[0], abm_max[1], abm_max[2]), 'b--')
plt.show()