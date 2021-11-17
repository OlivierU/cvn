from sample import Sample
import maximum_likelihood as ml
from matplotlib import pyplot as plt

sample = Sample(100, [0.44, 0.13, 41], 189)

data = sample.get_sample()
sample.save_sample(data)

# function to define model by params
def sqr(x, a,  b, m):
    return a*x*x + b*x + m

# define parameters to calculate likelihood with
mb1 = [0.5, 0.2, 40]
mb2 = [0.4, 0.1, 44]
mb3 = [0.46, 0.2, 10]

# calculate likelihoods for all lines for SD = 189
L1 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], mb1[0], mb1[1], mb1[2]), 189)
L2 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], mb2[0], mb2[1], mb2[2]), 189)
L3 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], mb3[0], mb3[1], mb3[2]), 189)

# plot the sample data as well as the potential models
plt.plot(data[0], data[1], 'r.')
plt.plot(data[0], sqr(data[0], mb1[0], mb1[1], mb1[2]), 'b--')
plt.plot(data[0], sqr(data[0], mb2[0], mb2[1], mb2[2]), 'b--')
plt.plot(data[0], sqr(data[0], mb3[0], mb3[1], mb3[2]), 'b--')
plt.show()
