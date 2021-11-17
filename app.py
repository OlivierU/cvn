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
abm1 = [0.5, 0.2, 40]
abm2 = [0.4, 0.1, 44]
abm3 = [0.46, 0.2, 10]

# calculate likelihoods for all lines for SD = 189
L1 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], abm1[0], abm1[1], abm1[2]), 189)
L2 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], abm2[0], abm2[1], abm2[2]), 189)
L3 = ml.likelihood(data[0], data[1], lambda x: sqr(data[0], abm3[0], abm3[1], abm3[2]), 189)

# plot the sample data as well as the potential models
plt.plot(data[0], data[1], 'r.')
plt.plot(data[0], sqr(data[0], abm1[0], abm1[1], abm1[2]), 'b--')
plt.plot(data[0], sqr(data[0], abm2[0], abm2[1], abm2[2]), 'b--')
plt.plot(data[0], sqr(data[0], abm3[0], abm3[1], abm3[2]), 'b--')
plt.show()
