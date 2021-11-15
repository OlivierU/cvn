import numpy as np
from matplotlib import pyplot as plt
from math import exp
from scipy.stats import norm

x = np.arange(0, 100)
dist = [0.44*exp(.031*_) for _ in x]
error = norm.rvs(0, scale=.95, size=100)

y = np.asanyarray(dist+error)

plt.plot(x, y, 'r.')
plt.show()

np.savetxt("simulated_data.csv", y, delimiter=',')