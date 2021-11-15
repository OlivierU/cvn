import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

x = np.arange(0, 100)
dist = [0.44*_*_ + 0.13*_ + 41 for _ in x]
error = norm.rvs(0, scale=189, size=100)

y = np.asanyarray(dist+error)

plt.plot(x, y, 'r.')
plt.show()

np.savetxt("simulated_data.csv", y, delimiter=',')