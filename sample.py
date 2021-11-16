import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

class Sample:
    def __init__(self, size, params, errorVariability):
        self.sampleSize = size
        self.distParams = params
        self.errorSD = errorVariability

    def get_sample(self):
        x = np.arange(0, self.sampleSize)
        dist = [self.distParams[0]*_*_ + self.distParams[1]*_ + self.distParams[2] for _ in x]
        error = norm.rvs(0, scale=self.errorSD, size=self.sampleSize)

        y = np.asanyarray(dist+error)

        return x, y

    def save_sample(self, data):
        np.savetxt("simulated_data.csv", data[1], delimiter=',')

        plt.plot(data[0], data[1], 'r.')
        plt.savefig("plot.png")



