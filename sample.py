import numpy as np
from scipy.stats import norm

class Sample:
    # handles sample data generation, accepts variables to define the size, distribution parameters and error characteristics
    def __init__(self, size, params, errorVariability):
        self.sampleSize = size
        self.distParams = params
        self.errorSD = errorVariability

    def get_sample(self):
        # generates the sample data by initialising an array of the size sampleSize, generating a list of values for these
        # x values according to the distribution parameters, generating a list of error offsets of the size sampleSize,
        # and finally adding the error to each y value and returning both the x and y values as numpy arrays

        x = np.arange(0, self.sampleSize)
        dist = [self.distParams[0]*_*_ + self.distParams[1]*_ + self.distParams[2] for _ in x]
        error = norm.rvs(0, scale=self.errorSD, size=self.sampleSize)

        y = np.asanyarray(dist+error)

        return x, y

    def save_sample(self, data):
        # saves a set of generated sample data to a csv file and saves a plot of it

        np.savetxt("simulated_data.csv", data[1], delimiter=',')



