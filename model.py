import numpy as np
from numpy.polynomial.polynomial import Polynomial


class Model:
    # function to define model by params
    def __init__(self):
        pass

    @staticmethod
    def func(params, x):
        poly = Polynomial(params)

        return poly(x)

    # calculate data likelihood for range of parameters for the given data under the assumption of a normal distribution
    def normal(self, x, median, sd):
        return np.exp(-np.power(x - median, 2.0) / (2.0 * np.power(sd, 2.0)))

    def likelihood(self, x, y, model, sd):
        median = model(x)
        ps = self.normal(y, median, sd)
        return np.logaddexp.reduce(ps)


    @staticmethod
    def series(steps, lower_bound, upper_bound):
        return np.linspace(lower_bound, upper_bound, num=steps)

    def optimise(self, data, steps, param_ranges):

        # generate a series of steps for each parametere based on the computational limits specified
        value_series = np.empty((len(param_ranges), steps))

        for p in range(len(param_ranges)):
            value_series[p] = self.series(steps, min(param_ranges[p]), max(param_ranges[p]))

        # combine the series of each parameter into a meshgrid and then flatten it out to get an array of all possible param combinations

        candidate_params = np.stack(np.meshgrid(*value_series)).T.reshape(-1,len(param_ranges))

        # calculate the SD of the given data to use in likelihood calculation
        sd = 15

        # calculate likelihoods
        likelihoods = np.empty(candidate_params.shape[0])

        for cand in range(candidate_params.shape[0]):
            likelihoods[cand] = self.likelihood(data["x"], data["y"], lambda x: self.func(candidate_params[cand], x), sd)


        # select parameter with maximum likelihood
        target_params = candidate_params[np.argmax(likelihoods)]

        return target_params
