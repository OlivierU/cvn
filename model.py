import numpy as np


class Model:
    # function to define model by params
    def __init__(self):
        pass

    @staticmethod
    def func(x, a, b, m):
        return a * x * x + b * x + m

    # calculate data likelihood for range of parameters for the given data under the assumption of a normal distribution
    def normal(self, x, median, sd):
        return np.exp(-np.power(x - median, 2.0) / (2.0 * np.power(sd, 2.0)))

    def likelihood(self, x, y, model, sd):
        median = model(x)
        ps = self.normal(y, median, sd)
        l = 1
        for p in ps:
            l = l * p
        return l

    def optimise(self, data, steps, param_ranges):

        a_range = param_ranges[0]
        b_range = param_ranges[1]
        m_range = param_ranges[2]

        # combine the series of each parameter into a meshgrid and then flatten it out to get an array of all possible param combinations
        A, B, M = np.meshgrid(np.linspace(a_range[0], a_range[1], steps), np.linspace(b_range[0], b_range[1], steps),
                              np.linspace(m_range[0], m_range[1], steps))
        ABM = np.c_[A.ravel(), B.ravel(), M.ravel()]

        # calculate likelihoods
        L = np.array(
            [self.likelihood(data[0], data[1], lambda x: self.func(x, abm[0], abm[1], abm[2]), 65) for abm in ABM]).reshape(
            M.shape)

        # select parameter with maximum likelihood
        abm_max = np.array([A[np.unravel_index(L.argmax(), L.shape)], B[np.unravel_index(L.argmax(), L.shape)],
                            M[np.unravel_index(L.argmax(), L.shape)]])

        return abm_max
