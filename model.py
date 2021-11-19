import numpy as np

class Model:
    # function to define model by params
    def __init__(self, x, abm):
        self.x = x
        self.a = abm[0]
        self.b = abm[1]
        self.m = abm[2]

    def func(self, x, a, b, m):
        return a * x * x + b * x + m

    # calculate data likelihood for range of parameters for the given data under the assumption of a normal distribution
    def normal(self, x, median, sd):
        return np.exp(-np.power(x - median, 2.0) / (2.0 * np.power(sd, 2.0)))

    def likelihood(self, x, y, model, sd):
        median = model(x)
        ps = self.normal(y, median, sd)
        l = 1
        for p in ps:
            l = l*p
        return l

