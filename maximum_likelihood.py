import numpy as np

# calculate data likelihood for range of parameters for the given data under the assumption of a normal distribution
def normal(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2.0 * np.power(sig, 2.0)))

def likelihood(x,y,model,std):
    mu = model(x)
    ps = normal(y,mu,std)
    l = 1
    for p in ps:
        l = l*p
    return l

