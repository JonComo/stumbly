import numpy as np
import util

def act(x):
    return np.tanh(x)

def act_p(x):
    t = np.tanh(x)
    return 1.0 - t**2

class Features(object):
    def __init__(self, xdim, hdim):
        self.W1 = util.randn([xdim, hdim])
        self.W2 = util.randn([hdim, xdim])

    def train(self, xs, learning_rate=0.1):
        for x in xs:
            z1 = x.dot(self.W1)
            h1 = np.tanh(z1)
            z2 = h1.dot(self.W2)
            h2 = np.tanh(z2)

            dh2 = (h2 - xs)
            dz2 = np.multiply(act_p(z2), dh2)
            g2 = np.multiply(h1.T, dz2)

            dh1 = self.W2.dot(dz2.T).T
            dz1 = np.multiply(act_p(z1), dh1)
            g1 = np.multiply(x.T, dz1)

            self.W1 -= g1 * learning_rate
            self.W2 -= g2 * learning_rate

    def ff(self, x):
        z1 = x.dot(self.W1)
        h1 = act(z1)
        return h1