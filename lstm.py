import numpy as np

# building blocks
def weights(shape):
    return np.random.random_sample(shape) * .1 - .05

def m(a, b):
    return np.multiply(a, b)

# activation functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_p(x):
    tmp = tanh(x)
    return 1.0 - tmp * tmp

# concat operation
def concat(vs):
    return np.concatenate(vs, axis=1)

def del_last(x):
    return x[:,:-1]

class FC(object):
    def __init__(self, act, act_p):
        self.act = act
        self.act_p = act_p

    def ff(self, x, W):
        self.x = x
        self.z = self.x.dot(W)
        self.h = self.act(self.z) if self.act else self.z
        return self

    def bp(self, d, W):
        self.dz = m(self.act_p(self.z), d) if self.act_p else d
        self.dx = W.dot(self.dz.T).T
        self.grad = m(self.x.T, self.dz)
        return self

class LSTM(object):
    def __init__(self):
        self.gf = FC(sigmoid, sigmoid_p)
        self.gi = FC(sigmoid, sigmoid_p)
        self.gc = FC(tanh, tanh_p)
        self.go = FC(sigmoid, sigmoid_p)
    
    def ff(self, x, h0, c0, W):
        self.x = x
        self.h0 = h0
        self.c0 = c0
        
        self.hxb = concat([h0, x, [[1]]])
        
        self.gf.ff(self.hxb, W['f'])
        self.gi.ff(self.hxb, W['i'])
        self.gc.ff(self.hxb, W['c'])
        self.go.ff(self.hxb, W['o'])
        
        self.c = m(self.gf.h, self.c0) + m(self.gi.h, self.gc.h)
        self.h = m(self.go.h, tanh(self.c))

    def bp(self, d1, d2, d3, W):
        # see diagram in repo to make sense of these deltas
        d4 = d1 + d2
        d5 = m(d4, self.go.h)
        d6 = m(d4, tanh(self.c))
        d7 = m(d5, tanh_p(self.c))
        d8 = d3 + d7
        self.d9 = m(d8, self.gf.h)
        d10 = m(d8, self.c0)
        d11 = m(d8, self.gi.h)
        d12 = m(d8, self.gc.h)

        d13 = del_last(self.go.bp(d6, W['o']).dx)
        d14 = del_last(self.gc.bp(d11, W['c']).dx)
        d15 = del_last(self.gi.bp(d12, W['i']).dx)
        d16 = del_last(self.gf.bp(d10, W['f']).dx)

        d17 = d13 + d14 + d15 + d16

        self.d18 = d17[0, self.h0.shape[1]:self.h0.shape[1]+self.x.shape[1]]
        self.d19 = d17[0, 0:self.h0.shape[1]]

class LSTMNetwork(object):
    def __init__(self, x_dim, h_dim, y_dim, time_steps, act, act_p):
        print('LSTM Network version 0.31')

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.time_steps = time_steps

        self.hxb_dim = h_dim + x_dim + 1

        # initialize weights
        self.randomize_weights()

        # initialize units
        self.units = {t: LSTM() for t in range(self.time_steps)}
        self.outputs = {t: FC(act, act_p) for t in range(self.time_steps)}

    def randomize_weights(self):
        self.W = {}
        self.W['f'] = weights([self.hxb_dim, self.h_dim])
        self.W['i'] = weights([self.hxb_dim, self.h_dim])
        self.W['c'] = weights([self.hxb_dim, self.h_dim])
        self.W['o'] = weights([self.hxb_dim, self.h_dim])
        self.W['y'] = weights([self.h_dim + 1, self.y_dim]) # output

    def ff(self, xs, h0, c0):
        h = h0
        c = c0
        for i, x in enumerate(xs):
            unit = self.units[i]
            unit.ff(x, h, c, self.W)

            h = unit.h
            c = unit.c
            
            self.outputs[i].ff(concat([unit.h, [[1]]]), self.W['y'])

    def out(self, i):
        return self.outputs[i].h

    def bp(self, dys, learning_rate=0.1):
        d1 = np.zeros([1, self.h_dim]) # h horizontal
        d3 = np.zeros([1, self.h_dim]) # c

        self.grad = {}
        for key in self.W:
            self.grad[key] = np.zeros_like(self.W[key])

        for i in reversed(range(len(dys))):
            dy = dys[i]

            out = self.outputs[i]
            d2 = del_last(out.bp(dy, self.W['y']).dx)

            #print('bp for unit {}:\nd1: {}\nd2: {}\nd3: {}\n'.format(i, d1, d2, d3))

            unit = self.units[i]
            unit.bp(d1, d2, d3, self.W)
            d1 = unit.d19
            d3 = unit.d9

            # accumulate grads
            self.grad['y'] += out.grad

            self.grad['f'] += unit.gf.grad
            self.grad['i'] += unit.gi.grad
            self.grad['c'] += unit.gc.grad
            self.grad['o'] += unit.go.grad
        
        # apply grads
        for key in self.W:
            self.W[key] -= self.grad[key] * learning_rate