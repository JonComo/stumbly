import numpy as np

class Agent(object):
    def __init__(self, features=2, actions=2, eps=.1, learning_rate=0.01, max_memory=5000):
        self.actions = actions
        self.features = features
        self.learning_rate = learning_rate
        self.max_memory = max_memory
        
        self.avg_r = 0.0 # average reward
        self.A = range(self.actions)
        self.M = [] # memory (s1, a1, r, s2, a2) tuples
        self.init_networks()
        
    def init_networks(self):
        self.qnet = NN(xdim=self.features, hdim=4, ydim=self.actions) # action-value prediction network
        self.vnet = NN(xdim=self.features, hdim=4, ydim=1) # value prediction network
        
    def q_approx(self, state):
        return self.qnet.ff(state)
    
    def v_approx(self, state):
        return self.vnet.ff(state)
    
    def advantage(self, state):
        q = self.q_approx(state)
        v = self.v_approx(state)
        return q - v[0, 0]
        
    def sample_action(self, qs):
        if all(qs[0, 0] == qs[0, :]): # if they're all the same
            return np.random.choice(self.A)
        
        dist = [self.eps/self.actions] * self.actions
        dist[np.argmax(qs[0])] += 1.0 - self.eps
        return np.random.choice(self.A, p=dist)
    
    def train(self, iters=1):
        for i in range(iters):
            xp = self.M[np.random.randint(len(self.M))]
            self.train_qnet(xp)
            self.train_vnet(xp)
            
    def train_qnet(self, xp):
        # train with SARSA
        s2_q = self.q_approx(xp['s2'])
        s1_q = self.q_approx(xp['s1'])

        target = s1_q.copy()
        target[0, xp['a1']] = xp['r'] + GAMMA * np.max(s2_q[0])

        deltas = s1_q - target
        self.qnet.bp(deltas, self.learning_rate)
        
    def train_vnet(self, xp):
        # train with SARS
        v2 = self.v_approx(xp['s2'])
        v1 = self.v_approx(xp['s1'])

        target = xp['r'] + GAMMA * v2
        
        delta = v1 - target
        self.vnet.bp(delta, self.learning_rate)
            
    def memorize(self, xp):
        self.avg_r += 0.001 * (xp['r'] - self.avg_r)
        if len(self.M) >= self.max_memory:
            self.M[np.random.randint(self.max_memory)] = xp
        else:
            self.M.append(xp)
            
    def visualize_q(self):
        T = np.linspace(0, 2.0 * np.pi, 100)
        q2 = np.array([self.q_approx(np.array([[np.sin(t), np.cos(t)]])) for t in T])
        fig = plt.plot(T, q2[:,0])
            
    def reset(self):
        self.init_networks()
        self.M = []
    
    @property
    def eps(self):
        return self._eps
    
    @eps.setter
    def eps(self, value):
        self._eps = np.round(min(1.0, max(0.0, value)), 2)

# Neural Network
def bias_add(x):
    return np.concatenate([x, [[1]]], axis=1) # bias add
def act(x):
    return np.tanh(x)
def actp(x):
    return 1.0 - np.tanh(x)**2

class Layer(object):
    def __init__(self, xdim, ydim, a=act, ap=actp):
        self.a = a
        self.ap = ap
        self.xdim = xdim
        self.ydim = ydim
        self.W = np.random.randn(xdim + 1, ydim)
    def ff(self, x):
        self.x = bias_add(x)
        self.z = self.x.dot(self.W)
        self.h = self.a(self.z) if self.a else self.z
        return self
    def bp(self, deltas, learning_rate=0.1):
        self.dz = np.multiply(deltas, self.ap(self.z)) if self.ap else deltas
        self.grad = np.multiply(self.x.T, self.dz)
        self.dx = self.W.dot(self.dz.T).T[:, :-1] # remove bias add
        
        self.W -= self.grad * learning_rate
    
class NN(object):
    def __init__(self, xdim, hdim, ydim):
        self.l1 = Layer(xdim, hdim)
        self.l2 = Layer(hdim, hdim)
        self.l3 = Layer(hdim, ydim)
    
    def ff(self, x):
        h1 = self.l1.ff(x).h
        h2 = self.l2.ff(h1).h
        h3 = self.l3.ff(h2).h
        return h3
    
    def bp(self, deltas, learning_rate=0.1):
        self.l3.bp(deltas, learning_rate)
        self.l2.bp(self.l3.dx, learning_rate)
        self.l1.bp(self.l2.dx, learning_rate)