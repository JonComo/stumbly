import numpy as np
import tensorflow as tf

class Agent(object):
    def __init__(self, features=2, actions=2, hdim=4, eps=.1, learning_rate=0.01, gamma=0.95, max_memory=5000):
        self.actions = actions
        self.features = features
        self.hdim = hdim
        self.learning_rate = learning_rate
        self.max_memory = max_memory
        self.gamma = gamma

        self.A = range(self.actions)
        self.M = [] # memory (s1, a1, r, s2, a2) tuples
        self.init_networks()
        
    def init_networks(self):
        self.qnet = NN(xdim=self.features, hdim=self.hdim, ydim=self.actions, learning_rate=self.learning_rate) # action-value prediction network
        self.vnet = NN(xdim=self.features, hdim=self.hdim, ydim=1, learning_rate=self.learning_rate) # value prediction network
        self.fatigue = np.zeros([1, self.actions])

    def advantage(self, state):
        q = self.q_approx(state)
        v = self.v_approx(state)
        return q - v[0, 0]
        
    def sample_action_eps(self, qs):
        if np.random.random_sample() > self.eps:
            return np.argmax(qs[0])

        return np.random.choice(self.A)

    def sample_action_eps_fatigue(self, qs):
        a = self.sample_action_eps(qs - self.fatigue * self.eps)
        self.fatigue[0, a] = 1
        self.fatigue *= 0.995
        return a

    def sample_action_softmax(self, qs):
        if all(qs[0, 0] == qs[0, :]): # if they're all the same
            return np.random.choice(self.A)
        
        dist = [np.exp(q) for q in qs[0, :]]
        dist /= np.sum(dist)
        return np.random.choice(self.A, p=dist)

    def sample_action_policy(self, state):
        # use policy_w to sample actions
        return np.random.choice(self.A, p=self.policy_distribution(state))

    def sample_action_memory(self, state):
        r_max = float('-inf')
        seen_as = []
        seen_rs = []
        for xp in self.M:
            if np.array_equal(xp['s1'], state):
                seen_as.append(xp['a1'])
                seen_rs.append(xp['r'])

        m = len(seen_as)
        if m == 0:
            print('didnt find an experience')
            return np.random.choice(self.A)

        dist = [self.eps/m] * m
        dist[np.argmax(seen_as)] += 1.0 - self.eps
        return np.random.choice(seen_as, p=dist)
    
    def train(self, iters=100, batch_size=10, keep_prob=0.5):
        for i in range(iters):
            x_batch, t_batch = self.q_batch(batch_size)
            self.qnet.train(x_batch, t_batch, keep_prob)

    def test(self, batch_size=10):
        x_batch, t_batch = self.q_batch(batch_size)
        return self.qnet.test(x_batch, t_batch)

    def q_batch(self, batch_size):
        x_batch = np.zeros([batch_size, self.features])
        t_batch = np.zeros([batch_size, self.actions])

        for b in range(batch_size):
            xp = self.M[np.random.randint(len(self.M))]

            s1_q = self.qnet.ff(xp['s1'])
            s2_q = self.qnet.ff(xp['s2'])

            target = s1_q.copy()
            target[0, xp['a1']] = xp['r'] + self.gamma * np.max(s2_q[0])

            x_batch[b] = xp['s1']
            t_batch[b] = target

        return x_batch, t_batch
        
    def train_vnet(self, xp):
        # train with SARS
        v2 = self.v_approx(xp['s2'])
        v1 = self.v_approx(xp['s1'])

        target = xp['r'] + self.gamma * v2
        
        delta = v1 - target
        self.vnet.bp(delta, self.learning_rate)
            
    def memorize(self, xp):
        if len(self.M) >= self.max_memory:
            self.M[np.random.randint(self.max_memory)] = xp
        else:
            self.M.append(xp)
            
    def reset(self):
        self.init_networks()
        self.M = []
    
    @property
    def eps(self):
        return self._eps
    
    @eps.setter
    def eps(self, value):
        self._eps = np.round(min(1.0, max(0.0, value)), 2)

# Neural Network using tensorflow
class NN(object):
    def __init__(self, xdim, hdim, ydim, learning_rate=0.001):
        with tf.device('/cpu:0'):
            self.x = tf.placeholder(tf.float32, [None, xdim])
            self.t = tf.placeholder(tf.float32, [None, ydim])
            self.keep_prob = tf.placeholder(tf.float32)

            self.params = []
            self.l1 = self.layer(self.x, xdim, hdim, tf.nn.tanh)
            #self.l2 = self.layer(self.l1, hdim, hdim, tf.nn.relu)
            self.y = self.layer(self.l1, hdim, ydim, None)

            self.cost = tf.reduce_mean(tf.square(self.y - self.t))
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.initialize_all_variables())
        
    def layer(self, in_data, xdim, ydim, act):
        W = tf.Variable(tf.random_normal(shape=[xdim, ydim], stddev=0.1))
        b = tf.Variable(tf.random_normal(shape=[ydim], stddev=0.1))
        self.params += [W, b]
        z = tf.matmul(in_data, W) + b
        h = act(z) if act else z
        h_drop = tf.nn.dropout(h, self.keep_prob)
        return h_drop
    
    def train(self, x_data, t_data, keep_prob=0.5):
        """x_data shape [BATCH, xdim], t_data shape [BATCH, ydim]"""
        self.session.run(self.train_step, feed_dict={self.x: x_data, self.t: t_data, self.keep_prob: keep_prob})

    def test(self, x_data, t_data):
        return self.cost.eval(session=self.session, feed_dict={self.x: x_data, self.t: t_data, self.keep_prob: 1.0})
        
    def ff(self, x_data):
        return self.y.eval(session=self.session, feed_dict={self.x: x_data, self.keep_prob: 1.0})