# There are several obvious downsides to auto-regressive models:
#
#   1. We need many input neurons to account for all of the previous outputs
#       that we care about. The input space can grow huge when we need to
#       account for many previous output time-steps.
#   2. This also mean that each timestep is learned separately because it uses
#       differnet neurons. Learning will take much longer.
#   2. We are limited by how far back we need to look. The decision needs to be
#       made ahead of time when designing the model, instead of being learned.
#   3. We are only only relying on the previous input and output, instead of
#       learning more complicated dynamics or state.
#
# To illustrate these limitations, in this exercise we'll train a model that
# performs binary addition (and by extension, any positive integer addition),
# which can be defined as:
#
#   x1, x2, Carry
#   ---------------------------
#   0,  0,  0   => 0 (Carry: 0)
#   0,  1,  0   => 1 (Carry: 0)
#   1,  0,  0   => 1 (Carry: 0)
#   1,  1,  0   => 0 (Carry: 1)
#   0,  0,  1   => 1 (Carry: 0)
#   0,  1,  1   => 0 (Carry: 1)
#   1,  0,  1   => 0 (Carry: 1)
#   1,  1,  1   => 1 (Carry: 1)
#
# In some sense it's like the auto-regressive model, except that we don't need
# to concatenate the full history of events, but only the Carry. Essentially
# that carry neuron along with the input is enough to produce the output (and
# the new bit to carry over to the next timestep). The carry is the "memory" or
# state of the network spanning over multiple timesteps. This way we can
# maintain a "running-state" over the time-steps, instead of having to access
# all of the previous time-steps.
#
# A Recurent Neural Network is a system that's able to learn some internal
# state (in this case, simply the carry bit), learn when and how the state
# should be modified and finally access the state to determine the new value.
# One way to think about it is like having two different models: one for
# learning the recurrent state, and the other for learning the actual results
# normally.
import numpy as np
np.random.seed(1)

EPOCHS = 650
ALPHA = 0.005
H = 5

# input are a few examples of small binary strings. NOTE that while developing
# it, I've also experimented with randomly generating and endless stream of
# inputs (randint(255)) and managed to converge within a few hundred epochs. But
# I've decided to not include this code here, because the randomization code was
# pretty big and hairy and distracted from the core idea of how this algorithm
# works. Also NOTE that we're using a dot to separate the numbers. This allows
# the network to learn when the carry state should be reset between numbers,
# without us having to force it on the network superficially.
X1 = "00101.01100.01001.101.0.01100.01101.0110.010.01110.010.01011.01010.01101."
X2 = "01011.01000.01011.000.1.00111.01100.1001.100.00101.010.01100.01100.01110."
T  = "10000.10100.10100.101.1.10011.11001.1111.110.10011.100.10111.10110.11011."

# binary addition is done right-to-left. it's impossible to know the output of
# the addition of two bits unless we know the result (carry) from the bits to
# the right.
X1 = "".join(reversed(X1))
X2 = "".join(reversed(X2))
T  = "".join(reversed(T))

# finally one-hot encode
X1 = np.array([np.eye(3)["01.".index(c)] for c in X1])
X2 = np.array([np.eye(3)["01.".index(c)] for c in X2])
T  = np.array([np.eye(3)["01.".index(c)] for c in T])
X  = np.concatenate((X1, X2), axis=1) # both digits are part of the same input

class Recurrent(object):
    def __init__(self, n, m):
        # we're using different weights for the input and the hidden state,
        # Wx and Wh respectively. Just like in the auto-regressive model, we can
        # just concatenate the input x and the hidden state h and then use a
        # single set of weights in size (m, n + m + 1). But this will make the
        # backwards pass a bit more involved. See below.
        self.Wx = np.random.randn(m, n + 1) * 0.01
        self.Wh = np.random.randn(m, m) * 0.01

        # initial hidden state.
        self._h = np.zeros(m)

        # for back propagation, we'll need the full history of all inputs (x, h)
        # and outputs (y). See below.
        self._prevs = []

    def forward(self, x):
        h = self._h # current state of the network

        # this is the exactly the same as concatenating x and h, and then
        # passing it through a single set of weights. If we concatenate two
        # values x, h:          z = x * w1 + h * w2
        # This is the same as:  z = (x * w1) + (h * w2)
        #
        # Which is the form we're using here. Using a different set of weights
        # keeps back propagation a bit simpler.
        x = np.append(x, 1.)
        z = np.dot(self.Wx, x) + np.dot(self.Wh, h)
        y = np.tanh(z)

        # for backprop, we'll need the full history of both inputs (x, h) and
        # the output y.
        self._prevs.append((x, h, y))

        # over hundreds of epochs, we'll end up with thousands of previous
        # values that will (a) consume a huge amount of memory and (b) be slow
        # as we need to iterate over all of them while back-proping. We're
        # truncating the history to only the past 25 instances. This means that
        # our algorithm is limited as it will be unable to discover events that
        # only emerge after more than 25 time steps. It might be better to
        # control this truncation limit via a hyper param.
        self._prevs = self._prevs[-25:] # truncate to the last 25 time steps

        # finally, this layer stores and returns its state. This means that the
        # state doesn't only encapsulate the carry, but also the rest of the
        # meaningful information in its input.
        self._h = y
        return y

    def backward(self, dy, t = 1):
        # back propagation was the most difficult part for me to grasp. Finding
        # the derivatives of our weights involves a slight modification to our
        # normal back propagation algorithm, known as Back Propagation Through
        # Time (BPTT). But really, it's just normal back propagation. Here's
        # how I managed to make sense of it. Basically, the idea is that as we
        # produce new h values, we can keep track of every operation we
        # performed going from the initial h of zeros all the way to the current
        # time step. For example, lets review the forward pass of the first 2
        # inputs (x0, x1):
        #
        #   h0 = zeros
        #   z1 = dot(Wx, x1) + dot(Wh, h0)
        #   y1 = tanh(z1)
        #
        #   h1 = y1
        #   z2 = dot(Wx, x2) + dot(Wh, h1)
        #   y2 = tanh(z2)
        #
        # Now, if we want to back propagate to find the derivative wrt to Wx
        # and Wh:
        #
        #   dy2 = <- given from next layer (softmax)
        #   dz2 = (1 - y2 ** 2) * dy        # backprop tanh
        #   dwx = [x2 * d for d in dz2]
        #   dwh = [h1 * d for d in dz2]
        #
        # Normally, we'd be done. But because of how h1 was computed, it's clear
        # that h1 is also depedent on Wx and Wh. So, unlike x1 which is just a
        # flat input, we need to continue back propagating through time to the
        # previous time-step that produced h1. See above how h1 was initially
        # computed:
        #
        #   z1 = dot(Wx, x1) + dot(Wh, h0)
        #   h1 = y1 = tanh(z1)
        #
        # So we need to back propagate:
        #
        #   dh1 = dot(Wh, dz2)
        #   dz1 = (1 - y1 ** 2) * dh1
        #   dwx = [x1 * d for d in dz1]
        #   dwh = [h0 * d for d in dz1]
        #
        # So we've now computed dwx and dwh twice: once per time-step. We can
        # sum these modifications to find the accumulated influence of Wx and Wh
        # on the final error through all of the time steps. Of course, with more
        # previous events, we'll have more such iterations to back prop. In fact
        # for every time t, we'll need t iterations. So in total, we'll need 1 +
        # 2 + 3 + 4 + ... iterations. By the 100th input, we'll need a total of
        # 4950 iterations. Over time, our learning code will slow to a halt.
        # This is why it's advised to truncate the history and not keep it
        # endless as it grows very big and slow.
        if t > len(self._prevs):
            # stopping condition. we've finished going back in time up to our
            # truncation limit.
            return

        # normal derivation, only for a specific point in time (-t).
        x, h, y = self._prevs[-t]
        dz = (1 - y ** 2) * dy
        dwx = np.array([d * x for d in dz])
        dwh = np.array([d * h for d in dz])
        dx  = np.dot(dz, self.Wx)
        dh  = np.dot(dz, self.Wh)

        self.Wx += ALPHA * -dwx
        self.Wh += ALPHA * -dwh

        # backprop to the previous point in time recursively. we don't care
        # about the returned derivative of x for the previous timestamp. We only
        # need to update the weights proportionaly to the sum of all of the
        # derivatives.
        self.backward(dh, t + 1)

        # only the initial caller, at the first timestep (t=1) is really
        # concerned about the return value. We may choose avoid computing dx
        # unless t == 1.
        return np.delete(dx, -1)

class Linear(object):
    def __init__(self, n, m):
        self.W = np.random.randn(m, n + 1) * 0.01

    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        y = np.dot(self.W, x) # derivate: x
        self._x = x
        return y

    def backward(self, dy):
        x = self._x
        dw = np.array([d * x for d in dy])
        dx = np.dot(dy, self.W)
        self.W -= ALPHA * dw
        return np.delete(dx, -1)

class Softmax(Linear):
    def forward(self, x):
        y = super(Softmax, self).forward(x)
        exps = np.exp(y - np.max(y))
        p = exps / np.sum(exps)
        self._p = p
        return p

    def error(self, t):
        p = self._p
        return -np.log(p[np.argmax(t)])

    def backward(self, t):
        p = self._p
        dy = p - t
        return super(Softmax, self).backward(dy)


l1 = Recurrent(6, H)
l2 = Softmax(H, 3)
layers = [l1, l2]
data = zip(X, T)
for i in xrange(EPOCHS):
    e = 0.
    accuracy = 0

    for x, t in data:

        # forward
        y = reduce(lambda x, l: l.forward(x), layers, x)

        # backward
        e += layers[-1].error(t)
        d = reduce(lambda d, l: l.backward(d), reversed(layers), t)

        # update the accuracy
        accuracy += 1 if np.argmax(t) == np.argmax(y) else 0

    e /= len(data)
    accuracy = accuracy * 100 / len(data)
    print "%s: ERROR = %s ; ACCURACY = %s%%" % (i, e, accuracy)
