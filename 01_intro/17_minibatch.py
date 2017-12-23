# The example before managed to find the perfect weights in just a few hundred
# epochs because the inputs were perfectly correlated to the outputs. In real
# life, this is rarely the case. Usually there will be some "noise" in the data,
# some outliers that doesn't fit perfectly. This is because it can be assumed
# that the inputs doesn't capture every possible feature in the problem domain.
# For example, if our inputs represent the number of items purchased, and the
# target output represents the cost of these items, it's possible to have
# outliers where the data doesn't fit perfectly, due to special promotions,
# sales, surges, negotiations, etc. that are not captured in the data inputs.
import numpy as np
np.random.seed(1)

ALPHA = 1
EPOCHS = 200

# In this simple example, we'll use a single binary weight. Our target will be
# the same as the input, except that we'll have an outlier: the last instance
# returns a wrong value. NOTE that obviously it means that a perfect loss of
# zero cannot be obtained, but still the network should learn the correlation.
X = np.array([[0.], [0.], [1.], [1.], [0.], [0.], [1.], [1.]])
T = np.array([[0.], [0.], [1.], [1.], [0.], [0.], [1.], [0.]]) # outlier last!
#                                                        ^^

# If we run out previous code as is, these outliers will have a significant
# influence over our learning, because they're given a full STEP in their
# graident direction, resulting in an error that than has to be fixed. Instead,
# here we'll use mini-batches to average out this "noise", by calculating the
# average derivative over a small mini-batch of the entire data, and then
# applying the STEP-sized learning for this average. This might seem like it
# slows learning a little bit, because we're only applying the STEP update twice
# for each epoch, but in reality it's better because there's less error-
# correction.
#
# NOTE that when we use fully-batched learning (BATCH = len(X)), it must be the
# case that the avg loss will go down between epochs. Otherwise, our leanring
# rate is too high and we're oscilating or diverging. This is useful for finding
# the right starting learning rate and ensuring that the algorithm works
# properly. This is not the case for online (BATCH = 1) or minibatches, because
# a single batch might contain outliers that pushes the weights in the wrong
# direction for future batches.
#
# Non-batched:   BATCH = 1
# Fully-batched: BATCH = len(X)
# Mini-batching: Anything in between  1 < BATCH < len(x)
BATCH = 4

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.N = n
        self.M = m
        self.W = np.random.random((m, n + 1)) # +1 bias

    # forward pass
    # now accepts a list of BATCH inputs, xs - one per test case - each with N
    # values - one per input neuron. Computes a list of BATCH outputs, ys.
    def forward(self, xs):

        # add the bias in one go to all of the data by creating a new vector of
        # 1s and then concatenating it with the input xs.
        bias = np.ones((xs.shape[0], 1)) # can be a constant
        xs = np.concatenate((xs, bias), axis=1)

        # BATCH x M outputs
        ys = np.zeros((len(xs), self.M))
        for i, x in enumerate(xs):
            z = np.dot(self.W, x) # derivate: x
            y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)
            ys[i] = y

        ys = ys
        self._last = xs, ys
        return ys

    # backward pass
    # now accepts a list of BATCH dys - one per test case - each with M values -
    # one per output neuron. Computes the average dw for all of these cases and
    # updates once for that average. It also returns a list of BATCH dxs for
    # backprop.
    def backward(self, dys):
        xs, ys = self._last
        dxs = np.zeros((len(dys), self.N)) # BATCH x N input derivatives
        dws = np.zeros((len(dys), self.N + 1)) # BATCH x N+1 weight derivatives
        for i, dy in enumerate(dys):
            x = xs[i]
            y = ys[i]

            # how the weights affect total loss (derivative w.r.t w)
            dz = dy * (y * (1 - y))
            dw = np.array([d * x for d in dz])
            dws[i] = dw

            # how the input (out of previous layer) affect total loss (derivative
            # w.r.t x). Derivates of the reverse of the forward pass.
            dx = np.dot(dz, self.W)
            dx = np.delete(dx, -1) # remove the bias input derivative
            dxs[i] = dx

        # update
        dw = sum(dws) / len(dws) # average out the weight derivatives
        self.W -= ALPHA * dw
        return dw, dx

l = Layer(1, 1) # linear problem, no hidden layer.
indices = range(len(X))
for i in xrange(EPOCHS):
    np.random.shuffle(indices)

    e = 0.
    for j in xrange(0, len(indices), BATCH):
        # choose the indices in the data (X, T) to be used in this mini-batch
        minib = indices[j:j+BATCH]

        # forward
        xs = X[minib]
        ts = T[minib]
        ys = l.forward(xs)

        # backward
        e += sum((ys - ts) ** 2 / 2)
        dys = ys - ts
        l.backward(dys)

    e /= len(X)
    print "%s: LOSS = %s" % (i, sum(e))
