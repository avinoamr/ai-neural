# As we've seen before, the inputs weren't always perfectly correlated to the
# outputs, and we needed to compute the full distribution of probabilities to
# account for all cases. In real life, this is often the case. Usually there
# will be some "noise" in the data, some outliers that doesn't fit perfectly.
# This is because it can be assumed that the inputs doesn't capture every
# possible feature in the problem domain. For example, if our inputs represent
# the number of items purchased, and the target output represents the cost of
# these items, it's possible to have outliers where the data doesn't fit
# perfectly, due to special promotions, sales, surges, negotiations, etc. that
# are not captured in the data inputs.
#
# Our current set up, while works, is somewhat inefficient and might have
# difficulty to converge, by the fact that it gives every data point an equally
# large influence over the learned weights. This also applies to the outliers,
# thus forcing the weights to take a big step in the incorrect direction, only
# to be fixed by the subsequent correct examples.
#
# An optimization process we'll examine here is to average out all of the data
# points in order to find the average derivative of the error function across
# all data points, and only applying an update step on that average. The result
# due to averaging is a smoother reduction in the error function over time, with
# significantly less fluctuations. This commonly leads to a faster, more
# convergent learning and is thus used wildly in machine learning.
#
# NOTE that when using fully-batched learning (see below), it can be expected
# that the error will always decrease between epochs, as all of the outliers are
# always averaged out - there's no single data point or epoch that pushes us,
# even slightly, in the wrong direction. Otherwise, the learning rate ALPHA
# is too big, or the initial weights are configured incorrectly.
import numpy as np
np.random.seed(1)

ALPHA = 1
EPOCHS = 85

# we'll use the same data as in the softmax exercise, and you'll notice that we
# achieve a more accurate distribution using less than half of the epochs.
X = np.array([[1.],     [1.],     [1.],     [1.],     [1.]])
T = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]])
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
# Online non-batched: BATCH = 1
# Fully-batched:      BATCH = len(X)
# Mini-batching:      1 < BATCH < len(x)
BATCH = len(X)

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.N, self.M = n, m
        self.W = np.random.randn(m, n + 1) * 0.01

    # forward pass
    # Same as before, except now the argument is a list of input x vectors, each
    # of size N. The return value is a list of output y vectors, each of size M.
    # We iterate over all of the inputs, and perform the same logic we had
    # before. Thus this function now handles a batch of inputs in one call.
    def forward(self, xs):
        bias = np.ones((xs.shape[0], 1))
        xs = np.concatenate((xs, bias), axis=1)
        ys = np.zeros((len(xs), self.M))
        for i, x in enumerate(xs):
            z = np.dot(self.W, x)
            y = np.tanh(z)
            ys[i] = y

        self._last = xs, ys
        return ys

    # backward pass
    # Same as before, except now the argument is a list of error derivatives,
    # each of size M. It computes the derivatives with respect to w and applies
    # a single weights update for the average of derivatives. The return value
    # is a list of derivatives of function's input xs vectors, each of size M,
    # to be used by back propagation. Thus this function now handles a batch of
    # error derivatives  in one call.
    def backward(self, dys):
        xs, ys = self._last
        dxs = np.zeros((len(dys), self.N))
        dws = np.zeros((len(dys),) + self.W.shape)
        for i, dy in enumerate(dys):
            x = xs[i]
            y = ys[i]

            # how the weights affect total error (derivative w.r.t w)
            dz = dy * (1 - y ** 2)
            dw = np.array([d * x for d in dz])
            dws[i] = dw

            # how the input (out of previous layer) affect total error
            # (derivative w.r.t x). Derivates of the reverse of the forward pass
            dx = np.dot(dz, self.W)
            dx = np.delete(dx, -1) # remove the bias input derivative
            dxs[i] = dx

        # update
        dw = sum(dws) / len(dws) # average out the weight derivatives
        self.W -= ALPHA * dw
        return dxs

l = Layer(1, 2) # linear problem, no hidden layer.
indices = range(len(X)) # list of indices in
for i in xrange(EPOCHS):
    np.random.shuffle(indices) # shuffle the list of different ordering

    e = 0.
    dist = 0.
    for j in xrange(0, len(indices), BATCH):
        # choose the BATCH indices in the data (X, T) to be used in this mini-
        # batch
        minib = indices[j:j+BATCH]
        xs = X[minib]
        ts = T[minib]

        # forward
        ys = l.forward(xs)
        ps = np.exp(ys) / np.sum(np.exp(ys), axis=1, keepdims=True) # softmax

        # backward
        e += sum(ts * -np.log(ps))
        dys = ps - ts
        l.backward(dys)

        # compute the distribution
        dist += ps

    dist = sum(dist) / len(X) # average out the probablity distribution
    e = sum(e) / len(X)
    print "%s: ERROR = %s ; DIST = %s" % (i, e, dist)
