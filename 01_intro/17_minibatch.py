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

ALPHA = 0.1
EPOCHS = 1000000
H = 2 ** 4 # 2^N different combinations.

# In this simple example, we'll use a single binary weight. Our target will be
# the same as the input, except that we'll have an outlier: the last instance
# returns a wrong value. NOTE that obviously it means that a perfect loss of
# zero cannot be obtain, but still the network should learn the correlation.
X = np.array([[0.], [0.], [1.], [1.], [0.], [0.], [1.], [1.]])
T = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 0.]]) # last index outlier!

# If we run out previous code as is, these outliers will have a significant
# influence over our learning, because they're given a full STEP in their
# graident direction, resulting in an error that than has to be fixed. Instead,
# here we'll use mini-batches to average out this "noise", by calculating the
# average derivative over a small mini-batch of the entire data, and then
# applying the STEP-sized learning for this average. This might seem like it
# slows learning a little bit, because we're only applying the STEP update twice
# for each epoch, but in reality it's better because there's less error-
# correction.
BATCH = 2 # len(X) / 2 # = 1 <- try this for non-batch for comparison

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1)) # +1 bias

    # forward pass is the same as before.
    def forward2(self, xs):
        xs2 = []
        for x in xs:
            x = np.append(x, 1.)
            xs2.append(x)

        xs = np.array(xs2)
        ys = []
        for x in xs:
            z = np.dot(self.W, x) # derivate: x
            y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)
            ys.append(y)

        ys = np.array(ys)

        self._last = xs, ys
        return ys

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward2(self, dys):
        # dy = dys[0]
        xs, ys = self._last

        dxs = []
        dws = []
        for i, dy in enumerate(dys):
            x = xs[i]
            y = ys[i]

            # how the weights affect total loss (derivative w.r.t w)
            dz = dy * (y * (1 - y))
            dw = np.array([d * x for d in dz])
            dws.append(dw)

            # how the input (out of previous layer) affect total loss (derivative
            # w.r.t x). Derivates of the reverse of the forward pass.
            dx = np.dot(dz, self.W)
            dx = np.delete(dx, -1) # remove the bias input derivative
            dxs.append(dx)

        # update
        dw = sum(dws) / len(dws)
        self.W -= ALPHA * dw
        return dw, dx

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total loss (derivative w.r.t w)
        dz = dy * (y * (1 - y))
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total loss (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        self.W -= ALPHA * dw
        return dw, dx

l1 = Layer(1, 2)
indices = range(len(X))
data = zip(X, T)
for i in xrange(EPOCHS):
    np.random.shuffle(indices)
    # np.random.shuffle(data)

    e = 0.
    for j2 in xrange(0, len(indices), BATCH):
        minib = indices[j2:j2+BATCH]
        xs = X[minib]
        ts = T[minib]

        ys = l1.forward2(xs)
        e += sum((ys - ts) ** 2 / 2)
        ds = ys - ts

        l1.backward2(ds)

    e /= len(data)
    print "%s: LOSS = %s" % (i, sum(e))
