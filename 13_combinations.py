# Lets try to generalize our previous findings to determine how does the number
# of hidden neurons affects the overall combinations that can be discovered.
# Each such neuron is defined by a bias, which is kind of like a threshold or
# baseline activation and then N weights that has to inhibit or excite beyond
# that bias baseline:  out = [W1, W2, ..., b]
#
# Since we're going through the sigmoid activation, the weights needs to
# overcome the bias by atleast 0.5 in order to fully excite the output. Using
# tanh would produce a more reasonable distribution with the mean at 0 instead
# of 0.5.
#
# We've seen that a hidden neuron can learn the AND and OR logic gates. This can
# extended to more complicated expressions. Here are a few (NOTE that here we're
# simplifying as if we were using the tanh activation where the mean tipping
# point is at 0. In sigmoid, the mean is at 0.5):
#
#    w1  w2  w3  b    =>  EXPRESSION
#    0   0   1   0        (w3)                 # IDENTITY
#    0   0  -1   1        (!w3)                # NOT
#    0   1   1  -1        (w2 || w3)           # OR
#    1   1   1  -1        (w1 || w2 || w3)     # OR
#    0   1   1  -2        (w2 && w3)           # AND
#    1   1   1  -3        (w1 && w2 && w3)     # AND
#    1   1   2  -2        (w1 && w2) || w3
#    1   1   2  -3        (w1 || w2) && w3
#    1   1   1  -2        (w1 && w2) || (w2 && w3) || (w1 && w3)   # 2 out of 3.
#    1  -1  -1  -1        (w1 && !w2 && !w3)
#    1   1  -2  -2        (w1 && w2 && !w3)
#   -1   1   1  -2        (!w1 && w2 && w3)
#    0  -1  -1   1        !(w2 || w3)          # NOR
#   -1  -1  -1   1        !(w1 || w2 || w3)    # NOR
#    0  -1  -1   2        !(w2 && w3)          # NAND
#   -1  -1  -1   3        !(w1 && w2 && w3)    # NAND
#    1  -1  -1   2        !(!w1 && w2 && w3)
#   -2  -2  -3   3        !((w1 && w2) || w3)
#
# So the question is - how many different such expressions (hidden neurons) we
# need to fully map the data? Obviously, at most there are 2^N different
# combinations of the N inputs:
#
#    w1 &&  w2 &&  w3
#    w1 &&  w2 && !w3
#    w1 && !w2 &&  w3
#    w1 && !w2 && !w3
#   !w1 &&  w2 &&  w3
#   !w1 &&  w2 && !w3
#   !w1 && !w2 &&  w3
#   !w1 && !w2 && !w3
#
# This is proven here by choosing a completely random output for every possible
# combination of 4 input weights - and as we'll see, it's always possible to
# reach an error of zero. That gives us the intuition that any arbitrarily
# complex classification problem can be solved with hidden layers. The question
# that I'm still unable to fully answer is why would we ever need more than 1
# layer, beyond the need to add different logic (CNN, RNN, dropout, softmax,
# etc.)
#
# That's obviously inefficient (16 Inputs => 65,536 hidden neurons). In fact,
# most real-life cases will not have nearly as many samples in the training data
# set - so the upper-bound of the number of combinations is:
#
#   Upper-Bound: min(2^N, # of samples)
#
# But, beyond being inefficient, it's also non-elegant and non-intelligent as it
# doesn't really learn complicated logic but simply remembers every possible
# input.
#
# More importantly, in a setup like that, the system is more likely to over-fit
# by the fact that it will tend to remember all of the training data and wouldn't
# generalize correctly. This to me seems like another limitation of artificial
# neural netowkrs, compared to biological ones, because adding the intelligence
# (number of neurons/synapses) would result in loss of intelligence when
# confronted with new data. Instead of learning the least amount of patterns to
# understand the data, our learning procedure will conviniently learn the most
# possible patterns - losing generalization in the process.
#
# By following Ockham's Razor - we're searching for the simplest model as it's
# most likely the best. The subject of how to achieve this is not explored here
# at this time, but involves many techniques: like regularization, pruning,
# growing, global searches, etc. For now, the basic intuition is that we want
# to find the least amount of hidden neurons that will correlate the data at an
# acceptable accuracy.
import numpy as np
np.random.seed(1)

ALPHA = 1
EPOCHS = 1300
H = 2 ** 4 # 2^N different combinations.

X = np.array([
    [ 0., 0., 0., 0. ],
    [ 0., 0., 0., 1. ],
    [ 0., 0., 1., 0. ],
    [ 0., 0., 1., 1. ],
    [ 0., 1., 0., 0. ],
    [ 0., 1., 0., 1. ],
    [ 0., 1., 1., 0. ],
    [ 0., 1., 1., 1. ],
    [ 1., 0., 0., 0. ],
    [ 1., 0., 0., 1. ],
    [ 1., 0., 1., 0. ],
    [ 1., 0., 1., 1. ],
    [ 1., 1., 0., 0. ],
    [ 1., 1., 0., 1. ],
    [ 1., 1., 1., 0. ],
    [ 1., 1., 1., 1. ],
])

# random output - there's no intelligence here, but with 2^N hidden neurons
# we'll always be able to fully eliminate the error.
T = np.random.choice([0, 1], len(X))
T = np.eye(2)[T]

# Layer represents a single neural network layer of weights
class Sigmoid(object):
    def __init__(self, n, m):
        self.W = np.random.randn(m, n + 1)

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z))

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total error (derivative w.r.t w)
        dz = dy * (y * (1 - y))
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total error (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        self.W -= ALPHA * dw
        return dw, dx

# we'll use a hidden layer of 2^N neurons to cover every possible combination
# of the input. One neuron per combination of the 4 input neurons.
l1 = Sigmoid(4, H)
l2 = Sigmoid(H, 2)

data = zip(X, T)
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0.
    accuracy = 0
    for x, t in data:
        # forward
        y = l1.forward(x)
        y = l2.forward(y)

        # backward
        e += (y - t) ** 2 / 2
        d = y - t
        _, d = l2.backward(d)
        _, d = l1.backward(d)

        accuracy += 1 if np.argmax(y) == np.argmax(t) else 0

    e /= len(data)
    print "%s: ERROR = %s ; ACCURACY = %s of %s" % (i, e, accuracy, len(data))
