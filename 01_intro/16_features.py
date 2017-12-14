# Lets try to generalize our previous findings to determine how does the number
# of hidden neurons affects the overall combinations that can be discovered.
# Each such neuron is defined by a bias, which is kind of like a threshold or
# baseline activation and then N weights that has to inhibit or excite beyond
# that bias baseline:  out = [W1, W2, ..., b]
#
# Since we're going through the sigmoid activation, the weights needs to
# overcome the bias by atleast 0.5 in order to fully excite the output. This is
# a big over-simplification, because we're using the sigmoid activation which
# would be closer to zero if the summed weights is below 0.5, so that's really
# the value by which the bias needs to be overcome. Using tanh would produce a
# more reasonable distribution with the mean at 0 instead of 0.5.
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
# reach a loss of zero. That gives us the intuition that any arbitrarily complex
# classification problem can be solved with hidden layers. The question that I'm
# still unable to fully answer, is why would we ever need more than 1 layer,
# beyond the need to add different logic (CNN, RNN, dropout, softmax, etc.)
#
# NOTE that of course this is horribly inefficient (16 Inputs => 65,536 hidden
# neurons), non-elegant and non-intelligent as it doesn't really learn
# complicated logic but simply remembers every possible input. A hash table will
# probably be more efficient than a neural network like that. NOTE that this
# code should never be used in reality. It's only here for experimental &
# educational purposes.
import numpy as np
np.random.seed(1)

ALPHA = 3
EPOCHS = 1000

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
# we'll always be able to fully eliminate the loss.
T = np.random.choice([0., 1.], len(X))

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1)) # +1 bias

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)

        self._last = x, y
        return y

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

# we'll use a hidden layer of 2^N neurons to cover every possible combination
# of the input. One neuron per combination of the 4 input neurons.
data = zip(X, T)
l1 = Layer(4, 2 ** 4)
l2 = Layer(2 ** 4, 2)
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0.
    for x, t in data:
        # forward
        y = l1.forward(x)
        y = l2.forward(y)

        # backward
        e += (y - t) ** 2 / 2
        d = y - t
        _, d = l2.backward(d)
        _, d = l1.backward(d)

    e /= len(data)
    print "%s: LOSS = %s" % (i, sum(e))
