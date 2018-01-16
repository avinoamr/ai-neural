# One of the greatest challenges I had when I started delving into multi-layered
# networks is the intuition and understanding of the precise behavior exhibited
# by the hidden layer. It felt more of an experimental property that I needs to
# be tuned instead of an accurate property that I can reason about. This
# exercise is an attempt at understaing this property better. We will also
# re-introduce the entire learning process instead of just tinkering with one
# data instance.
#
# In regression, a hidden layer can be thought of as composition of several
# simple non-linear functions building up into a single arbitrarily complex
# non-linear function as shown in (1). But in classification, we can think of
# the output of the hidden layer as a set of new features dataset that's fed
# into the final layer and thus the whole purpose of the hidden layer is to find
# the features that are most relevant for decreasing the error. In other words,
# it needs to pick the features that are most interesting/relevant for the
# prediction, using any combination of the input features.
#
# (1) http://neuralnetworksanddeeplearning.com/chap4.html
import numpy as np
np.random.seed(1)

ALPHA = 1
EPOCHS = 1500
H = 2 # number of hidden neurons

# In this example we're intentionally designing the data such that it's not
# linearily separable. Thus there's no possible set of weights that can
# perfectly predict the result. This can obviously happen in the real-world,
# where one feature may produce different results, depending on the activation
# on a different feature. We'll use the typical XOR example.
#
# XOR = !(x1 AND x2) AND (x1 OR x2)
X = np.array([ [0., 0.], [0., 1.], [1., 0.], [1., 1.] ])
T = np.array([ [0.],     [1.],     [1.],     [0.]     ])

class Sigmoid(object):
    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1))

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
        return dx

class SquaredError(object):
    def forward(self, x):
        self._y = x
        return x

    def error(self, t):
        y = self._y
        return (y - t) ** 2 / 2

    # squared error function just returns the simple derivative
    def backward(self, t):
        y = self._y
        return y - t

# we'll use a hidden layer of H neurons and examine the learned weights
data = zip(X, T)
l1 = Sigmoid(X.shape[1], H)
l2 = Sigmoid(H, T.shape[1])
l3 = SquaredError()
layers = [l1, l2, l3] # try removing l2 to see that it's unable to learn
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0.
    accuracy = 0
    for x, t in data:
        # forward
        y = reduce(lambda x, l: l.forward(x), layers, x)

        # backward
        e += l3.error(t)
        d = reduce(lambda d, l: l.backward(d), reversed(layers), t)

        # we're rounding again for accuracy calculation because I didn't want
        # to have multi-class inputs and outputs.
        accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

    e /= len(data)
    print "%s: ERROR = %s ; ACCURACY = %s" % (i, sum(e), accuracy)

print


# print all of the data for fun.
for x, t in data:
    h = l1.forward(x)
    y = l2.forward(h)
    print "x=", x
    print "h=", h
    print "y=", y
    print "t=", t
    print

# NOTE that if we remove the hidden layer, we obviously can't reach an error
# close to zero because the problem was designed such that no set of weights can
# describe the data perfectly.
#
# But if we look at the weights of the hidden layer, we'll be able to see what
# the network has learned:
print "l1.W=", l1.W

# l1.W = [
#   [-3.43359201 -3.42420851  4.94561233]
#   [-5.99802997 -5.89528226  2.03292505]
# ]
#
# Remember from classification (that although it's not mathmatically accurate
# due to the way multi-class predictions are made) we can think of the boolean
# output as: wx > -b. We've learned the following boolean expressions in the
# hidden layer:
#
#   h1 = !x1 OR  !x2        => Same as: !(x1 AND x2)
#   h2 = !x1 AND !x2        => Same as: !(x1 OR x2)
#
# This isn't XOR yet, so let see what the second layer is doing:
print "l2.W=", l2.W

# l2.W = [
#   [ 6.7717498  -7.26318539 -2.96906094]
# ]
#
# Again, if we apply wx > -b:
#
#   y = h1 AND !h2
#       ^^      ^^              # replace with the expressions we defined above
#
#   y = !(x1 AND x2) AND !(!(x1 OR x2))
#                        ^^^^   # remove the double negatives
#
#   y = !(x1 AND x2) AND (x1 OR x2)
#
# That's the definition of XOR!
