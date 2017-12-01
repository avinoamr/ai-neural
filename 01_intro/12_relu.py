# While the sigmoid function works for many cases, there are other activation
# functions worth discussing: first is the hyperbolic function (tanh) which
# is very similar to sigmoid:
#
#   y = tanh(z) = 2 * sigmoid(2z) - 1)
#   dy/dz = tanh'(z) wrt z = 1 - y^2   # look it up on line to learn more
#
# It has a interesting property that it's mean is at zero (when z = 0, y = 0,
# instead of y = .5 in sigmoid). We will not implement it here as it's a bit
# redundant.
#
# Second is the Rectifier Linear Unit (ReLU):
#
#   y = max(0, z)
#   dy/dz = { 1 when z > 0 ; 0 otherwise }
#
# It's basically just like having no activation function, except that negative
# values are clamped at zero. A few things to note:
#   1. It's still non-linear (below and above 0) and proven to be able to
#       approximate any non-linear function when combined in multiple layers.
#   2. It has the range of [0...inf] which means that it might be a bit
#       inefficient when used with the squared difference loss function (it's
#       more often used with softmax/cross-entropy - discussed later).
#   3. It generates sparse activations. Because negative y values are clamped at
#       zero, many of the neurons (the irrelevant ones that do not contribute to
#       the prediction of a given class) are not going to fire. This makes the
#       model more elegant, simpler and easier to debug because we don't have a
#       lot of negative numbers - we're only seeing the ones that actually
#       affect the output. In a multi-layered network, it means that the input
#       to the next layer is more likely to have a lot of zeros instead of a
#       dense activations where all of the neruons have some value.
#   4. Because the derivative is sometimes zero, it means that weights can get
#       stuck at zero and stop responding to errors. This is called the dying
#       ReLU problem and there are several variations, like Leaky ReLU that help
#       prevent that (at a cost of sparsity)
import numpy as np

STEP = .001
EPOCHS = 1000
BATCHSIZE = 1

# Say we want to build a simple program to dechiper code. We're given a few
# words in the input and their corresponding dechipered text as output. Each
# character will be a class
X = "croj dsmujlayfxjpygjtdwzbjyeoajcrojvkihjnyq*"
T = "the quick brown fox jumps over the lazy dog!"
data = zip(X, T)

class OneHot(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.N = len(alphabet)

    def encode(self, v):
        x = np.zeros(self.N)
        idx = self.alphabet.index(v)
        x[idx] = 1.
        return x

    def decode(self, y):
        return self.alphabet[np.argmax(y)]


# initialize & learn
inp = OneHot(list(set(X)))
out = OneHot(list(set(T)))
w = np.random.random((out.N, 1 + inp.N))
for i in xrange(EPOCHS):
    l = 0
    accuracy = 0

    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw = 0

        for v, target in minib:
            x = inp.encode(v)
            x = np.insert(x, 0, 1.)
            z = np.dot(w, x)
            y = np.maximum(0, z) # ReLU
            res = out.decode(y)
            accuracy += 1 if res == target else 0

            # loss and derivatives
            t = out.encode(target) # encode target string to one-hot activation
            l = (y - t) ** 2 / 2
            dy = y - t

            # Now we need to derive the ReLU activation, and chain it with the
            # error derivative wrt to y (dy)
            #
            # Recall: dy/dz = { 1 when z > 0 ; 0 otherwise }
            dz = dy * np.greater(z, 0).astype(float)

            # and finally, same as before, chain with dz/dw:
            dw += np.array([dyi * x for dyi in dz])

        dw /= len(minib)
        w += STEP * -dw

    l = sum(l) / len(data)
    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))


# print w
