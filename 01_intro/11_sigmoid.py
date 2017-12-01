# Classification is much more involved than simply encoding/decoding classes
# before feeding it into a normal linear regression process. One reason for the
# difference is the fact that unlike linear regression, for classification we
# don't care about the absolute output values, but only the predicted classes
# in relative terms (ie which output value has the highest activation). Consider
# the following example that shows the discrepency:
#
#   y = [-5, 5]     t = [0, 1]      loss = [25, 16]
#
# The classes would be predicted correctly, while the loss is high. The
# derivatives of this loss will cause y[0] to go up towards 0, and y[1] to go
# down towards 1 - exactly in the opposite direction of the correct prediction.
# This is because the range of the output is [-inf .. +inf] while the range of
# the target is [0..1].
#
# In the next few exercises, we'll explore several activation functions to help
# model our output more elegantly. An activation function defines the output of
# a node given its inputs. It allows use to add non-linear operations to make
# the input fit within the constraints we've set up. In this example, we'll
# explore the sigmoid function that squashes the weighted values (z) into the
# desired range: [0..1]:
#
#   y = sigmoid(z) = 1 / (1 + exp(-z))
#   dy/dz = sigmoid'(z) wrt z = y * (1 - y)   # look it up on line to learn more
#
# This function has the interesting property that it's an S-curved function that
# goes towards 0 for infinitely negative z, and goes towards 1 for infinitely
# positive z. So it squashes any z value to [0..1]. So now:
#
#   z = [-5, 5]     sigmoid(z) = [0.006, 0.993]     t = [0, 1]  loss = [0, 0]
#
# Much better! If anything, derivatives would stay unchanged, or get better.
#
# A NOTE about non-linearity: everything we've done until now is obviously
# limited to linear problems. For example, our code cannot learn to approximate
# the function f(x) = x^2 with any possible set of weights. One way around it
# is to expand our features domain such that generate new inputs that are non-
# linear, like x1 = x^1, x2 = x^2, x3 = x^3, and then learn the weights as if
# that was the original input. But this still wouldn't cover any non-linear
# function (sin, exp, log, etc.). A better approach which we'll explore later
# is multi-layered networks that are able to combine multiple functions into
# a single one. But you'll notice that still combining many linear functions can
# only produce a linear - so the problem isn't solved. A good activation
# function is one that adds non-linearity to the model by allowing combinations
# of non-linear functions to approximate any non-linear function. We'll revisit
# this issue in the later on, but for now it's important to note that this
# non-linearity is the main purpose of activation functions, more than squashing
# for example.
import numpy as np

STEP = .1
EPOCHS = 100

# Say we want to build a simple program to dechiper code. We're given a few
# words in the input and their corresponding dechipered text as output. Each
# character will be a class
X = "croj dsmujlayfxjpygjtdwzbjyeoajcrojvkihjnyq*"
T = "the quick brown fox jumps over the lazy dog!"
data = zip(X, T)

# since we know that the data has no noise, using mini-batches will
# significantly slow down learning. We're turning it off by using a batch size
# of 1 which means that the weights will be updated for every single observation
# in the data, similar to online learning. If our data was based on a real-life
# text that might have some imperfections - bigger batches would've been able
# to cancel out that noise.
BATCHSIZE = 1

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
w = np.zeros((out.N, 1 + inp.N))
for i in xrange(EPOCHS):
    l = 0
    accuracy = 0

    for j in xrange(0, len(data), BATCHSIZE):
        minib = data[j:j+BATCHSIZE]
        dw = 0

        for v, target in minib:
            x = inp.encode(v)
            x = np.insert(x, 0, 1.)

            # predict, and before decoding, we'll squash the weighted values
            # with the sigmoid activation function
            z = np.dot(w, x)
            y = 1. / (1. + np.exp(-z)) # in the range of [0..1]
            res = out.decode(y)
            accuracy += 1 if res == target else 0

            # loss and derivatives
            t = out.encode(target) # encode target string to one-hot activation
            l = (y - t) ** 2 / 2
            dy = y - t

            # now - because we've added a term to the forward pass (computing
            # the output / predicting), we need to symmetrically derive that
            # newly added expression. We need to find (1) how y changes when z
            # changes (dy/dz), and then use the chain rule to combine that
            # affect with (2) dloss/dy to deterine how the loss changes when we
            # change z (dE/dz). But because (2) was already computed above,
            # we're only left with the derivative of the sigmoid function itself
            #
            # NOTE the theme here: every expression we add to the forward
            # computation pass, must be derived in reverse order on the backward
            # computation pass (derivation) while chaining all of these partial
            # derivatives.
            #
            # Recall: sigmoid'(z) wrt z = y * (1 - y)
            dz = dy * (y * (1 - y))

            # and finally, same as before, chain with dz/dw:
            dw += np.array([dyi * x for dyi in dz])

        dw /= len(minib)
        w += STEP * -dw

    l = sum(l) / len(data)
    print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))

# decipher another message
X = "scjfyaub*"
result = ""
for v in X:

    # copy-paste of the forward pass. TODO: DRY
    x = inp.encode(v)
    x = np.insert(x, 0, 1.)
    z = np.dot(w, x)
    y = 1. / (1. + np.exp(-z))
    result += out.decode(y)

print
print X + " = " + result
print