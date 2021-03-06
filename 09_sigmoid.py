# Classification is much more involved than simply encoding/decoding classes
# before feeding it into a normal linear regression process. One reason for the
# difference is the fact that unlike linear regression, for classification we
# don't care about the absolute output values, but only the predicted classes
# in relative terms (ie which output value has the highest activation). Consider
# the following example that shows the discrepency:
#
#   y = [-5, 5]     t = [0, 1]      error = [25, 16]
#
# The classes would be predicted correctly, while the error is high. The
# derivatives of this error will cause y[0] to go up towards 0, and y[1] to go
# down towards 1 - exactly in the opposite direction of the correct prediction.
# This is because the range of the output is [-inf .. +inf] while the range of
# the target is [0..1].
#
# An activation function defines the output of a node given its inputs. It
# allows use to add non-linear operations to make the input fit within the
# constraints we've set up. In this example, we'll explore the sigmoid function
# that squashes the weighted values (z) into the desired range: [0..1]:
#
#   y = sigmoid(z) = 1 / (1 + exp(-z))      # z is the net input (sum of w * x)
#   dy/dz = y * (1 - y)                     # look it up on line to learn more
#
# This function has the interesting property that it's S-curved. It goes towards
# 0 for infinitely negative z, and goes towards 1 for infinitely positive z. So
# it squashes any z value to [0..1]. So now:
#
#   z = [-5, 5]     sigmoid(z) = [0.006, 0.993]     t = [0, 1]  error = [0, 0]
#
# Much better! If anything, derivatives would stay unchanged, or get better.
#
# There are other types of activation functions worth mentioning, but will not
# be covered further here - for example: the hyperbolic function (tanh) which is
# a scaled version of sigmoid:
#
#   y = tanh(z) = 2 * sigmoid(2z) - 1
#   dy/dz = 1 - y^2   # look it up on line to learn more
#
# It has an interesting property that it's mean is at zero (when z = 0, y = 0,
# instead of y = .5 in sigmoid). We will not implement it here as it's a bit
# redundant.
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
np.random.seed(1)

ALPHA = .1
EPOCHS = 100

# Say we want to build a simple program to dechiper code. We're given a few
# words in the input and their corresponding dechipered text as output. Each
# character will be a class
X = "croj dsmujlayfxjpygjtdwzbjyeoajcrojvkihjnyq*"
T = "the quick brown fox jumps over the lazy dog!"
INPUTS = sorted(list(set(X)))
OUTPUTS = sorted(list(set(T)))

class OneHot(list):
    def encode(self, data):
        x = np.zeros((len(data), len(self)))
        for i, vs in enumerate(data):
            indices = [self.index(v) for v in sorted(vs)]
            x[i][indices] = 1.
        return x

# initialize & learn
X = OneHot(INPUTS).encode([[c] for c in X])
T = OneHot(OUTPUTS).encode([[c] for c in T])
data = zip(X, T)

# NOTE about the initial weights. When using sigmoid, it's best to start with
# very small weights (around zero), because when z = 0, the derivative of
# sigmoid is largest (0.25). if we would pick large initial weights (over 1-2),
# the sigmoid(1-2) function would be almost flat, limiting at 0 or 1, and thus
# its derivative will be close to 0. It will learn many iterations to move the
# weights enough for the weights to have a larger effect. This is an issue known
# as "Vanishing Gradients".
w = np.random.randn(len(OUTPUTS), len(INPUTS) + 1) * 0.01
for i in xrange(EPOCHS):
    e = 0
    accuracy = 0

    for x, t in data:
        x = np.append(x, 1.)

        # predict, and before decoding, we'll squash the weighted values
        # with the sigmoid activation function
        z = np.dot(w, x)
        y = 1. / (1. + np.exp(-z)) # sigmoid; in the range of [0..1]

        # error and derivatives
        e += (y - t) ** 2 / 2
        dy = y - t

        # now - because we've added a term to the forward pass (computing
        # the output / predicting), we need to symmetrically derive that
        # newly added expression. We need to find (1) how y changes when z
        # changes (dy/dz), and then use the chain rule to combine that
        # affect with (2) derror/dy to deterine how the error changes when we
        # change z (derror/dz). But because (2) was already computed above,
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
        dw = np.array([dyi * x for dyi in dz])
        w += ALPHA * -dw

        # decode the predicted value to determine equality/accuracy
        accuracy += 1 if np.argmax(y) == np.argmax(t) else 0

    e = sum(e) / len(data)
    print "%s: ERROR = %s; ACCURACY = %d of %d" % (i, e, accuracy, len(data))

# decipher another message
X = "scjfyaub*"
result = ""
for x in OneHot(INPUTS).encode([[c] for c in X]):
    # copy-paste of the forward pass.
    x = np.append(x, 1.)
    z = np.dot(w, x)
    y = 1. / (1. + np.exp(-z))
    result += OUTPUTS[np.argmax(y)]

print
print X + " = " + result
