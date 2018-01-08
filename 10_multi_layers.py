# Everything we've achieved until now can arguably be done with simple
# statistics and Perceptrons because it only applies to linear relationships
# between the input and output classes. But in order to generalize to more
# complex non-linear examples, we'll want to use multiple layers in our neural
# network. That's effectively a pipeline of weights, same as before, where the
# output of each layer is fed as the input to the next. The real input is fed to
# the bottom-most layer, and the output of the top-most layer is the final
# output of our network.
#
# In linear regression, we may want to approximate a function like: f(x) = x^2
# Our current model can't achieve that obviously. (1) One way around it is to
# pre-process our input to also contain non-linear variations of the input, like
# x^2, x^3, log(x), exp(x), etc. But it's unlikely we'll cover every possible
# non-linear function and parameters, and it will take a massive input size. (2)
# we can use the bias trick to feed in additional mathmatical components, like
# the exponent 2 - but still unlikely to cover everything. This still holds for
# classification problems where we may want something that's more like a
# step-function (if-then clause; will be explored later).
#
# It would be easier if we would have a way to learn these different inputs
# features, instead of hand-coding all of the ones we can think of. That's
# exactly what we can achieve with stacking multiple layers of a neural network,
# by the fact that the output of one layer becomes the input to the next.
# Stacking layers is therefore similar to combining functions. But that's not
# enough - combining two linear functions will produce a linear function. So
# we also need to guarantee that each of the internal activation functions we
# combine is nonlinear - like in the case of sigmoid, tanh or relu.
#
# In fact, it was proven that any function can be approximated using some
# combinations of the aforementioned activation functions, regardless of how
# weird or complicated it is. See the link[1] below for more intuition, but
# generally this is how can think about it:
#
# (1) say we have a function that generates some curve (or even a step function)
# along its x-axis. The curve, or height, of the function can be controlled via
# the learned weights and bias. All of the non-linear activation functions can
# achieve that.
#
# (2) We use different such functions along the x-axis, with a smooth transition
# between them. Such that a sub-segment of the x-axis (for example 0 <= x < 10)
# can go through function A, a different sub-segment (10 <= x < 20) through
# function B, etc.
#
# (3) If we choose enough such functions for infinitely small sub-segments, we
# can control the error-rate of our approximation.
#
# [1] See more: http://neuralnetworksanddeeplearning.com/chap4.html
#
# So, within a single hidden layer, we want to assign some params for
# controlling the first function, and then another set for controlling the
# second function, etc. And in the final layer we have a single set of weights
# to control the contributions of each such function on the overall result. In
# linear regression, the last output layer might not have an activation function
# in order to allow the output to be an unbound combination of the functions
# in the hidden layers.
import numpy as np

ALPHA = .5
EPSILON = 0.0001 # it's back!

# our example data includes only a single 2D instance. We're using fixed values,
# and only a single epoch in order to avoid all randomness and measure the
# exact updated weights.
X = np.array([.05, .10])
T = np.array([.01, .99])

# our weights are also fixed at a certain starting point. But this time, we have
# two sets of weights - one per layer. Each layer is currently configured as
# MxN where N is 3D (input 2D + bias) and M is 2D.
Wxh = np.array([[.15, .20, .35], [.25, .30, .35]]) # input to hidden
Why = np.array([[.40, .45, .60], [.50, .55, .60]]) # hidden to output

# In order to avoid code repetition for each weights matrix, we'll use a TanH
# class to implement the prediction and derivatives:
class TanH(object):
    def __init__(self, w):
        self.W = w

    # forward pass - compute the predicted output for the given input
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = np.tanh(z) # tanh non-linear activation
        return y

# We will also use a separate, final error layer. This will make our network
# structure more generic, by embedding the error calculation into a layer of its
# own.
class SquaredError(object):
    def forward(self, x):
        # squared error layer doesn't modify the output. We will see other error
        # functions that do modify the output (Softmax, for example).
        self._y = x
        return x

    # error layers receive the target vector and returns the error
    def error(self, t):
        y = self._y
        return (y - t) ** 2 / 2

# now lets create our two layers with the weights we've created before:
l1 = TanH(Wxh)
l2 = TanH(Why)
l3 = SquaredError()
layers = [l1, l2, l3]

# Now's the tricky bit - how do we learn the weights? Before, we've used
# calculus to compute the derivative of the error function w.r.t each weight. We
# can do it again here: dw = np.array([d * x for d in y - t])   But this will of
# course only apply to the last layer, because we're disregarding the internal
# weights and hidden state. Instead, we want to learn how every weight, in both
# layers, affects the final error. The best known algorithm to calculate all of
# these weights at once is the Back Propagation algorithm that will be discussed
# later.
#
# For now - we'll use a different approach: pertubations. This is similar to
# what we did initially with numeric gradient descent. We will try making a tiny
# change in each weight, and re-compute the total error produced. The normalized
# difference in error will be our approximation of the derivative. While this
# approach is insanely ineffective for any production code, it will still be
# useful in the future for checking that our back propoagation code was
# implemented correctly (a process called Gradient Checking). Yes, this code is
# very messy - but that's the nature of these gradient checks functions.
#
# NOTE this function can be used for gradients checks whenever the provided
# layers adhere to the conventional API for forward(), error() and backward()
def gradients(layers, x, t, epsilon = 0.0001):
    # compute the error of the given x, t
    last = layers[len(layers) - 1]
    y = reduce(lambda x, l: l.forward(x), layers, x)
    e = last.error(t)

    # now, shift all of the weights in all of the layers, and for each such
    # weight recompute the error to determine how that weight affects it
    dws = [] # output derivatives per layer
    for l in layers:
        # some layers may just do a calculation without any weights (like the
        # final error layers).
        w = getattr(l, "W", np.array([]))
        dw = np.zeros(w.shape) # output derivatives for the layer
        for i in range(len(w)):
            for j in range(len(w[i])):
                w[i][j] += epsilon # shift the weight by a tiny epsilon amount
                yij = reduce(lambda x, l: l.forward(x), layers, x)
                eij = last.error(t) # re-run the network for the new error
                dw[i][j] = sum(e - eij) / epsilon # normalize the difference
                w[i][j] -= epsilon # rever our change

        dws.append(dw)

    return dws

# predict the output of our single-instance training set:
last = layers[len(layers) - 1] # last error function
y = reduce(lambda x, l: l.forward(x), layers, X)
e = last.error(T)
print "ERROR Correct? = %s" % np.allclose(e, [0.253637, 0.027490])
print e

# we can now use our gradient function to numerically compute the derivatives
# of all of the weights in the network
dws = gradients(layers, X, T)
for l, dw in zip(layers, dws):
    w = getattr(l, "W", 0.)
    w += ALPHA * dw
    l.W = w

# print the updated weights
print
print "l1.W Correct? = %s" % np.allclose(l1.W, [
    [0.148130, 0.196260, 0.312602],
    [0.247892, 0.295784, 0.307847]
])
print l1.W

print
print "l2.W Correct? = %s" % np.allclose(l2.W, [
    [0.338580, 0.386369, 0.429644],
    [0.518139, 0.568792, 0.650312]
])
print l2.W
