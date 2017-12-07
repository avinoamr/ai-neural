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
# the exponent 2 - but still unlikely to cover everything.
# This still holds for classification problems where we may want something
# that's more like a step-function (if-then clause).
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
# weird or complicated it is. See the link below for more intuition, but
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
# See more: http://neuralnetworksanddeeplearning.com/chap4.html
#
# So, within a single hidden layer, we want to assign some params for
# controlling the first function, and then another set for controlling the
# second function, etc. And in the final layer we have a single set of weights
# to control the contributions of each such function on the overall result. In
# linear regression, the last output layer might not have an activation function
# in order to allow the output to be an unbound combination of the functions
# in the hidden layers.
import numpy as np

STEP = .5
EPSILON = 0.0001 # it's back!

# our example data includes only a single 2D instance. We're using fixed values,
# and only a single epoch in order to avoid all randomness and measure the
# expected updated weights.
X = np.array([.05, .10])
T = np.array([.01, .99])

# our weights are also fixed at a certain starting point. But this time, we have
# two sets of weights - one per layer. Each layer is currently configured as
# MxN where N is 3D (input 2D + bias) and M is 2D.
Wxh = np.array([[.15, .20, .35], [.25, .30, .35]]) # input to hidden
Why = np.array([[.40, .45, .60], [.50, .55, .60]]) # hidden to output

# In order to avoid code repetition for each weights matrix, we'll use a Layer
# class to implement the prediction and derivatives:
class Layer(object):
    W = None

    def __init__(self, w):
        self.W = w

    # forward pass - compute the predicted output for the given input
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z)) # sigmoid non-linear activation
        return y

# now lets create our two layers with the weights we've created before:
l1 = Layer(Wxh)
l2 = Layer(Why)

# predict the output and loss for the given input and target
def predict(x, t):
    h = l1.forward(X)
    y = l2.forward(h) # output from first layer is fed as input to the second

    # now compute our error, same as before
    e = (y - t) ** 2 /2
    return y, e

# predict the output of our single-instance training set:
_, e = predict(X, T) # = (0.421017, 0.000106)
print "LOSS %s" % sum(e) # = 0.421124

# Now's the tricky bit - how do we learn the weights? Before, we've used
# calculus to compute the derivative of the loss function w.r.t each weight. We
# can do it again here: dw = np.array([d * x for d in y - t])   But this will of
# course only apply to the last layer, because we're disregarding the internal
# weights and hidden state. Instead, we want to learn how every weight, in both
# layers, affects the final error. The best known algorithm to calculate all of
# these weights at once is the Back Propagation algorithm that will be discussed
# later.
#
# For now - we'll use a different approach: pertubations. This is similar to
# what we did initially with numeric gradient descent. We will try making a tiny
# change in each weight, and re-compute the total loss produced. The normalized
# difference in loss will be our approximation of the derivative. While this
# approach is insanely ineffective for any production code, it will still be
# useful in the future for checking that our back propoagation code was
# implemented correctly (a process called Gradient Checking)
Ws = [l1.W, l2.W] # all of the weights in the network
dWs = [] # derivatives of all weights in both layers.
for w in Ws: # iterate over all weight matrices in the network
    dW = np.zeros(w.shape)

    # for every weight - re-run the entire network after applying a tiny change
    # to that weight in order to measure how it affects the total loss.
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j] += EPSILON # add a tiny epsilon amount to the weight
            _, e_ = predict(X, T) # re-run our network to predict the new error
            dW[i][j] = sum(e - e_) / EPSILON
            w[i][j] -= EPSILON # revert our change.

    dWs.append(dW)

# Now we're ready for our update - same as before:
for W, dW in zip(Ws, dWs):
    W += STEP * dW

# print the updated weights
print "l1.W ="
print l1.W # = (0.149780, 0.199561, 0.345614), (0.249751, 0.299502, 0.345022)
print "l2.W ="
print l2.W # = (0.358916, 0.408666, 0.530751), (0.511300, 0.561369, 0.619047)
