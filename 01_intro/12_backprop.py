# Everything we've achieved until now can arguably be done with simple
# statistics and Perceptrons because it only applies to linear relationships
# between the input and output classes. But in order to generalize to more
# complex non-linear examples, we'll need to take advantage of one of the major
# breakthroughs in Machine Learning: Back Propogation.
#
# We'll now take a small detour from our previous code, to implement the
# backpropagation tutorial:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
import numpy as np

ALPHA = 0.5
EPSILON = 0.01

# We have 2 input neurons (+ bias = 3 total), 2 hidden neurons
# (+ bias = 3 total) and 2 output neurons. So we need, one weights matrix per
# layer (input to hidden, hidden to output), both of size 2x3 (3 input, 2
# output). We'll use the preset weights from the example:
#
# input to hidden:      w1    w2  b1     w3   w4   b1
Wxh =       np.array([[.15, .20, .35], [.25, .30, .35]])

# hidden to output:     w5   w6   b2     w7   w8   b2
Why =       np.array([[.40, .45, .60], [.50, .55, .60]])

# single instance inputs/targets from example:
x   = np.array([.05, .10])
t   = np.array([.01, .99])

# a single layer is just a wrapper around a set of weights. As you'll see below,
# we now need to separate the forward pass (generating the outputs) from the
# backward pass (generating the derivatives) - because now we want to compute
# the output before learning the weights. We also need to "remember" the input
# and output at each phase, so we need to add some more state here.
class Layer(object):
    W = None # the weights
    _last_out = None
    _last_inp = None

    def __init__(self, w):
        self.W = w

    # the forward pass is the same as before.
    def forward(self, x):
        self._last_inp = x

        x = np.append(x, 1.) # add the fixed input for bias
        net = np.dot(self.W, x) # derivate: x
        y = 1 / (1 + np.exp(-net)) # sigmoid activation; derivate: y(1 -y)

        self._last_out = y
        return y

    def backward(self, dE_dy):
        x, y = self._last_inp, self._last_out

        # compute the derivatives - differentiate the reverse of the forward
        # pass. So for every mathematical operation in the forward pass, we need
        # the respective derivative in this backward pass. This is exactly like
        # what we've done before.
        dy_dnet = y * (1 - y) # = (0.1868156, 0.17551005)
        dnet_dw = x # = (0.59326999, 0.59688438)
        dE_dnet = dE_dy * dy_dnet
        dE_dw = np.array([np.append(d * dnet_dw, 0.) for d in dE_dnet])
        print "dE_dw:", dE_dw

        # before we update the weights, we'll compute our return value. That
        # return value will become the input to the previous layer - or how the
        # total error changes w.r.t changes to the output of that previous
        # layer. Since the previous layer is the input to this current layer,
        # this is equivalent to the derivative w.r.t our input - so when we
        # change the input - how does the total error changes?
        # dy_dnet = y * (1 - y)
        dnet1_dx1 = self.W[0][0] # = 0.4
        dnet1_dx2 = self.W[0][1]
        dnet2_dx1 = self.W[1][0]
        dnet2_dx2 = self.W[1][1]

        dE1_dx1 = dE_dnet[0] * dnet1_dx1
        dE2_dx1 = dE_dnet[1] * dnet2_dx1
        dE_dx1 = dE1_dx1 + dE2_dx1

        dE1_dx2 = dE_dnet[0] * dnet1_dx2
        dE2_dx2 = dE_dnet[1] * dnet2_dx2
        dE_dx2 = dE1_dx2 + dE2_dx2

        ret = np.array([dE_dx1, dE_dx2])

        # update
        self.W -= ALPHA * dE_dw # = (0.358916480, 0.408666186, 0.6)
                                #   (0.511301270, 0.561370121, 0.6)

        return ret

# build the two layers in the network
l1 = Layer(Wxh)
l2 = Layer(Why)

# forward pass
h = l1.forward(x) # = (0.593269992, 0.596884378)
o = l2.forward(h) # = (0.751365070, 0.772928465)

# total loss. Same as before.
E = (o - t) ** 2 / 2 # = (0.274811083, 0.023560026)
Etotal = sum(E) # = 0.298371109


# Pertub. We're going to try to change every weight in the system to measure the
# derivative numerically. This is only use to test our math going forward, not
# for production.
def re_run(W, i, j):
    W[i][j] += EPSILON
    o_ = l2.forward(l1.forward(x))
    E_1 = (o_ - t) ** 2 / 2
    W[i][j] -= EPSILON
    return sum(E_1 - E) / EPSILON

print re_run(l2.W, 1, 1)

# Now we want to walk backwards and compute the derivatives of the total error
# w.r.t every weight in both layers. In other words, we want to know how every
# weight affects the loss function. There are several ways to go about that. One
# option is to use pertubations: basically attempt to change one weight at a
# time and measure what effect it had to the total loss. This is similar to our
# initial numeric gradient descent. The Back Propogation algorithm is much
# faster - it allows us to learn all of the derivatives using math all at once.
#
# The basic idea here is that each layer can compute its own derivatives
# indepentendly. Since the layer implemented it's own forward pass, it knows
# which mathematical operations were used, and can thus derive all of these
# operations. The only part that's missing is knowing how it's final output
# affected the final error on all future layers, without having any knowledge
# about the math or weights used on these layers. This is solved by propagating
# these output derivatives backwards - starting from the final loss derivative
# and allowing each layer to finally feed the derivatives of its inputs to the
# layer below.
#
# Expected weights are:
# l2.W = (0.358916480, 0.408666186, 0.6) , (0.511301270, 0.561370121, 0.6)
# l1.W = (0.149780716, 0.19956143, 0.35) , (0.249751143, 0.29950229, 0.35)
d = o - t # start loss derivative at the top layer. Same as before.
d = l2.backward(d) # = (0.036350306, 0.041370322)
_ = l1.backward(d)

# BACKWARD PASS - Hidden Layer
# ----------------------------
#
# Now comes the tricky bit. We similarly want to find the derivative of our
# error w.r.t Wxh - that is the weights in our hidden layer. While it appears
# more complicated at first, it's easy to see that as we've already solved some
# part of that problem: we already know how the error changes w.r.t to the net
# input (h) of every output neuron (o). So all that's left to compute is how
# each weight in Wxh affects that net input to o; For example, consider w1 that
# goes from input neuron 1 (x[0]) to hidden neuron 1 (h[0]), and finally to
# every output neuron (o):
#
#   dEtotal   dEtotal   dout_h1   dnet_h1
#   ------- = ------- * ------- * -------
#   dw1       dout_h1   dnet_h1   dw1
#
# Almost exactly like before! Super easy. The math is again the chain rule, and
# the intuition is a few steps again: (a) every change in w1 will create a
# change in the net total input to the hidden neuron h1 (net_h1), that change to
# net_h1 will create some change via the sigmoid function to output of neruon
# h1 (out_h1), and finally that change to the output will create some change to
# the final error due to how it interacts with the following output layer as
# already computed above. In other words, computing (a) and (b) is exactly the
# same as before. While computing (c) relies on informaiton we already computed
# in the top output layer. This is why it's called "Back Propogation" - we're
# going backwards from the derivatives at the top, and use those to compute the
# derivatives at the bottom.
#
# (a) we will start with the derivative of the error function w.r.t changes in
# the output of our hidden neuron h1. This is arguably the only tricky part here
# because while we're only looking at a single weight and a single output, that
# output will be used as input to every neuron in the output layer. So we need
# to sum up these effects. It means that as we change w1, the output of our
# hidden layer will change, and it will then be multiplied by many different
# weights to many different neurons in the next layer - each producing a
# different error. But we've already computed that top layer, so we just need to
# re-use our previous results:
#
#   dEtotal   dEo1      dEo2
#   ------- = ------- + ------- + ....
#   dout_h1   dout_h1   dout_h1
#
# We want to see how our output at neuron h1 affects the total error which is
# the sum of the individual errors:
#
#   dEtotal   dEtotal   dnet_o1
#   ------- = ------- * -------
#   dout_h1   dnet_o1   dout_h1
#
# If we know how our output from h1 affects the net input to o1, and how that
# net input to o1 affects the error - we're done with step (a). We'll only need
# to multiply by (b) and (c) and we're done!
