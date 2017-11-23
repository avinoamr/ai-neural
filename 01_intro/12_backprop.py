# Pertubations are a good way to approximate the derivative of the error
# function without any knowledge about the specific implementation of the code.
# But it's super slow and cumbersome - as we need to change every weight and
# then re-run the entire network. This means that for every instance in the data
# we will need to run our network porportional to the number of weights in the
# system.
#
# But since we do know the exact implementation details of the functions we're
# deriving, we can do much better by deriving analytically. We can achieve that
# with one of the major breakthroughs in Machine Learning: Back Propogation.
#
# Before continuing, it's advised to read this great step-by-step tutorial about
# back propagation that might provide some more insight and intuition:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
import numpy as np

ALPHA = 0.5
EPSILON = 0.01

# we're going to use the same fixed input and weights as before:
X   = np.array([.05, .10])
T   = np.array([.01, .99])

Wxh = np.array([[.15, .20, .35], [.25, .30, .35]])
Why = np.array([[.40, .45, .60], [.50, .55, .60]])

# Layer represents a single
class Layer(object):
    W = None

    # we will need to store the information of the last input and output in
    # order to derive analytically.
    _last_out = None
    _last_inp = None

    def __init__(self, w):
        self.W = w

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        net = np.dot(self.W, x) # derivate: x
        y = 1 / (1 + np.exp(-net)) # sigmoid activation; derivate: y(1 -y)

        self._last_inp, self._last_out = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    #
    # This function is the heart of the back-propagation algorithm.
    def backward(self, dE_dy):
        x, y = self._last_inp, self._last_out

        # compute the derivatives - differentiate the reverse of the forward
        # pass. So for every mathematical operation in the forward pass, we need
        # the respective derivative in this backward pass. This is exactly like
        # what we've done before.
        dy_dnet = y * (1 - y)
        dE_dnet = dE_dy * dy_dnet # 2
        dnet_dw = x # 3

        dE_dw = np.array([d * dnet_dw for d in dE_dnet])

        # before we update the weights, we'll compute our return value. That
        # return value will become the input to the previous layer - or how the
        # total error changes w.r.t changes to the output of that previous
        # layer. Since the previous layer is the input to this current layer,
        # this is equivalent to the derivative w.r.t our input - so when we
        # change the input - how does the total error changes?
        dnet_dx = self.W

        dE1_dx = dE_dnet[0] * dnet_dx[0]
        dE2_dx = dE_dnet[1] * dnet_dx[1]
        dE_dx = dE1_dx + dE2_dx
        ret = np.delete(dE_dx, -1) # remove the bias derivative

        # update
        self.W -= ALPHA * dE_dw

        return ret

# build the two layers in the network
l1 = Layer(Wxh)
l2 = Layer(Why)

# forward-pass
h = l1.forward(X)
y = l2.forward(h) # output from first layer is fed as input to the second

# now compute our error, same as before
e = (y - T) ** 2 /2
print "LOSS %s" % sum(e) # = 0.298371109

# backward-pass
#
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
d = y - T # start loss derivative at the top error layer. Same as before.
d = l2.backward(d)
_ = l1.backward(d)

# print the updated weights
print "l1.W ="
print l1.W # = (0.149780, 0.199561, 0.345614), (0.249751, 0.299502, 0.345022)
print "l2.W ="
print l2.W # = (0.358916, 0.408666, 0.530751), (0.511301, 0.561370, 0.619047)
