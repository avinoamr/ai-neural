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
    _last = (None, None) # input, output

    def __init__(self, w):
        self.W = w

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        y = np.dot(self.W, x) # derivate: x

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    #
    # This function is the heart of the back-propagation algorithm. It does two
    # things: (a) it computes the derivatives of its local weights and updates,
    # and (b) it returns the derivatives of the output from the previous  layer.
    # This is required in order to allow the previous layer to compute its own
    # derivatives in (a).
    #
    # The implementation details - while important - are secondary to the fact
    # that the role of this backward() function is to compute the derivatives of
    # whatever we did in the forward() function. This symmetry gurantees that we
    # can chain multiple layers together, even if the implementation of each one
    # is completely different, and still achieve the correct results.
    def backward(self, dy):
        x, y = self._last

        # compute the derivatives - differentiate the reverse of the forward
        # pass. So for every mathematical operation in the forward pass, we need
        # the respective derivative in this backward pass. This is exactly like
        # what we've done before except that now we don't need to also compute
        # the derivative of the total loss of each of our output neurons - it's
        # given as the input to this layer from the next layer.
        dw = np.array([d * x for d in dy])

        # before we update the weights, we'll compute our return value, which
        # will become the input to the previous layer - or how the total error
        # changes w.r.t changes to the output of that previous layer. Since the
        # previous layer is the input to this current layer, this is equivalent
        # to the derivative w.r.t our input - so when we change the input - how
        # does the total error changes?
        #
        # This amount is equal to (a) how does our input affect our output,
        # or dy/dx multiplied by (b) how does our output affects the total error
        # which was given to us as input. We multiply these terms via the chain
        # rule to chain these effects.
        #
        # We start with the derivative of the output as a function of the input.
        # So when we change each input, how will the output be affected? Exactly
        # by the weight amounts of course, because: y = w1x1 + w2x2 + ...
        # So if we increase x1 by 1, y will increase by exacly w1.
        dy_dx = self.W # MxN matrix - rows are y, columns are x.

        # Now all that's left to know is how that output affects the error. We
        # already know that! It's the input we received to this function (dy).
        # All that's left to do is chain these two terms together. NOTE that
        # this entire loop is can be re-written as: dx = np.dot(dy, dy_dx)
        dx = np.zeros(len(x)) # our result - derivative of total loss w.r.t x
        for i in xrange(len(y)):
            # this y[i] affects the total loss by dy[i]. We know that. Now we
            # need to compute how each input affects this y. This was computed
            # before as dy_dx[i] - a 1xN matrix showing us the effect of each
            # input on this y. So we just need to chain these terms together, to
            # get the effect of each input on the total error. NOTE that as each
            # output i is affected by all inputs, we need to sum up these
            # effects per input to get the derivative of the total loss for each
            # individual input
            dx += dy[i] * dy_dx[i]

        # update
        self.W -= ALPHA * dw

        # we can remove the computed derivative of the final bias input because
        # it was solely computed by this layer and it's not part of the output
        # from the previous layer.
        return np.delete(dx, -1)

# build the two layers in the network
l1 = Layer(Wxh)
l2 = Layer(Why)

# forward-pass
h = l1.forward(X)
y = l2.forward(h) # output from first layer is fed as input to the second

# now compute our error, same as before
e = (y - T) ** 2 /2
print "LOSS %s" % sum(e) # = 0.421124

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

# print the updated weights. NOTE that we can check our math by comparing the
# weights from the back-prop algorithm with the results we received before with
# pertubations. This is known as gradient checking.
print "l1.W ="
print l1.W # = (0.140640, 0.181281, 0.162808), (0.239475, 0.278951, 0.139499)
print "l2.W ="
print l2.W # = (0.226794, 0.269912, 0.141162), (0.497235, 0.547125, 0.592662)
