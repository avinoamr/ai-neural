# Our current system is able to approximate functions with hundreds or thousands
# of inputs - but at a massive performance cost. The computation of the slope,
# or derivative of the loss function, w.r.t every single input param. Thus, for
# each iteration, we perform O(N) probes into the f function, where N is the
# number of parameters in the input vector.[1]. This method is known as
# numerical gradient descent. Suprisingly (to me at least) we can do better. We
# can compute the derivatives without accessing the loss function at all!
#
# Remember that our loss function is the distance between our prediction and the
# correct target value squared:
#
#       loss = (y - t)^2
#
# where y is our prediction and t is the correct target value. We now want to
# find the derivatives of this function w.r.t every i in w without probing the
# original function (t = f(x)) for every infinitisimal change. Deriving this
# loss function is done by the chain rule, but here I prefer to work out the
# intuition.
#
# First, lets determine how this function changes w.r.t changes in y. So
# regardless of how we compute y - we want to see how its final value will
# affect the overall loss. According to the power rule:
#
#   loss'(y, t) w.r.t y = 2(y - t)
#
# Now, we want to see how the y function changes w.r.t a change in w[i]. i.e. we
# want to see how a tiny change in the weights affects our prediction. We know
# that our prediction has a linear form: w1x1 + w2x2 + ... + wNxN + C
# so for any specific w[i] - say w1 - an increase of 1 unit will increase y by
# corresponding input:
#
#   y(x, w) = x*w =>
#   y'(x, w) w.r.t w = x
#
# That's correct for every w[i] respectively. Thus:
#
#   y'(x, w) w.r.t w[i] = x[i]
#
# So now we know that increasing w[i] will increase y by x[i], which will
# increase loss by 2(y -t). Thus the entire derivative of the loss function is
# a multiplication of the two factors:
#
#   loss'(x) w.r.t w[i] = 2(y - t) * x[i]
#
# The same result can be achieve more directly via the chain rule, but I wanted
# the underlying effects to be clearer for future me. NOTE that in most texts,
# the original function is divided by 2 (which keeps it linear, and has no
# effect on the steepest descent) for convinience while deriving thus resulting
# in:
#
#   loss()  = (y - t)^2 / 2
#   loss'() = (y - t) * x[i]
#
# This is actual version we'll use going forward.
#
# The implication is that we only need to compute the derivative of our loss
# function directly for all parameters in the input, and then multiply
# individually by the input correspnding to each weight, using simple algebra,
# instead of repetitive probing.
#
# [1] Generally, the purpose of these excercises are not to get the best
# performance, but to build intuition, however this iterative process also
# generates a lot of code that will make it more difficult to proceed without
# eliminating some of the overhead.
import numpy as np
np.random.seed(1) # static seed to make results reproducible

def main():
    # same as before, weights to be learned are: [1, 2, 3, ...], and bias 10
    f = lambda x: 10 + 8 * x[0] - 2 * x[1] + x[2] / 2
    N = 3

    lr = LinearRegression(N)
    for i in xrange(1000):
        # just like before - we're assuming that these inputs were given
        inp = np.random.rand(N) * 2 - 1 # cheat.
        target = f(inp)
        loss = lr.minimize(inp, target)
        print "#%d f(%s) = %f (loss: %f)" % (i, inp, target, loss)

    # print the weights
    print "W = %s" % lr.w

# LinearRegression algorithm
class LinearRegression(object):
    # hyper-parameters
    # Notice that E (epsilon) is now removed because we're not probing anymore!
    STEP = 0.01

    # parameters
    w = None # the weights we want to learn for each input

    def __init__(self, n):
        self.n = n

        # initial weights. before we've learned the input, now that the input is
        # given and immutable we learn the weights by which to multiply this
        # input. In fact, these weights can represent any linear function, so
        # all that's left to do is find these weights and we have our function
        # approximation!
        self.w = np.zeros(1 + n) # 1-extra weight for the bias at index 0

    # attempt to minimize the loss function for a given observation of N inputs
    # and the target result for this input. We don't have access to the function
    # being fit, so we must only rely on the inp and target
    def minimize(self, inp, target):
        w = self.w

        # make our prediction based on our current weights
        inp = np.insert(inp, 0, 1.)
        out = sum(inp * w)

        # same loss function as before, except that now it can't rely on a
        # constant target value for all inputs, but instead it receives the
        # value as input. It's no longer a separate function because we don't
        # need to re-use this 1-line.
        l = (out - target) ** 2 / 2

        # now is the big change: we compute the derivative of the loss function
        # w.r.t each w[i] by multiplying the input, element-wise, with the
        # derivative of the loss function w.r.t the prediction of weights:
        #
        #   loss'(X) w.r.t w[i] = 2(y - t) * x[i]
        #
        # See the top comment for the full intuition and math. As an additional
        # incentive - the fact that it's a simple scalar-vector multiplication,
        # without probes into f, GPUs will be able to significantly further improve
        # the performance of this operation!
        d = (out - target) * inp # that's it! no probes in f

        # now update the weights and bias, same as before.
        self.w += self.STEP * d * -1

        return l


if __name__ == "__main__":
    main()
