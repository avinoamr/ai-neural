# Our current system is able to approximate functions with hundreds or thousands
# of inputs - but at a massive performance cost. The computation of the slope,
# or derivative of the error function, w.r.t every single input param. Thus, for
# each iteration, we perform O(N) probes into the f function, where N is the
# number of parameters in the input vector[1]. This method is known as
# numerical gradient descent. We can do better. We can compute the derivatives
# without accessing the error function at all! This is based on the fact that we
# are currently only considering linear function with a very strictly defined
# expression, thus we can derive that expression.
#
# Remember that our error function is the distance between the prediction and
# the correct target value squared:     error = (y - t)^2
#
# We now want to find the derivatives of this function w.r.t every i in w
# without probing the original function f for every infinitisimal change.
# Deriving this error function is done by the chain rule, but here I prefer to
# work out the intuition.
#
# First, lets determine how this function changes w.r.t changes in y. So
# regardless of how we compute y - we want to see how its final value will
# affect the overall error. According to the power rule:
#
#   error'(y, t) w.r.t y = 2(y - t)
#
# NOTE that in most texts, the original error function is divided by 2 (which
# keeps it linear, and has no effect on the steepest descent) for convinience
# while deriving, thus resulting in:
#
#   error(y, t)  = (y - t)^2 / 2     => error'(y, t) = y - t
#
# This is actual version we'll use going forward.
#
# Now, we want to see how the y function changes w.r.t a change in w[i]. i.e. we
# want to see how a tiny change in the weights affects the prediction. We know
# that the prediction has a linear form: w1x1 + w2x2 + ... + wNxN + C
# so for any specific w[i] - say w1 - an increase of 1 unit will increase y by
# corresponding input:
#
#   y(x, w) = x*w => y'(x, w) w.r.t w = x
#
# That's correct for every w[i] respectively. Thus:
#
#   y'(x, w) w.r.t w[i] = x[i]
#
# So now we know that increasing w[i] will increase y by x[i], which will
# increase error by (y - t). Thus the entire derivative of the error function is
# a multiplication of the two factors:
#
#   error'(x) w.r.t w[i] = (y - t) * x[i]
#
# The same result can be achieve more directly via the chain rule, but I wanted
# the underlying effects to be clearer for future me.
#
# The implication is that we only need to compute the derivative of our error
# function directly for all parameters in the input, and then multiply
# individually by the input correspnding to each weight, using simple algebra,
# instead of repetitive probing.
#
# [1] Generally, the purpose of these excercises are not to get the best
# performance, but to build intuition, however this iterative process also
# generates a lot of code that will make it more difficult to proceed without
# eliminating some of the overhead.
import numpy as np
np.random.seed(1)

# Notice that E (epsilon) is now removed because we're not probing anymore!
N = 3
ALPHA = 0.01
ITERATIONS = 1300

# same as before, weights to be learned are: [10, 8, -2, .5].
def f(X):
    return 10 + 8 * X[0] - 2 * X[1] + X[2] / 2

w = np.random.random(1 + N)
for j in xrange(ITERATIONS):

    # just like before - we're assuming that these inputs were given
    x = np.random.rand(N) * 2 - 1 # cheat.
    t = f(x)

    # make our prediction based on our current weights
    x = np.insert(x, 0, 1.)
    y = sum(w * x)

    # compute the error
    e = (y - t) ** 2 / 2
    print "%d: f(%s) = %f == %f (ERROR: %f)" % (j, x, y, t, e)

    # now comes the big change: we compute the derivative of the error function
    # with respect to the output y. This gives us a sense of how the error will
    # change when we change the output, element-wise.
    #
    # Recall: e = (y - t) ** 2 / 2    => de/dy = (y - t)
    dy = y - t

    # but what we're really after is the derivative of the error function,
    # with respect to w, in order to know in which direction to update the
    # weights. This is built of two terms (1) how the error is affected by the
    # output - already computed as dy; (2) how the aforementioned output is
    # affected by the weights. Finally, we'll need to multiply these two terms
    # to retrieve the combined effect of w on the error function following the
    # chain rule.
    #
    # See the top comment for the full intuition and math. As an additional
    # incentive - the fact that it's a simple scalar-vector multiplication,
    # without probes into f, GPUs will be able to significantly further improve
    # the performance of this operation!
    #
    # NOTE that we can combine these two terms in a single line of code, but
    # much later on, when we'll look into back-propagation, we'll see why we're
    # going to need these two terms separated, so that's the convention we'll
    # stick with.
    #
    # Recall: y = w*x   => dy/dw = x
    dw = dy * x

    # now update the weights and bias, same as before.
    w += ALPHA * dw * -1

print "W = %s" % w
