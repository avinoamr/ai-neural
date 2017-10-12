# Our current system is able to approximate functions with hundreds or thousands
# of inputs - but at a massive performance cost. The computation of the slope,
# or derivative of the loss function, w.r.t every single input param. Thus, for
# each iteration, we perform O(N) probes into the f function, where N is the
# number of parameters in the input vector.[1] Suprisingly (to me at least) we
# can do better. We can compute the derivatives without accessing the f function
# more than once. We'll start here with the assumption that the function we're
# optimizing for is linear - i.e. can be reduced to the form:
#
#       f(x, y, ...) = ax + by + ... + C
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
# the underlying effects to be clearer for future me. Note that in most texts,
# the original function is divided by 2 (which keeps it linear, and has no
# effect on the steepest descent) for convinience while deriving thus resulting
# in: loss'(x) w.r.t w[i] = (y - t) * x[i] . We're not using this version here.
# this is also known as the delta rule.
#
# Another intuition is that if we get an error in our prediction, the the form
# of some loss numeric value, we want to update our prediction in proportion to
# that loss - thus a large mistake is likely to trigger a large change to our
# prediction - but now we scale that change by the actual input, such that a
# large input will result in a larger change than a small input.
#
# The implication is that we only need to compute (y - t) once for all
# parameters in the input, and then multiply individually by the input
# correspnding to each weight.
#
# [1] Generally, the purpose of these excercises are not to get the best
# performance, but to build intuition, however this iterative process also
# generates a lot of code that will make it more difficult to proceed without
# eliminating some of the overhead.
import numpy as np

# constants
# Notice that E (epsilon) is now removed because we're not probing anymore!
N = 2
STEP = 0.01
ITERATIONS = 1000

# same as before, weights to be learned are: [1, 2, 3, ...], and bias 10
def f(X):
    return 10 + sum([(i + 1) * x for i, x in enumerate(X)])

def loss(actual, target):
    return (actual - target) ** 2

w = np.random.rand(N) * 2 - 1
b = np.random.rand() * 2 - 1
for j in xrange(ITERATIONS): # can we stop early once we reach our target?

    # just like before - we're assuming that these inputs were given
    inp = np.random.rand(N) * 2 - 1 # cheat.
    target = f(inp)

    # make our prediction based on our current weights
    out = sum(inp * w, b)

    # compute the loss
    l = loss(out, target)
    print "#%d f(%s) = %f (loss: %f)" % (j, inp, target, l)

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
    d = 2 * (out - target) * inp # that's it! no probes in f
    db = 2 * (out - target) # bias not dependent on inp, only on the loss

    # now update the weights and bias, same as before.
    w += STEP * d * -1
    b += STEP * db * -1

    # notice that the update rule for the bias is identical to the one of the
    # weights, except that the input is fixed at value of 1. We can use that to
    # remove these extra lines of code if we add one artificial input equal to
    # 1 and its corresponding weight. Maybe next time :)

print "W = %s ; b = %s" % (w, b)