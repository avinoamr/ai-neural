# Remember that cheat we had before where we just generated random inputs? Well,
# in the real world you can't really do that. Learning is based on a finite set
# of observations - or instances - including both the inputs and outputs. We're
# using this small training set to try to find a model that best describes the
# relationship between the input and output. In this example, we will not
# generate inputs, and will have no access to the function we're trying to model
# so that we can't experiment with it. All we have is a list of observations:
import numpy as np

# There's no f function anymore. We only have 10 observations encoded as a list
# of inputs (X) and a list of targets (T). For a human it's super easy to pick
# up on the pattern here and understand that the underlying function is
# likely: f(x) 100 + 100x.     But by running this code you'll see that for our
# linear regression it's very difficult as there aren't enough observations
# here. Perhaps that's an early demonstration of how our current algorithms are
# so unlike the human brain.
X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ] # 1-dimensional input
T = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001] # output
N = 1 # 1-dimension in the data above.

# constants. ITERATIONS isn't here anymore because we can't just iterate
# arbitrarily as we have a finite set of inputs.
STEP = 0.01
def loss(actual, target):
    return (actual - target) ** 2

w = np.random.rand(N) - .5
b = np.random.rand() - .5
for x, t in zip(X, T):

    # same as before, we'll compute our prediction
    y = sum(x * w, b)

    # compute the loss
    l = loss(y, t)
    print "f(%s) = %f (y: %f, loss: %f)" % (x, t, y, l)

    # derivatives
    dw = 2 * (y - t) * x
    db = 2 * (y - t)

    # debug the derivatives ; read below.
    # print "dw = %f ; db = %f" % (dw, db)

    w += STEP * dw * -1
    b += STEP * db * -1

# you'll notice that after the full iteration, our algorithm was unable to fully
# eliminate the loss - although the function is perfectly linear and simple.
# If we uncomment the debugging line above, we start to understand why: both
# the weight and bias are being updated by the exact same step size. The only
# difference is the derivatives. But as you'll notice, the derivatives start
# out very similar, and only very slowly they're starting to diverge such that
# the bias (supposed to be 1) is exhibiting smaller changes than the weight
# (supposed to be 100).
#
# This will happen in all cases of gradient descent where there the different
# parameters are skewed such that they're not on a somewhat similar scale. We
# wouldn't have had this problem is the target bias or weight were all on a
# scale of -1 to 1. For example. One fix is to first scale the inputs into a
# similar scale (Feature Scaling).
print
print "W = %s ; b = %s" % (w, b)
