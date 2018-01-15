# Everything we've seen to this point has been a problem known as regression in
# which we're trying to predict an actual numeric value for each observation of
# N input numeric values. A more common problem is that of classification -
# predicting a single binary occurance, class or label for each input. The
# example we'll explore now is attempting to predict for every passenger aboard
# the Titanic, if they survived or not. Clearly, this is not a numeric value,
# but a boolean one: True (survived) or False (didn't survive)
#
# A different way to think about classification is in terms closer to regression
# where instead of approximating an output value for each input, we're
# learning a threshold line in the function where values below these threshold
# doesn't belong to a class, and values above it do.
#
# The weights of an output unit determine the logical expression for the
# corresponding input, while the bias acts as a threshold (axon hillock) that
# must be surpassed in order for the unit to activate. So the bias basically
# describe the excitability of the unit, or how likely it is to fire. While the
# weights are the effect of the individual inputs. Mathematically:
#
#   y = w * x + b >= 1      =>   w * x >= -b
#
# That means that in order for the output of the unit to be greater than 1 we
# need w * x to be greater than the negative of the bias. Remember that in
# classification the input x is a binary 0 or 1, so we have two cases:
#
#   x = 0   => w * 0 > -b = 0 > -b
#   x = 1   => w * 1 > -b = w > -b
#
# So basically, the bias describes two properties: (a) the default activation of
# the unit, whether it should fire or not on zero input (x = 0). And (b) how big
# should the weights be to excite or inhibit that default activation for a non-
# zero input (x = 1). A positive bias (1) will fire unless there are enough
# negative weights (where the input is 1) to inhibit it, while a negative bias
# (-1) will not fire unless there are enough positive weights to excite it. With
# these two variables, we can describe any single-argument boolean function:
#
#   f       w   b
#   T       0   1       y = 0 * x + 1  =  1
#   F       0  -1       y = 0 * x - 1  = -1
#   x       1   0       y = 1 * x + 0  =  x     = 0 (when x = 0) or 1 (x=1)
#  !x      -1   1       y = -1 * x + 1 = -x + 1 = 0 (when x = 1) or 1 (x=0)
#
# So if we learn these w & b values, we can approximate any single-argument
# boolean function. But when we add arguments, we can add boolean operations
# like AND and OR. Lets start with AND: we will need the sum of a subgroup of
# the weights exceed the negative bias:
#
#   w1 + w2 > -b ; w1 < -b ; w2 < -b
#
# It's possible to have other weights, but there's a subgroup of the weights
# where each is not big enough to exceed -b by itself, but the sum of these
# weights does exceed. Both of these weights needs to be activated (by an input
# of 1) in order for the sum to be greater than -b. Thus the AND operator on the
# input:
#
#   w1 = 1 ; w2 = 1 ; b = -2
#   y = 1 * x1 + 1 * x2 >= 2
#   f = x1 AND x2
#
# Because we might have several such subgroups that satisfy this relationship,
# each subgroup can, by itself, exceed -b. Thus there's an OR operator between
# them:
#
#   w1 = 1 ; w2 = 1 ; w3 = 1 ; b = 2
#   y = 1 * x1 + 1 * x2 + 1 * x3 > 2
#   f = (x1 AND x2) OR (x2 AND x3) OR (x1 AND x3)
#
# To generalize, we can approximate any function with the structure of:
#
#   f = (x1 AND x2 AND ...) OR (x5 AND x6 AND ...)
#
# Where the OR separates all subgroups of the weights that has a sum greater
# than -b, while the AND separates the individual weights within each such
# group. NOTE that more complex, non-linear boolean functions are still
# impossible to approximate. The typical example is XOR, but more applicable is
# conditinals, like:
#
#   f = if x1 AND x2:   x5
#       if x3 AND x4:   x6
#
# This is common in many real-life examples as we'll see later.
import numpy as np
np.random.seed(1)

EPOCHS = 300
ALPHA = 0.01

# Our 1-dimensional input is the sex of the passenger: m (male) or f (female)
# Our output is a number, either 1 (survived) or 0 (didn't survive)
X = ["f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m"]
T = [ 1,   0,   1,   1,   1,   0,   0,   0,   1,   0,   1,   1,   1,   0 ]

# The main issue to take care of is encoding: how do we transform these textual
# categories into numeric inputs that we can estimate. One naive approach might
# be to use a single input feature, say a value of 0 represents a male, and 1
# represents a female. That wouldn't work, because any kind of weight we'll use
# will end up increasing for females. Thus we have no way to find different
# weights for the different categories. This is not necessarily correct for
# ordinal values like age or fare cost, but it's still common to learn these
# weights independently by grouping multiple numeric values into a discrete
# set of categories ("young", "old" for age; "cheap", "expansive" for fare cost)
# The same limitation obviously applied if we use more values with binary
# encoding.
#
# The best known approach currently is one-hot (or one-of-k) in which each value
# is assigned a completely different input. If we have k values, we'll use
# k input neurons (one for male and the other for female) in which only one
# neuron can be lit (value of 1) for any given training case. If we have
# multiple categories we can concatenate multiple such one-of-k's as needed as
# that maintains the fact that each value is assign a separate input and weight.
N = len(set(X)) # 1 per unique value

# encode the input data strings into a list of one-of-k's. We want to return a
# list of numbers, where all are set zeros, but only one is to set to one. That
# should be applied to each feature - one for value. More features would require
# a concatenation of such one-of-k's
def one_of_k(v):
    x = np.zeros(N)
    idx = ["m", "f"].index(v)
    x[idx] = 1.
    return x

X = np.array([one_of_k(x) for x in X])
w = np.random.randn(N + 1) * 0.01 # start with small random weights
data = zip(X, T)
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0

    # we will now also compute the accuracy as a count of how many instances in
    # the data were predicted correctly. This is a more quantitive way of
    # representing the correctness of the prediction as opposed to an arbitrary
    # error function
    accuracy = 0

    # mini-batches
    for x, t in data:

        # predict
        x = np.append(x, 1.) # add the fixed bias.
        y = sum(w * x)

        # error & derivatives
        e += (y - t) ** 2 / 2
        dy = (y - t)
        dw = dy * x

        # update
        w += ALPHA * -dw # mini-batch update

        # did we predict correctly? We need to transform the output number
        # into a boolean prediction: whether the label should be turned on
        # or off. For this example, we'll simply see if the prediction is
        # closer to 0 or 1, by first clipping to the [0, 1] range in order
        # to trim values outside of this range, and then rounding.
        accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

    e /= len(data)
    print "%s: ERROR = %f ; ACCURACY = %d of %d" % (i, e, accuracy, len(data))

print
print "W = %s" % w
