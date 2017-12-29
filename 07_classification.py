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
