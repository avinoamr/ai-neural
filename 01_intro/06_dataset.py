# Remember that cheat we had before where we just generated random inputs? Well,
# in the real world you can't really do that. Learning is based on a finite set
# of observations - or instances - including both the inputs and outputs. We're
# using this small training set to try to find a model that best describes the
# relationship between the input and output. In this example, we will not
# generate inputs and will not have access to the function we're trying to model
# so we can't experiment with it. All we have is a list of observations:
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

# ITERATIONS isn't here anymore because we can't just iterate arbitrarily as we
# have a finite set of inputs. Instead of iterations, we now have a concept of
# epochs. In the past, we had an infinite data set which we consumed until the
# model was learned. Here we don't. And just 10 data points isn't enough for our
# model to learn. There are several work arounds to this like: feature-scaling,
# adaptive learning rate, mini-batching - but to keep it simple here, we can
# just "duplicate" the data multiple times. We'll do it by iterating over the
# data "EPOCHS" times.
#
# NOTE that if we're using just a few epochs (10-20), (a) the loss wouldn't
# converge to zero just yet, and (b) the bias wouldn't converge to 1.
EPOCHS = 300
ALPHA = 0.01

w = np.random.random(1 + N)
data = zip(X, T) # single data set of (x, y) tuples

# instead generating a massive list here, we're just repeating the same one.
for i in xrange(EPOCHS):
    np.random.shuffle(data)

    l = 0 # total loss in this epoch
    for x, t in data:

        # same as before, we'll compute our prediction
        x = np.insert(x, 0, 1.)
        y = sum(x * w)

        # compute the loss & derivatives
        l += (y - t) ** 2 / 2
        dy = (y - t)
        dw = dy * x

        # update
        w += ALPHA * dw * -1

    # instead of printing the loss after every observation (which can be way
    # too verbose), we'll print out the total loss for the network
    l = l / len(data) # average the loss.
    print "%s: LOSS = %f" % (i, l)

print
print "W = %s" % w
