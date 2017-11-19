# The example before managed to find the perfect weights in just a few hundred
# epochs because the inputs were perfectly correlated to the outputs. In real
# life, this is rarely the case. Usually there will be some "noise" in the data,
# some outliers that doesn't fit perfectly. This is because it can be assumed
# that the inputs doesn't capture every possible feature in the problem domain.
# For example, if our inputs represent the number of items purchased, and the
# target output represents the cost of these items, it's possible to have
# outliers where the data doesn't fit perfectly, due to special promotions,
# sales, surges, negotiations, etc. that are not captured in the data inputs.
#
# NOTE that it's impossible to achieve an average loss of 0 anymore, because
# of this noise, but the loss can still be minimized.
import numpy as np

# This is exactly the same data - except that now it has 1 outlier at the third
# value in which the output doesn't fit the formula: 1 + 10x:
X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ]
T = [101, 201, 331, 401, 501, 601, 701, 801, 901, 1001] # notice the 3rd value
N = 1

EPOCHS = 2000
STEP = 0.01

# If we run out previous code as is, these outliers will have a significant
# influence over our learning, because they're given a full STEP in their
# graident direction, resulting in an error that than has to be fixed. Instead,
# here we'll use mini-batches to average out this "noise", by calculating the
# average derivative over a small mini-batch of the entire data, and then
# applying the STEP-sized learning for this average. This might seem like it
# slows learning a little bit, because we're only applying the STEP update twice
# for each epoch, but in reality it's better because there's less error-
# correction.
BATCHSIZE = len(X) / 2 # = 1 <- try this for non-batch for comparison

w = np.zeros(1 + N)
data = zip(X, T)

# batch learn a set of data. Similar code as before, except that now we're only
# updating the weights once for the average derivatives across the batched data
# in order to cancel out the noise. This logic is moved to its own function in
# order to keep the code a bit flatter and more readable. Returns the total sum
# of all of the losses encountered.
def learn(w, data):
    l = 0 # sum of losses, to be averaged later.
    dw = 0
    for x, t in data:

        # same as before, we'll compute our prediction
        x = np.insert(x, 0, 1.)
        y = sum(x * w)

        # sum the loss & derivatives
        l += (y - t) ** 2 / 2
        dw += (y - t) * x

    # update the weights only once for this entire batch. The result is an
    # update that averages out the influence of the noise.
    dw /= len(data)
    w += STEP * dw * -1
    return l

for i in xrange(EPOCHS):
    l = 0

    # Break down the data into several mini-batches, each of size BATCHSIZE.
    # Each batch is handdled as if it was the entire input, except that we
    # update the weights once for the average derivatives.
    for j in xrange(0, len(data), BATCHSIZE):
        l += learn(w, data[j:j+BATCHSIZE])

    l /= len(data) # average the loss
    print "%s LOSS = %f" % (i, l)

print
print "W = %s" % w
