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

def main():
    # This is exactly the same data - except that now it has 1 outlier at the third
    # value in which the output doesn't fit the formula: 1 + 10x:
    X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ]
    T = [101, 201, 331, 401, 501, 601, 701, 801, 901, 1001] # notice the 3rd value
    N = 1

    # If we run out previous code as is, these outliers will have a significant
    # influence over our learning, because they're given a full STEP in their
    # graident direction, resulting in an error that than has to be fixed.
    # Instead, here we'll use mini-batches to average out this "noise", by
    # calculating the average derivative over a small mini-batch of the entire
    # data, and then applying the STEP-sized learning for this average. This
    # might seem like it slows learning a little bit, because we're only
    # applying the STEP update twice for each epoch, but in reality it's better
    # because there's less error-correction.
    BATCHSIZE = len(X) / 2 # = 1 <- try this for non-batch for comparison

    lr = LinearRegression(N, BATCHSIZE)
    data = zip(X, T)
    for i in xrange(200):
        l = lr.learn(data)
        print "%s LOSS = %f" % (i, l)

    print
    print "W = %s" % lr.w


# LinearRegression algorithm
class LinearRegression(object):
    # hyper-parameters
    STEP = 0.01
    batchsize = 1
    n = None

    # parameters
    w = None

    # batchsize = 1 -> online learning
    # batchsize = len(data) -> fully-batched learning
    def __init__(self, n, batchsize):
        self.batchsize = batchsize
        self.w = np.zeros(1 + n) # 1-extra weight for the bias at index 0

    # loss computes the loss (error; difference) between the target and the
    # output predicted by our current weights along with the derivatives of this
    # loss w.r.t every weight
    def loss(self, inp, target):
        w = self.w

        # make our prediction based on our current weights
        inp = np.insert(inp, 0, 1.)
        out = sum(inp * w)

        # compute the loss & derivatives
        l = (out - target) ** 2 / 2
        d = (out - target) * inp
        return l, d

    # learn uses the loss function to actually perform the learning by updating
    # the weights once for the entire mini-batch. Data must be an array of
    # tuples in the form (input, target). Returns the overall average loss for
    # the entire provided data-set
    def learn(self, data):
        l_avg = 0 # average of the loss for the whole data

        # Break down the data into several mini-batches, each of size BATCHSIZE.
        # Compute the average loss for this entire mini-batch, and then update
        # the weights only once for this entire batch. The result is an update
        # that averages out the influence of the noise.
        for i in range(0, len(data), self.batchsize):
            batch = data[i:i + self.batchsize]
            d_avg = 0 # average of the derivatives for the whole mini-batch

            for inp, target in batch:
                l, d = self.loss(inp, target)
                d_avg += d / len(batch)
                l_avg += l / len(data)

            # single update for the average gradient (derivatives) over the
            # entire minibatch
            self.w += self.STEP * d_avg * -1

        return l_avg


if __name__ == "__main__":
    main()
