# Remember that cheat we had before where we just generated random inputs? Well,
# in the real world you can't really do that. Learning is based on a finite set
# of observations - or instances - including both the inputs and outputs. We're
# using this small training set to try to find a model that best describes the
# relationship between the input and output. We don't have a function that will
# immediately find the perfect output for every possible input, instead we need
# to discover the relationships in the data that best describe such a function.
# In this example, we will not generate inputs, and will have no access to the
# function we're trying to model so that we can't experiment with it. All we
# have is a list of observations:
import numpy as np

def main():
    # There's no f function anymore. We only have 10 observations encoded as a list
    # of inputs (X) and a list of targets (T). For a human it's super easy to pick
    # up on the pattern here and understand that the underlying function is
    # likely: f(x) 100 + 100. But by running this code you'll see that for our
    # linear regression it's very difficult as there aren't enough observations
    # here. Perhaps that's an early demonstration of how our current algorithms are
    # so unlike the human brain.
    X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ] # 1-dimensional input
    T = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001] # output
    data = zip(X, T) # single data set of (x, y) tuples
    N = 1 # 1-dimension in the data above.

    # Instead of iterations, we now have a concept of epochs. In the past, we
    # had an infinite data set which we consumed until the model was learned.
    # Here we don't. And just 10 data points isn't enough for our model to
    # learn. There are several work arounds to this like: feature-scaling,
    # adaptive learning rate, mini-batching - but to keep it simple here, we can
    # just duplicate the data multiple times.
    lr = LinearRegression(N)
    for i in xrange(300): # 300 epochs - or iterations over the entire data-set
        l = 0. # sum of losses, to be averaged later.

        for inp, target in data:
            l += lr.minimize(inp, target)

        # instead of printing the loss after every observation (which can be way
        # too verbose), we'll print out the average loss for the entire data set
        print "%s LOSS = %f" % (i, l/len(data))

    print
    print "W = %s" % lr.w


# LinearRegression algorithm
class LinearRegression(object):
    # hyper-parameters
    STEP = 0.01

    # parameters
    w = None

    def __init__(self, n):
        self.n = n
        self.w = np.zeros(1 + n) # 1-extra weight for the bias at index 0

    # minimize the loss function for the given input and target values
    def minimize(self, inp, target):
        w = self.w

        # make our prediction based on our current weights
        inp = np.insert(inp, 0, 1.)
        out = sum(inp * w)

        # compute the loss & derivatives
        l = (out - target) ** 2 / 2
        d = (out - target) * inp

        # update
        self.w += self.STEP * d * -1
        return l



if __name__ == "__main__":
    main()
