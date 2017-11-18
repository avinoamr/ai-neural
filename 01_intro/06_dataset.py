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
