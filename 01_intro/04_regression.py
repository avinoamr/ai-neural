# We are now able to find N x-values such that:
#
#   f(x) = TARGET ; x = ?
#
# But the real-life use-cases are exactly inversed: we're given an input X
# vector and we need to find the answer:
#
#   f(x) = ?
#
# of course, we don't know what f does. But lets start by first covering all
# linear functions[1], thus ones that can be reduced to the form:
#
#   f(x,y,z, ...) = ax + by + cz + ... + C = ?
#
# Where the x values are given. All that's left for us to do is to discover the
# weights (a, b, c, ...) & bias (C). These weights are encoded in a w-vector
# that we'll need to learn as we're given examples of inputs and outputs.
#
# Similarily the bias is a single scalar we'll need to learn, in exactly the
# same way as the other weights. So supporting it might include a duplication
# of the same code used for the weights. Instead of duplicating the code, we can
# just regard the bias as 1 extra weight to be learned (N + 1), with the
# exception that the input for that weight is always 1. In other words, the bias
# behaves just like a normal weight for an input that's always fixed at 1.
#
# [1] Of course, that's an unsubstentiated assumption to make for many use
# cases. Generally, there are two ways to generalize for non-linear functions:
#   1. We can extend the input to include polynomial components. Instead of just
#       using the input as is, we can first produce multiple copies of the same
#       inputs, only raised to the power of 2, 3, etc.
#   2. Using hidden layers. A concept I will maybe cover later.
#
import numpy as np

def main():
    # slightly different implementation of f() just for fun:
    #   f(X) = 10 + 8x1 -2x2 + x3/2
    #
    # So the learned weights are [8, -2, .5] and the bias 10.
    #
    # Notice that this code will not be able to learn non-linear functions
    # (x^2). But many real-life situations can be reduced to a linear expression
    f = lambda x: 10 + 8 * x[0] - 2 * x[1] + x[2] / 2
    N = 3
    lr = LinearRegression(3)
    for i in xrange(1000):
        # first we need a new input for each iteration. In reality, we should
        # receive these inputs from an external training data set. But for now
        # we'll cheat, by just randomizing an input
        inp = np.random.rand(N) * 2 - 1 # cheat.

        # the target, or correct value, for this input
        target = f(inp)
        loss = lr.minimize(inp, target)
        print "#%d f(%s) = %f (loss: %f)" % (i, inp, target, loss)

    # finally - lets print out our weights to see what we did. You should expect
    # the weights to resemble the ones in f(). Remember that the first weight is
    # actually the bias
    print "W = %s" % lr.w



class LinearRegression(object):
    # hyper-parameters
    # no constant TARGET; we need to find it.
    # no function - we're given inputs and targets directly
    E = 0.0001
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

    # same loss function as before, except that now it can't rely on a constant
    # target value for all inputs, but instead it receives the value as input
    def loss(self, actual, target):
        return (actual - target) ** 2

    # attempt to minimize the loss function for a given observation of N inputs
    # and the target result for this input. We don't have access to the function
    # being fit, so we must only rely on the inp and target
    def minimize(self, inp, target):
        w, loss = self.w, self.loss

        # we start by making our prediction. Again, because it's assumed to be a
        # linear function, thus it must be reducible to the form of:
        #
        #   f(x) = w1x1 + w2x2 + w3x3 + ... + C = ?
        #
        # thus we need to multiply all inputs with their weights, element wise,
        # and sum up the result.
        inp = np.insert(inp, 0, 1.) # inject fixed input 1 for the bias weight
        out = sum(inp * w)

        # now, lets find our current loss - comparing our prediction to the
        # actual value produced by f():
        l = loss(out, target)

        # just as before, we now want to make infinitisimal changes to our
        # weights, in order to find how the loss changes w.r.t to every
        # individual weight. This is identical to what we did before.
        d = np.zeros(w.shape) # one derivative per weight
        for i in range(len(d)):
            # add an inifinitsimal change to the current index of the input.
            # It's an immutable version of: w[i] += E
            wtemp = np.copy(w)
            wtemp[i] += self.E

            # sample the loss function after adding E to w[i]
            # we're making a new prediction, just like before, only now we add
            # the epsilon to the current weight i. Also notice that the target
            # of our loss doesn't change obviously (because the inp is the
            # same), only the predition does
            li = loss(sum(inp * wtemp), target)

            # derviative of the input - or how the loss() changes w.r.t inp[i]
            d[i] = (li - l) / self.E

        # now we update the weights, same as before.
        # element-wise update to the new inp in the gradient direction. ie:
        #   inp[i] = STEP * d[i] * - 1 ; for every i in N = all of the inputs
        self.w += self.STEP * d * -1

        return loss(sum(inp * self.w), target)



if __name__ == "__main__":
    main()
