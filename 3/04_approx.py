# We are now able to find N x-values such that:
#   f(x) = TARGET ; x = ?
#
# But the real-life use-cases are exactly inversed: we're given an input X
# vector and we need to find the answer:
#   f(x) = ?
#
# of course, we don't know what f does. But lets start by first covering all
# linear functions[1], thus ones that can be reduced to the form:
#
#   f(x,y,z, ...) = ax + by + cz + ... + C = ?
#
# Where the x values are given, all that's left for us to do is to discover the
# weights (a, b, c, ...) & bias (C). These weights are encoded in a w-vector
# that we'll need to learn as we're given examples of inputs and outputs.
#
#
# [1] Of course, that's an unsubstentiated assumption to make for many
# use-cases, but we'll see later how to generalize for non-linear functions
import numpy as np

# no constant target; we need to find it.
N = 2 # size of the input vector; number of parameters as input
E = 0.0001 # epsilon; infinitisimal size of probes to find derivatives
STEP = 0.01 # size of the steps to take in the gradient direction
ITERATIONS = 1000 # number of probes/step to take - in fact, we only need ~100

# slightly different implementation of f() just for fun:
#   f(X) = 10 + x1 + 2x2 + 3x3 + ...
#
# So the learned weights are [1, 2, 3, ...] and the bias 10.
#
# Notice that this code will not be able to learn non-linear functions (x^2).
# But many real-life situations can be reduced to a linear expression.
def f(X):
    return 10 + sum([(i + 1) * x for i, x in enumerate(X)])

# same loss function as before, except that now it can't rely on a constant
# TARGET value for all inputs, but instead it receives the value as input
def loss(actual, target):
    return (actual - target) ** 2

# initial weights. before we've learned the input, now that the input is given
# and immutable we learn the weights by which to multiply this input. In fact,
# these weights can represent any linear function (almost), so all that's left
# to do is find these weights and we have our function!
w = np.random.rand(N) * 2 - 1 # N-sized weights vector of random numbers
b = np.random.rand() * 2 - 1
for j in xrange(ITERATIONS): # can we stop early once we reach our target?

    # first we need a new input for each iteration. In reality, we should
    # receive these inputs from an external training data set. But for now we'll
    # cheat, by just randomizing an input
    inp = np.random.rand(N) * 2 - 1 # cheat.

    # we start by making our prediction. Again, because it's assumed to be a
    # linear function, thus it must be reducible to the form of:
    #
    #   f(x) = w1x1 + w2x2 + w3x3 + ... + C = ?
    #
    # thus we need to multiply all inputs with their weights, element wise, and
    # sum up the result.
    prediction = sum(inp * w, b) # adding the bias as well.

    # now, lets find our current loss - comparing our prediction to the actual
    # value produced by f():
    out = loss(prediction, f(inp))
    print "#%d f(%s) = %f (loss: %f)" % (j, inp, f(inp), out)

    # just as before, we now want to make infinitisimal changes to our weights,
    # in order to find how the loss changes w.r.t to every individual weight.
    # This is identical to what we did before.
    d = np.zeros(N) # N derivatives - one per weight
    for i in range(N):
        # add an inifinitsimal change to the current index of the input. It's an
        # immutable version of: w[i] += E
        e = np.zeros(N)
        e[i] = E

        # sample the loss function after adding E to w[i]
        # we're making a new prediction, just like before, only now we add the
        # epsilon to the current weight i. Also notice that the target of our
        # loss doesn't change obviously (because the inp is the same), only the
        # predition does
        outi = loss(sum(inp * (w + e), b), f(inp))

        # derviative of the input - or how the loss() changes w.r.t inp[i]
        d[i] = (outi - out) / E

    outb = loss(sum(inp * w, b + E), f(inp))
    db = (outb - out) / E
    b += STEP * db * -1

    # now we update the weights, same as before.
    # element-wise update to the new inp in the gradient direction. ie:
    #   inp[i] = STEP * d[i] * - 1 ; for every i in N = all of the inputs
    w += STEP * d * -1


# finally - lets print out our weights to see what we did:
# you should expect the weights to resemble the ones in f().
print "W = %s ; b = %s" % (w, b)
