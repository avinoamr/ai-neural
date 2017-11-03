# Our previous code was limited to just functions that accept two values (x, y).
# We will now generalize it to any function using linear algebra. For this
# experiment we use a vector of size 2 (ie x, y) to keep the entire inp/out
# logic exactly the same as before, only without hard-coding the parameters
# themselves. You can change N below to see how it's generalizing to
# 3 (x * y * z), 4 (x * y * z * w), or more...
import numpy as np

# constants
TARGET = 3 # same as before.
N = 2 # size of the input vector; number of parameters as input
E = 0.0001
STEP = 0.01
ITERATIONS = 1000

# same function as before, except that now it accepts a vector of N inputs (X0,
# X1, ...) instead of a single input. It will be a generalized version of our
# previous f(x, y) = x + y function such that f(X) = X0 + X1 + X2 + X3...
def f(X):
    return sum(X)

def loss(actual):
    return (actual - TARGET) ** 2

# initial values. instead of hard coding exactly 2 inputs we'll use N values to
# have a more generalized code that can adhere to any N-input function
inp = np.zeros(N) # N-sized vector of random numbers
for j in xrange(ITERATIONS):
    l = loss(f(inp))
    print "#%d f(%s) = %f" % (j, inp, f(inp))

    # N samples (instead of 2), inifinitsimal points around the current inp
    # gradient of the loss function w.r.t the input, element-wise. d is a vector
    # with N values - corresponding to the derivate of loss function w.r.t every
    # index in the inp vector.
    d = np.zeros(N)
    for i in range(N):
        # add an inifinitsimal change to the current index of the input. It's an
        # immutable version of: inp[i] += E
        inptemp = np.copy(inp)
        inptemp[i] += E

        # sample the loss function after adding E to inp[i]
        li = loss(f(inptemp))

        # derviative of the input - or how the loss() changes w.r.t inp[i]
        d[i] = (li - l) / E

    # element-wise update to the new inp in the gradient direction. ie:
    #   inp[i] = STEP * d[i] * - 1 ; for every i in N = all of the inputs
    inp += STEP * d * -1
