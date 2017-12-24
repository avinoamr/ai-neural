# Our previous code was limited to just functions that accept two values
# (x0, x1). We will now generalize it to any function using linear algebra. For
# this experiment we use a vector of size 2 (ie x0, x1) to keep the entire
# input/output logic exactly the same as before, only without hard-coding the
# parameters themselves. You can change N below to see how it's generalizing to
# 3 (x + y + z), 4 (x + y + z + w), or more...
import numpy as np
np.random.seed(1)

# constants
T = 3 # same as before.
N = 2 # size of the input vector; number of parameters as input
E = 0.0001
ALPHA = 0.01
ITERATIONS = 200

# same function as before, except that now it accepts a vector of N inputs (x0,
# x1, ...) instead of a single input. It will be a generalized version of our
# previous f(x0, x1) = x0 + x1 function such that f(X) = X0 + X1 + X2 + X3...
# But remember that this function can be anything!
def f(X):
    return sum(X)

def loss(y):
    return (y - T) ** 2

# initial values. instead of hard coding exactly 2 inputs we'll use N values to
# have a more generalized code that can adhere to any N-input function
x = np.random.random(N) # N-sized vector of random numbers
for j in xrange(ITERATIONS):
    y = f(x)
    e = loss(y)
    print "%d f(%s) = %f (LOSS: %f)" % (j, x, y, e)

    # N samples (instead of 2), inifinitsimal points around the current x
    # gradient of the loss function w.r.t the input, element-wise. d is a vector
    # with N values - corresponding to the derivate of error function w.r.t
    # every index in the inp vector.
    d = np.zeros(N)
    for i in range(N):
        # add an inifinitsimal change to the current index of the input. It's an
        # immutable version of: x[i] += E
        xtemp = np.copy(x)
        xtemp[i] += E

        # sample the loss function after adding E to x[i]
        ei = loss(f(xtemp))

        # derviative of the input - or how the loss() changes w.r.t x[i]
        d[i] = (ei - e) / E

    # element-wise update to the new x in the gradient direction. ie:
    x += ALPHA * d * -1
