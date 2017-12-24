# implementation of numerical gradient optimization, trying to choose the best
# inputs (x0, x1) that will minimize the function f. This is done by starting
# with two random inputs, and then probing the function to find the slope around
# these random inputs. In other words, we compute the derivatives (dx0, dx1)
# which are numbers that indicates how the function f changes in response to a
# tiny change in x0, x1. Then, we walk against that slope in the direction of
# steepest descent - we add some amount to our initial inputs such that the f
# function is expected to be lower. Repeat for a 1000 iterations and stop. The
# final inputs will be the local minima of the function, if one exists.
import numpy as np
np.random.seed(1) # constant seed for reproducible results

# f(x0,x1) - the function we want to minimize. It can be anything you want! Note
# however, that if this function doesn't have a minima, it can get inifinitely
# negative, and thus no minima will be found even after infinite iterations. So,
# for this demonstration, we'll remove negatives by squaring it. It can be
# assumed that this function is unknown, and resides as a compiled black-box and
# may contain arbitrary and complicated logic
def f(x0, x1):
    return (x0 - x1) ** 2

# constants - these can be learned as well! Maybe we'll cover it later.
E = 0.0001 # epsilon; infinitisimal size of probes to find derivatives
ALPHA = 0.01 # size of the steps to take in the gradient direction
ITERATIONS = 150 # number of probes/step to take

# initial values in [-10 .. +10]
x0 = np.random.rand()
x1 = np.random.rand()

for i in xrange(ITERATIONS):
    y = f(x0, x1)
    print "%s: f(%f, %f) = %f" % (i, x0, x1, y)

    # two samples, inifinitsimal points around x0 and x1
    yx0 = f(x0 + E, x1)
    yx1 = f(x0, x1 + E)

    # derivatives of x0 and x1 - or how the output of f changes w.r.t x0, x1
    dx0 = (yx0 - y) / E
    dx1 = (yx1 - y) / E

    # update to the new x, y in the gradient direction
    x0 += ALPHA * dx0 * -1 # -1 because we're after the minima, we want to walk
    x1 += ALPHA * dx1 * -1 # downwards against the slope
