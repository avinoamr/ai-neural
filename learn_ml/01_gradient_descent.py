# implementation of numerical gradient optimization, trying to choose the best
# inputs (x, y) that will minimize the function f. This is done by starting with
# two random inputs, and then probing the function to find the slope around
# these random inputs. In other words, we compute the deltas (dx, dy) which is
# a number that indicates how the function f changes in response to a tiny
# change in x, y. Then, we walk against that slope in the direction of steepest
# descent - we add some amount to our initial inputs such that the f function
# is expected to be lower. Repeat for a 1000 iterations and stop. The final
# inputs will be the local minima of the function, if one exists.
#
# f(x,y) - the function we want to maximize. It can be anything you want!
# Note however, that if this function doesn't have a minima, it can get
# inifinitely negative, and thus no minima will be found even after infinite
# iterations. So, for this demonstration, we'll remove negatives by squaring it.
def f(x, y):
    return (x - y) ** 2

# constants - these can be learned as well! Maybe we'll cover it later.
E = 0.0001 # epsilon; infinitisimal size of probes to find derivatives
STEP = 0.01 # size of the steps to take in the gradient direction
ITERATIONS = 1000 # number of probes/step to take

# initial values
x = +2. # numpy.random.rand() * 2 - 1
y = -3. # numpy.random.rand() * 2 - 1

for i in xrange(ITERATIONS):
    out = f(x, y)
    print "f(%f, %f) = %f" % (x, y, out)

    # two samples, inifinitsimal points around x and y
    outx = f(x + E, y)
    outy = f(x, y + E)

    # derivatives of x and y - or how the output of f(x, y) changes w.r.t x, y
    dx = (outx - out) / E
    dy = (outy - out) / E

    # update to the new x, y in the gradient direction
    x += STEP * dx * -1 # -1 because we're after the minima, we want to walk
    y += STEP * dy * -1 # for slope
