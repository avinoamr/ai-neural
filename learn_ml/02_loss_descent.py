# Now, instead of finding the maxima of the function f, we want to find the
# inputs (x, y) that would produce a specific target value, say 3. That might
# seem slightly more involved, because we can't just use the slope to find our
# value. But it turns out that we can wrap our function f, with another function
# known as a loss (or error) fucntion that just computes the difference between
# our actual value and the target value. Then, all we need is to minimize that
# loss function just like we did before. The result will thus be the inputs that
# produce the minimal difference between our value, and our target. If that
# difference is zero, it means that there's no difference and thus our function
# is equal to the target value. Neat!
import numpy as np

TARGET = 3 # this is number we want to hit
E = 0.0001
STEP = 0.01
ITERATIONS = 1000

# f(x,y) - the function we want to fix at the TARGET
# It can be assumed that this function is unknown, and resides as a compiled
# black-box and may contain arbitrary, complicated and human-supervised logic
def f(x, y):
    return x + y

# we're searching for an arbitrary target number, so we can't just minimize or
# maximize the f function. Instead, we use a separate loss function that has a
# minima at the target number (thus, unlike f, it must be lower-bounded). Then
# we can minimize that function to find the best parameters to produce our
# target. In this example, we're using the error squared function which has a
# parabola with a minima at the target. It decends for all values below the
# target, and ascends for all values above the target.
#
# the input to the loss function, is the output of the actual output function f
# which is then compared against the target value to produce the difference (or
# distance; variance) between the actual and expected value. It's squared so
# that we'll have a lower-bound (no negatives) and the right gradient before
# (decsent) and after (ascent) of the target.
def loss(actual):
    return (actual - TARGET) ** 2

# initial values - random [-.5 ... +.5]
x = np.random.rand() * 2 - 1
y = np.random.rand() * 2 - 1

# we're starting with computing the loss of our function. The actual value of f
# is of no interest for us, only its loss is. When the loss = 0 (minima), we
# know we've landed on the right parameters:
#
#       (actual - target)^2 = 0         // sqrt
#       actual - target = 0             // + target
#       actual = target                 // win!
#
# in fact - in supervised machine learning, it's assumed that there's no well-
# defined mathmatical output function because its results are modelled
# (supervised) by a human for every given input. We can instead regard that
# value as a compiled black-boxed function.
for i in xrange(ITERATIONS): # can we stop early once we reach our target?
    l = loss(f(x, y))
    print "#%d f(%f, %f) = %f" % (i, x, y, f(x, y))

    # two samples, inifinitsimal points around x and y
    lx = loss(f(x + E, y))
    ly = loss(f(x, y + E))

    # derivatives of x and y - or how the output of loss() changes w.r.t x, y
    dx = (lx - l) / E
    dy = (ly - l) / E

    # update to the new x, y in the gradient direction
    x += STEP * dx * -1 # looking for the minima, walk against the gradient
    y += STEP * dy * -1 # otherwise, it will find the maxima (infinity)
