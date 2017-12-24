# Now, instead of finding the minima of the function f, we want to find the
# inputs (x0, x1) that would produce a specific target value, say 3. That might
# seem slightly more involved, because we can't just use the slope to find our
# value. But it turns out that we can wrap our function f, with another function
# known as a loss (or error) function that just computes the difference between
# our actual value y, and the target value. Then, all we need is to minimize
# that loss function just like we did before. The result will thus be the inputs
# that produce the minimal difference between the value, and the target. If that
# difference is zero, it means that there's no difference and thus our function
# is equal to the target value. Neat!
import numpy as np
np.random.seed(1)

T = 3. # target: this is number we want to hit
E = 0.0001
ALPHA = 0.01
ITERATIONS = 150

# f(x0,x1) - the function we want to fix at the T value.
# It no longer need to be lower-bounded, because we're not minimizing it
def f(x, y):
    return x + y

# we're searching for an arbitrary target number, so we can't just minimize or
# maximize the f function. Instead, we use a separate loss function that has a
# minima at the target number (thus, unlike f, it must be lower-bounded). Then
# we can minimize that function to find the best parameters to produce our
# target. In this example, we're using the squared error function which has a
# parabola with a minima at the target. It decends for all values below the
# target, and ascends for all values above the target.
#
# The input to the loss function is the output of the function f which is then
# compared against the target value to produce the difference (or distance;
# variance) between the actual and expected value. It's squared so that we'll
# have a lower-bound (no negatives) and the right gradient before (decsent) and
# after (ascent) of the target. This is the function we're minimizing.
def loss(y):
    return (y - T) ** 2

# initial values, start at zero
x0 = np.random.rand() * 20 - 10
x1 = np.random.rand() * 20 - 10

# we're starting with computing the loss of our function. The actual value of f
# is of no interest for us, only its loss is. When the loss = 0 (minima), we
# know we've landed on the right parameters:
#
#       (y - target)^2 = 0         // sqrt =>
#       y - target = 0             // + target =>
#       y = target                 // win!
#
# in fact - in supervised machine learning, it's assumed that there's no well-
# defined mathmatical output function because its results are modelled
# (supervised) by a human for every given input. We can instead regard that
# value as a compiled black-boxed function.
for i in xrange(ITERATIONS):
    y = f(x0, x1) # predict
    e = loss(y) # different between prediction & target
    print "%d: f(%f, %f) = %f (LOSS: %f)" % (i, x0, x1, y, e)

    # sample around x0, x1
    ex0 = loss(f(x0 + E, x1))
    ex1 = loss(f(x0, x1 + E))

    # derivatives of x0, x1
    dx0 = (ex0 - e) / E
    dx1 = (ex1 - e) / E

    # update to the new x0, x1
    x0 += ALPHA * dx0 * -1 # looking for the minima, walk against the gradient
    x1 += ALPHA * dx1 * -1 # otherwise, it will find the maxima (infinity)
