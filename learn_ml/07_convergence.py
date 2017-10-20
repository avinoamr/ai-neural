import numpy as np
import random
import itertools

X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ]
T = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
N = 1

STEP = 0.01
def loss(actual, target):
    return (actual - target) ** 2

data = zip(X, T) # single data set of (x, y) tuples

# instead generating a massive list here, we're just repeating the same one.
losses = []
w = np.zeros(1 + N)
for i in itertools.count(): # infinite loop
    avgl = 0
    for x, t in data:
        x = np.insert(x, 0, 1.)
        y = sum(x * w)

        # compute the loss
        avgl += loss(y, t) / len(data)

        # derivatives
        dw = 2 * (y - t) * x

        # update
        w += STEP * dw * -1

    # compute the avg loss of the current epoch
    losses.append(avgl)

    # stop if the last 5 epochs didn't significantly improve the fit.
    if len(losses) == 5:
        diffs = np.diff(losses)
        if np.allclose(diffs, 0):
            break

        losses = []

print
print "EPOCHS = %s ; W = %s" % (i, w)
