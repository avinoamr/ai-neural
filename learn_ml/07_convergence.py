# In the previous example, we had to pick the right number of epochs ahead of
# time, which may involve some hand-tuning. We will see here how we can detect
# the convergence of the loss function automatically. It's based on the idea
# that if the loss function didn't improve significantly in the last 5 epochs,
# it's safe to assume that there's very little learning still happening - thus
# the weights are pretty much fixed. By running this code you'll see that we
# really need only ~300 epochs to find the right weights, instead of 1000.
import numpy as np
import itertools

X = [1,   2,   3,   4,   5,   6,   7,   8,   9 ,  10 ]
T = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
N = 1
STEP = 0.01

def loss(actual, target):
    return (actual - target) ** 2

data = zip(X, T)

# losses is an array that will hold the average loss of the last 5 epochs. For
# each epoch, we'll append the average loss to this list, and when it's full
# (len = 5), we will find the difference between these losses to determine if
# the learning has stopped and converged. If the losses are pretty much static
# it means that no significant improvement was made in the last few epochs, so
# we should stop. Otherwise, we should keep going.
losses = []
w = np.zeros(1 + N)
for i in itertools.count(): # infinite loop
    avgl = 0 # average loss for this epoch
    for x, t in data:
        x = np.insert(x, 0, 1.)
        y = sum(x * w)

        # add the current loss to the average
        avgl += loss(y, t) / len(data)

        # derivatives
        dw = 2 * (y - t) * x

        # update
        w += STEP * dw * -1

    # remember this last loss in our window of 5 latest losses
    losses.append(avgl)

    # convergence-test
    # if the last 5 epochs didn't significantly improve the loss it means that
    # our weights are pretty much fixed and the minima of the loss function was
    # found.
    if len(losses) == 5:
        # compute the difference between the losses. Equal loss values would
        # produce a difference of zero - indicating that no improvement was
        # made. Notice that while in this example we can search directly for
        # loss = 0 , it wouldn't generalize because it's likely to have data
        # with many outliers in which the best loss we can achieve is higher
        # than zero. Thus we're only interested in the improvement of the loss
        # being stagnate, and not the absolute value of the loss being zero.
        diffs = np.diff(losses)
        if np.allclose(diffs, 0):
            # all of the differences are (almost) zero, this means that there's
            # virtually no difference between the losses of the last few epochs,
            # thus they're (almost) equal. We're not learning any more.
            break

        losses = [] # start a new window of 5 last epochs

print
print "EPOCHS = %s ; W = %s" % (i, w)
