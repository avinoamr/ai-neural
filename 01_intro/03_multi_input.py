# Our previous code was limited to just functions that accept two values (x, y).
# We will now generalize it to any function using linear algebra. For this
# experiment we use a vector of size 2 (ie x, y) to keep the entire inp/out
# logic exactly the same as before, only without hard-coding the parameters
# themselves. You can change N below to see how it's generalizing to
# 3 (x * y * z), 4 (x * y * z * w), or more...
import numpy as np

def main():
    # same function as before, except that now it accepts a vector of N inputs
    # (X0, X1, ...) instead of a single input. It will be a generalized version
    # of our previous f(x, y) = x + y function such that f(X) = X0 + X1 + X2...
    f = sum
    ld = TargetLossDescentN(f, 3, 2)
    for i in xrange(200):
        inp, _ = ld.minimize()
        print "#%d f(%s) = %f" % (i, inp, f(inp))


class TargetLossDescentN(object):
    # hyperparams
    E = 0.0001
    STEP = 0.01
    f = None
    target = 3 # same as before.
    n = 2 # size of the input vector; number of parameters as input

    # params
    inp = None

    def __init__(self, f, target, n):
        self.f, self.target, self.n = f, target, n

        # initial values. instead of hard coding exactly 2 inputs we'll use N
        # values to have a more generalized code that can adhere to any N-input
        # function
        self.inp = np.zeros(n) # N-sized vector of zero values

    def loss(self, actual):
        return (actual - self.target) ** 2

    def minimize(self):
        inp, f, loss = self.inp, self.f, self.loss
        l = loss(f(inp))

        # N samples (instead of 2), inifinitsimal points around the current inp
        # gradient of the loss function w.r.t the input, element-wise. d is a
        # vector with N values - corresponding to the derivate of loss function
        # w.r.t every index in the inp vector.
        d = np.zeros(self.n)
        for i in range(self.n):
            # add an inifinitsimal change to the current index of the input.
            # It's an immutable version of: inp[i] += E
            inptemp = np.copy(inp)
            inptemp[i] += self.E

            # sample the loss function at a point infinitisimally close to
            # inp[i]
            li = loss(f(inptemp))

            # derviative of the input - or how the loss() changes w.r.t inp[i]
            d[i] = (li - l) / self.E

        # element-wise update to the new inp in the gradient direction. ie:
        #   inp[i] = STEP * d[i] * - 1 ; for every i in N = all of the inputs
        self.inp += self.STEP * d * -1
        return inp, l



if __name__ == "__main__":
    main()
