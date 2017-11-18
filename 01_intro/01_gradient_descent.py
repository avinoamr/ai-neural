# implementation of numerical gradient optimization, trying to choose the best
# inputs (x, y) that will minimize the function f. This is done by starting with
# two random inputs, and then probing the function to find the slope around
# these random inputs. In other words, we compute the derivatives (dx, dy) which
# are numbers that indicates how the function f changes in response to a tiny
# change in x, y. Then, we walk against that slope in the direction of steepest
# descent - we add some amount to our initial inputs such that the f function
# is expected to be lower. Repeat for a few iterations and stop. The final
# inputs will be the local minima of the function, if one exists.
import random
random.seed(1) # static seed to make results reproducible

def main():
    # f(x,y) - the function we want to minimize. It can be anything you want!
    # Note however, that if this function doesn't have a minima, it can get
    # inifinitely negative, and thus no minima will be found even after infinite
    # iterations. So, for this demonstration, we'll remove negatives by squaring
    # it.
    f = lambda x, y: (x - y) ** 2

    # minimize over a few iterations, printing the newly found x, y values and
    # the output of f(x, y) for these inputs. We're expecting the code to reach
    # an output of zero (minima of f)
    gd = GradientDescent(f)
    for i in xrange(150):
        x, y, out = gd.minimize()
        print "f(%f, %f) = %f" % (x, y, out)

# GradientDescent algorithm fixed to functions for arity 2. Given an input
# function f, this object will gradually find inputs such that the output of the
# function is minimized.
class GradientDescent(object):
    # constants/hyperparameters - these can be learned as well! Maybe we'll
    # cover it later.
    f = None # the function we want to minimize
    E = 0.0001 # epsilon; infinitisimal size of probes to find derivatives
    STEP = 0.01 # size of the steps to take in the gradient direction

    # parameters (actual inputs) to be learned
    x = 0
    y = 0

    # initialize for a function f. Must be of arity 2.
    def __init__(self, f):
        self.f = f

        # init parameters x, y with random points in the range [-1, 1]
        self.x = random.random() * 2 - 1
        self.y = random.random() * 2 - 1

    # one-pass minimalization. Returns the new (x, y) inputs and the output
    # computed for these inputs. It's guaranteed that if the function has a
    # minima, it will be eventually found after enough calls to minimize()
    def minimize(self):
        x, y, f = self.x, self.y, self.f
        out = f(x, y)

        # two samples, inifinitsimal points around the current x and y
        outx = f(x + self.E, y)
        outy = f(x, y + self.E)

        # derivatives of x and y - or how the output of f(x, y) changes
        # w.r.t x, y
        dx = (outx - out) / self.E
        dy = (outy - out) / self.E

        # update to the new x, y in the negative gradient direction
        x += self.STEP * dx * -1 # -1 because we're after the minima, we
        y += self.STEP * dy * -1 # want to walk downwards against the slope

        self.x, self.y = x, y
        return x, y, f(x, y)



if __name__ == "__main__":
    main()
