# Now, instead of finding the minima of the function f, we want to find the
# inputs (x, y) that would produce a specific target value, say 3. That might
# seem slightly more involved, because we can't just use the slope to find our
# value. But it turns out that we can wrap our function f, with another function
# known as a loss (or error) function that just computes the difference between
# our actual value and the target value. Then, all we need is to minimize that
# loss function just like we did before. The result will thus be the inputs that
# produce the minimal difference between our value, and our target. If that
# difference is zero, it means that there's no difference and thus our function
# is equal to the target value. Neat!
def main():
    # f(x,y) - the function we want to fix at the TARGET
    # It can be assumed that this function is unknown, and resides as a compiled
    # black-box and may contain arbitrary, complicated and human-supervised
    # logic. It also doesn't need to be lower-bounded, because we're not
    # minimizing it
    f = lambda x, y: x + y
    ld = TargetLossDescent(f, 3) # find inputs x, y such that f(x, y) = 3
    for i in xrange(1000):
        x, y, out, loss = ld.minimize()
        print "#%d f(%f, %f) = %f" % (i, x, y, out)

# TargetLossDescent is just like GradientDecent, except that it minimized the
# loss function for a single fixed target value. In other words, it finds the
# parameters x, y that will produce the desired target output, if such exists.
class TargetLossDescent(object):
    # hyper parameters
    target = 0 # our target output
    f = None
    E = 0.0001
    STEP = 0.01

    # parameters - starts at zero
    x = 0
    y = 0

    # f must be of arity 2; target is our target output
    def __init__(self, f, target):
        self.f, self.target = f, target

        # we're not randoming the parameters, because unlike the GradientDecent
        # algorithm, a value of 0 might still produce a significant loss (error)
        # when compared with a certain targets.

    # loss function wraps around f, returns a value indicative of how close we
    # are to the target output (zero being a perfect hit, large value = large
    # error).
    #
    # we're searching for an arbitrary target number, so we can't just minimize
    # or maximize the f function. Instead, we use a separate loss function that
    # has a minima at the target number (thus, unlike f, it must be lower-
    # bounded). Then we can minimize that function to find the best parameters
    # to produce our target. In this example, we're using the error squared
    # function which has a parabola with a minima at the target. It decends for
    # all values below the target, and ascends for all values above the target.
    #
    # the input to the loss function is the output of the actual output function
    # f which is then compared against the target value to produce the
    # difference (or distance; variance) between the actual and expected value.
    # It's squared so that we'll have a lower-bound (no negatives) and the right
    # gradient before (decsent) and after (ascent) of the target.
    def loss(self, actual):
        return (actual - self.target) ** 2

    # Code is very similar to GradientDecent, except that it minimized the loss
    # function, instead of the actual function.
    #
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
    def minimize(self):
        x, y, f, loss = self.x, self.y, self.f, self.loss
        l = loss(f(x, y))

        # sample around x, y
        lx = loss(f(x + self.E, y))
        ly = loss(f(x, y + self.E))

        # derivatives of loss(x, y) w.r.t. x and y
        dx = (lx - l) / self.E
        dy = (ly - l) / self.E

        # update to the new x, y
        x += self.STEP * dx * -1
        y += self.STEP * dy * -1

        self.x, self.y, out = x, y, f(x, y)
        return x, y, out, loss(out)


if __name__ == "__main__":
    main()
