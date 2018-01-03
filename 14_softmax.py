# Finding that final prediction, while useful, may prove insufficient - because
# it hides away every other outcome except for the one most likely one. In real
# life data, there will rarely be a direct perfect correlation between the input
# and output. Usually there will be some "noise" in the data, some outliers that
# doesn't fit perfectly. This is because it can be assumed that the inputs
# doesn't capture every possible feature in the problem domain. For example, if
# our inputs represent the number of items purchased, and the target output
# represents the cost of these items, it's possible to have outliers where the
# data doesn't fit perfectly, due to special promotions, sales, surges,
# negotiations, etc. that are not captured in the data inputs.
#
# Imagine the other case when the output is equally likely to be in two classes.
# Our current model will just pick the one that's marginally more likely (51% vs
# 49%), thus eliminating any occurances of the other. The predicted value will
# always be the same (100% vs 0%). For example, in visual recognition, a picture
# might contain multiple objects, and we wish to know all of these objects
# (along with their prominence in the image). Another example in visual
# recognition is that our model might confuse a cheetah and a giraffe, and we
# wish to know the likelihood of each option.
#
# Instead of our current model, we prefer to have a probablistic output -
# one that captures the statistical likelihood of the input to fit within each
# class, in order to (a) allow ourselves to manually examine all of the
# possibilities, or maybe a few of the top ones, and (b) perhaps even
# stochastically choose one of them based on that distribution. The latter is
# useful in generative systems where we want our model to choose an output that
# seems more aligned with the training data.
#
# One way of achieving it is the softmax final layer along with the
# cross-entropy error function that together forces the output to represent a
# distribution of probablities.
#
# See: https://deepnotes.io/softmax-crossentropy
import numpy as np
np.random.seed(1)

ALPHA = 0.01
EPOCHS = 400

# in our new data, regardless of the input, the target is almost always 1. But,
# in a fifth of the cases output is turned off. Out goal is that our final
# output woudln't just be the most likely prediction (in this case - 1), but
# instead will show the 80%/20% probablity distribution of these outputs. So
# when we want our output to be [0.8, 0.2]
X = np.array([[1.],     [1.],     [1.],     [1.],     [1.]])
T = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]])
#                                                      ^^ outlier

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.W = np.random.randn(m, n + 1)

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = np.tanh(z)

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total error (derivative w.r.t w)
        dz = dy * (1 - y ** 2)
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total error (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        self.W -= ALPHA * dw
        return dw, dx

l1 = Layer(1, 2)
data = zip(X, T)
for i in xrange(EPOCHS):
    np.random.shuffle(data)

    e = 0.
    dist = 0.
    for x, t in data:

        # forward
        y = l1.forward(x)

        # softmax
        # this step (often implemented as a separate final layer) squashes the
        # arbitrary values we used until now to one where the sum of all of the
        # values add up to 1. This is obviously required for representing
        # probablistic distribution of possibilities, as the total has to be
        # 100% (or 1):
        #
        #   softmax(X) = exp(X) / sum(exp(X))
        #
        # This function has several important properties: (a) it maintains the
        # relative sizes of the different values, thus it's a generalization of
        # the previous example. It also means that we can still use the max
        # value to predict the most likely result similar to how it was done
        # beforehand. (b) it sums to 1, which makes it kind of a zero-sum game,
        # as we increase one of the values, we have to decrease all others. A
        # fact that will be useful later when we look at the error and
        # derivatives. (c) it computes the log probablities, rather than the
        # flat probabilities such that the differences between become more
        # emphasied. For example:
        #
        #   y = [1., 2., 3., 4.]                =>
        #   p = [0.032, 0.087, 0.237, 0.644]    =>
        #   SUM(p) = 1
        #
        # NOTE that this is somewhat similar to lateral inhibition in biological
        # neural networks due to the fact that the largest activation inhibits
        # the other lesser activations, by the fact that we find the log
        # probablities rather than just standard normalization.
        p = np.exp(y) / np.sum(np.exp(y))

        # error & derivative
        # For the first time we're introducing a different error function: the
        # cross-entropy error function. Remember that the purpose of the error
        # function is to measure the difference between a computed value and an
        # actual target value. When we walk against the gradient of that
        # function, we're forcing all of the lower layers to learn weights that
        # would reduce this difference. Our new cross-entropy function does
        # exactly that:
        #
        #   cross-entropy = t * -log(y)     ; for every (t, y) neurons.
        #
        # Because t is either 0 or 1, the cross entropy will be either 0 or
        # log(y). If the target value is zero, that specific neuron will have
        # zero error, regardless of its output. But if it's one, the negative
        # log will be very for large errors (like when y = 0.001)[1]. NOTE That
        # this means that as long as the correct answer is predicted, the loss
        # will be zero even if the other predictions are wrong (false
        # positives) - but due to softmaxing it's increasingly impossible
        # because we forced the values to sum to 1. Every increase in the
        # correct value will always mean a large decrease in the incorrect
        # values. In other words, we only care about the loss of the target
        # prediction, as it will force all of the other ones into place.
        #
        # [1] https://www.wolframalpha.com/input/?i=-log(x)
        e += sum(t * -np.log(p))

        # Now for the derivatives. We've added two expressions that we need to
        # derive: the softmax function and the cross-entropy error function. I
        # will not cover the entire derivation process here, as it's way too
        # lengthy[2], but it turns out that the chained derivative of both the
        # cross-entropy error function and the softmax function is: p - t
        #
        # [2] https://deepnotes.io/softmax-crossentropy
        dy = p - t

        # and now, backwards:
        l1.backward(dy)

        # instead of accuracy, we'll now measure the distribution
        dist += p

    dist /= len(data) # average out the probablity distribution
    e /= len(data) # average out the error
    print "%s: ERROR = %s ; DISTRIBUTION = %s" % (i, e, dist)
