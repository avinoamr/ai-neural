# One of the greatest challenges I had when I started delving into multi-layered
# networks is the intuition and understanding of the precise behavior exhibited
# by the hidden layer. It felt more of an experimental property that I needs to
# be tuned instead of an accurate property that I can reason about. This
# exercise is an attempt at understaing this property better. We will also
# re-introduce the entire learning process instead of just tinkering with one
# data instance.
#
# In regression, a hidden layer can be thought of as composition of several
# simple non-linear functions building up into a single arbitrarily complex
# non-linear function as shown in (1). But in classification, we can think of
# the output of the hidden layer as a set of new features dataset that's fed
# into the final layer and thus the whole purpose of the hidden layer is to find
# the features that are most relevant for decreasing the error. In other words,
# it needs to pick the features that are most interesting/relevant for the
# prediction, using any combination of the input features.
#
# In this example we'll intentionally design data that's not linearily separable
# thus there's no possible set of weights that can perfectly predict the result.
# This can obviously happen in the real-world, where one feature may produce
# different results, depending on the activation on a different feature.
#
# (1) http://neuralnetworksanddeeplearning.com/chap4.html
import numpy as np
np.random.seed(1)

ALPHA = 0.5
EPOCHS = 1500
H = 2 # number of hidden neurons

# XOR(feature 1, feature 2)
X = np.array([
    [0., 1., 0., 1.],   # XOR(0,0) = 0
    [0., 1., 1., 0.],   # XOR(0,1) = 1
    [1., 0., 0., 1.],   # XOR(1,0) = 1
    [1., 0., 1., 0.]    # XOR(1,1) = 0
])

T = np.array([
    [0., 1.],           # XOR(0,0) = 0
    [1., 0.],           # XOR(0,1) = 1
    [1., 0.],           # XOR(1,0) = 1
    [0., 1.]            # XOR(1,1) = 0
])

class Sigmoid(object):
    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1))

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z))

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total error (derivative w.r.t w)
        dz = dy * (y * (1 - y))
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total error (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        self.W -= ALPHA * dw
        return dx

class SquaredError(object):
    def forward(self, x):
        self._y = x
        return x

    def error(self, t):
        y = self._y
        return (y - t) ** 2 / 2

    # squared error function just returns the simple derivative
    def backward(self, t):
        y = self._y
        return y - t

# we'll use a hidden layer of 2 neurons and examine the learned weights
data = zip(X, T)
l1 = Sigmoid(4, H)
l2 = Sigmoid(H, 2)
l3 = SquaredError()
layers = [l1, l2, l3] # try removing l2 to see that it's unable to learn
for i in xrange(EPOCHS):
    np.random.shuffle(data)
    e = 0.
    accuracy = 0
    for x, t in data:
        # forward
        y = reduce(lambda x, l: l.forward(x), layers, x)
        accuracy += 1 if np.argmax(y) == np.argmax(t) else 0

        # backward
        e += l3.error(t)
        d = reduce(lambda d, l: l.backward(d), reversed(layers), t)

    e /= len(data)
    print "%s: ERROR = %s ; ACCURACY = %s" % (i, sum(e), accuracy)

print

# NOTE that if we remove the hidden layer, we obviously can't reach an error
# close to zero because the problem was designed such that no set of weights can
# describe the data perfectly.
#
# But if we look at the weights of the hidden layer, we'll be able to see what
# the network has learned:
print "l1.W=", l1.W

#            YOUNG      OLD      FEMALE      MALE      BIAS
# l1.W =    -2.07       4.34     3.89       -2.45      1.28
#           -5.25       2.73     3.07       -5.13     -2.26
#
# What does that mean? Lets start with (a) the first neruon: We can see that
# the bias is high, thus by default this neruon will fire. Further, the Old
# weight is much higher than the Young weight, and similar for Female vs Male.
# So, the only way to inhibit this neuron is if both the Young AND Male inputs
# are both turned on. Any other case and this neuron will fire. Thus, this
# neuron encodes the case of either Old OR Female (O|F)
#
# Now for (b) the second neuron. The bias is negative by ~-2.5, so it wouldn't
# fire by default. Also, like (a) Old is much bigger than Young, and Female is
# much bigger than Male. But neither are big enough to overcome the bias on its
# own. The only way to overcome the negative bias is if both Old AND Female are
# turned on together. Thus, this neuron encodes the case of both Old AND Female
# (O&F)
#
# What this means is that our network has learned two mutually inexclusive
# combinations of the input, weirdly enough by only caring about the Old and
# Female inputs: (a) is when either Old OR Female (O|F) are turned on and (b) is
# when both Old AND Female (O&F) are turned on. NOTE that this is no longer one-
# of-k, and thus both, or none, are likely to be turned on. To see how it works
# we'll also need to look at the second output layer:
print "l2.W=", l2.W

#            O|F      O&F      BIAS
# l2.W =    -5.49     5.78     2.56           # Yes
#            5.54    -5.85    -2.59           # No
#
# Now we see that the first output neuron (a) which indicates that a claim is
# likely to be filed, will fire by default due to the big positive bias. The Y|F
# weight has a similar size and is opposite to Y&F, so if both are turned on
# (which happens whenever O&F is on) - it will still not be enough to inhibit
# the large bias and thus it will still fire. The only way to inhibit it is if
# O|F and not O&F. Or, when only Old xor Female are turned on. In our case that
# can only happen with Old & Male or Young & Female. These are exactly the two
# cases where a claim is unlikely to be filed. The opposite of which, when the
# neuron will fire is in the case of Young & Male or Old & Female. The full
# activation expression is:         O&F || !(O|F). This is like an if-statement
# for Old & Female or Young & Male. Exactly matching our data!
#
# The second neuron (b) is the opposite: it's turned off by default due to the
# negative bias, and has the opposite weights to neuron a. Thus when one will
# fire, the other will not.
#
# NOTE that this is just one possible way to dissect the data correctly. Since
# Y|M and Y&M are the opposite that will also work. It all depends on the
# randomness which is seeded with a constant here for reproducible results.
#
# NOTE also that if we use a hidden layer of size 3, following the same exercise
# we end up with some duplicate features in this case: O&F, O|F, O|F.
# So 2 hidden neurons suffices here.
