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
# non-linear function. But in classification, we can think of the output of the
# hidden layer as a set of new features dataset that's fed into the final layer
# and thus the whole purpose of the hidden layer is to find the features that
# are most relevant for decreasing the loss. In other words, it needs to pick
# the features that are most interesting/relevant for the prediction, using any
# combination of the input features.
#
# In this example, we'll explore a simplified version of the family-tree example
# where we're given a list of family relationships (jacob, son-of, abraham) and
# we need to predict the latter (target of the relationship; abraham). This same
# example can be extended to any kind of relationship modelling. What will be
# especially interesting, is reviewing the learned neurons at the hidden layer.
#
# https://www.coursera.org/learn/neural-networks/lecture/IfaB4/learning-to-predict-the-next-word-13-min
# http://www.cs.toronto.edu/~hinton/absps/families.pdf
import numpy as np
np.random.seed(1)

ALPHA = 3
EPOCHS = 300

# We'll use faux car insurance data, where we wish to predict the likelihood of
# an insurance holder (by Gener & Age) to file a claim:
#
#   1. Young, Female => No
#   2. Young, Male => Yes
#   3. Old, Female => Yes
#   4. Old, Male => No
#
# The data is intentionally designed such that each dimension in the input
# (Gender or Age) cannot perfectly predict the output - thus there's no possible
# set of weights that can compute the correct output.
X = np.array([
    # Young Old     Female Male
    [ 1.,   0.,     1.,    0.],     # Young Female
    [ 1.,   0.,     0.,    1.],     # Young Male
    [ 0.,   1.,     1.,    0.],     # Old Female
    [ 0.,   1.,     0.,    1.],     # Old Male
])

T = np.array([
    [0., 1.],  # No
    [1., 0.],  # Yes
    [1., 0.],  # Yes
    [0., 1.]   # No
])

data = zip(X, T)

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, n, m):
        self.W = np.random.random((m, n + 1)) # +1 for bias

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        z = np.dot(self.W, x) # derivate: x
        y = 1. / (1. + np.exp(-z)) # derivate: y(1 - y)

        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them.
    def backward(self, dy):
        x, y = self._last

        # how the weights affect total loss (derivative w.r.t w)
        dz = dy * (y * (1 - y))
        dw = np.array([d * x for d in dz])

        # how the input (out of previous layer) affect total loss (derivative
        # w.r.t x). Derivates of the reverse of the forward pass.
        dx = np.dot(dz, self.W)
        dx = np.delete(dx, -1) # remove the bias input derivative

        # update
        self.W -= ALPHA * dw
        return dw, dx

# we'll use a hidden layer of 2 neurons and examine the learned weights
l1 = Layer(4, 2)
l2 = Layer(2, 2)
for i in xrange(EPOCHS):
    e = 0.
    for x, t in data:
        # forward
        y = l1.forward(x)
        y = l2.forward(y)

        # backward
        e += (y - t) ** 2 / 2
        d = y - t
        _, d = l2.backward(d)
        _, d = l1.backward(d)

    e /= len(data)
    print "%s: LOSS = %s" % (i, sum(e))

print

# NOTE that if we remove the hidden layer, we obviously can't reach an error
# close to zero because the problem was designed such that no set of weights can
# describe the data perfectly.
#
# But if we look at the weights of the hidden layer, we'll be able to see what
# the network has learned:
print "l1.W=", l1.W

#            Young      Old       Female   Male       Bias
# l1.W = [  [-2.070330  4.346880  3.893700 -2.452050  1.285958]
#           [-5.259462  2.738003  3.075649 -5.133379 -2.261241]   ]
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

#            O|F            O&F           Bias
# l2.W = [  [-5.492996       5.786855       2.567549]  # Yes
#           [ 5.549726      -5.852389      -2.593249]] # No
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
