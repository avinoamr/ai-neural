# One conceptual problem some people face when using multiple layers, is
# understanding the intuition around what kind of capabilities it adds to the
# network - especially under certain configurations - like the dimensions and
# number of hidden layers. We will explore that here.
#
# Essentially - having 2-layers network (input to hidden, hidden to output), is
# exactly the same as having 1-layer (input to output) except that we have some
# arbitrary code modify our input. So it's the same kind of logic, but with a
# different input. This means that we can think of adding hidden layers just as
# a process of transforming the input in some way. The broader term here is
# feature-selection.
#
# The features of our data are the encoded inputs that are fed into the network.
# In the titanic example, we had some hand-selected features. You can imagine
# that that the original records from the titanic had a lot more information
# than the limited set we've used. So someone had to manually pick which
# features likely matter the most and input this sub-set of the data into our
# training and test set. This is obviously a huge limitation because it requires
# some assumptions about the correlation between the input and output. Imagine
# problems that involve image or video recognition where the input may contain
# 1024x1024 pixels as input. It's far more difficult to choose a subset in these
# cases. Of course - we could just use all possible inputs, resulting in much
# slower and more complicated learning - but it will work. For linear cases.
#
# Another, perhaps more complicated challenge of feature selection - is that
# features may be more abstract - like some combinations of other features in
# the data. This means that the correlation with the output may be non-linear.
# For example - in the titanic example, we might want to distinguish Old-Female
# as a separate feature because it's not just a linear summation of the two
# separate features: Old + Female. It's possible that the rules (weights) of
# Old-Female passengers are completely different (perhaps even the opposite of)
# Old + Female. One such speculation is that maybe if the passenger was female
# and old she's perhaps even more likely to survive than if she's female + kid.
# This means that in order to support that with one layer, we'll need every
# possible combinations of all features - exploding the dataset.
#
# NOTE that in the previous exercise we took a detour away from the fully-fleged
# learning procedure in favor of focusing on the back-prop algorithm for one
# specific instance. In this example we'll combine the back-prop algorithm with
# our full learning process.
import numpy as np

# In this dataset we will attempt to learn the same data using different
# configurations and then inspect the weights produced by the hidden layer to
# understand what was encoded by that layer.
X = []
T = []

# Layer represents a single neural network layer of weights
class Layer(object):
    W = None
    _last = (None, None) # input, output

    def __init__(self, w):
        self.W = w

    # forward pass is the same as before.
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        y = np.dot(self.W, x) # derivate: x
        self._last = x, y
        return y

    # backward pass - compute the derivatives of each weight in this layer and
    # update them, returning the derivatives of the input to this layer
    def backward(self, dy):
        x, y = self._last
        dw = np.array([d * x for d in dy])
        dx = np.dot(dy, self.W)
        self.W -= ALPHA * dw # update
        return np.delete(dx, -1) # remove the bias derivative
