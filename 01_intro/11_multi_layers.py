# Everything we've achieved until now can arguably be done with simple
# statistics and Perceptrons because it only applies to linear relationships
# between the input and output classes. But in order to generalize to more
# complex non-linear examples, we'll want to use multiple layers in our neural
# network. That's effectively a pipeline of weights, same as before, where the
# output of each layer is fed as the input to the next. The real input is fed to
# the bottom-most layer, and the output of the top-most layer is the final
# output of our network.
import numpy as np

STEP = .5
EPSILON = 0.0001 # it's back!

# our example data includes only a single 2D instance. We're using fixed values,
# and only a single epoch in order to avoid all randomness and measure the
# expected updated weights.
X = np.array([.05, .10])
T = np.array([.01, .99])

# our weights are also fixed at a certain starting point. But this time, we have
# two sets of weights - one per layer. Each layer is currently configured as
# MxN where N is 3D (input 2D + bias) and M is 2D.
Wxh = np.array([[.15, .20, .35], [.25, .30, .35]]) # input to hidden
Why = np.array([[.40, .45, .60], [.50, .55, .60]]) # hidden to output

# In order to avoid code repetition for each weights matrix, we'll use a Layer
# class to implement the prediction and derivatives:
class Layer(object):
    W = None

    def __init__(self, w):
        self.W = w

    # forward pass - compute the predicted output for the given input
    def forward(self, x):
        x = np.append(x, 1.) # add the fixed input for bias
        net = np.dot(self.W, x) # derivate: x
        y = 1 / (1 + np.exp(-net)) # sigmoid activation; derivate: y(1 -y)
        return y

# now lets create our two layers with the weights we've created before:
l1 = Layer(Wxh)
l2 = Layer(Why)

# predict the output and loss for the given input and target
def predict(x, t):
    h = l1.forward(X)
    y = l2.forward(h) # output from first layer is fed as input to the second

    # now compute our error, same as before
    e = (y - t) ** 2 /2
    return y, e

# predict the output of our single-instance training set:
_, e = predict(X, T) # = (0.274811083, 0.023560026)
print "LOSS %s" % sum(e) # = 0.298371109

# Now's the tricky bit - how do we learn the weights? Before, we've used
# calculus to compute the derivative of the loss function w.r.t each weight. We
# can do it again here: dw = np.array([d * x for d in y - t])   But this will of
# course only apply to the last layer, because we're disregarding the internal
# weights and hidden state. Instead, we want to learn how every weight, in both
# layers, affects the final error. The best known algorithm to calculate all of
# these weights at once is the Back Propagation algorithm that will be discussed
# later.
#
# For now - we'll use a different approach: pertubations. This is similar to
# what we did initially with numeric gradient descent. We will try making a tiny
# change in each weight, and re-compute the total loss produced. The normalized
# difference in loss will be our approximation of the derivative.
Ws = [l1.W, l2.W] # all of the weights in the network
dWs = [] # derivatives of all weights in both layers.
for w in Ws: # iterate over all weight matrices in the network
    dW = np.zeros(w.shape)

    # for every weight - re-run the entire network after applying a tiny change
    # to that weight in order to measure how it affects the total loss.
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j] += EPSILON # add a tiny epsilon amount to the weight
            _, e_ = predict(X, T) # re-run our network to predict the new error
            dW[i][j] = sum(e - e_) / EPSILON
            w[i][j] -= EPSILON # revert our change.

    dWs.append(dW)

# Now we're ready for our update - same as before:
for W, dW in zip(Ws, dWs):
    W += STEP * dW

# print the updated weights
print "l1.W ="
print l1.W # = (0.149780, 0.199561, 0.345614), (0.249751, 0.299502, 0.345022)
print "l2.W ="
print l2.W # = (0.358916, 0.408666, 0.530751), (0.511301, 0.561370, 0.619047)
