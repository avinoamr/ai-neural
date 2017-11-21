# Everything we've achieved until now can arguably be done with simple
# statistics and Perceptrons because it only applies to linear relationships
# between the input and output classes. But in order to generalize to more
# complex non-linear examples, we'll need to take advantage of one of the major
# breakthroughs in Machine Learning: Back Propogation.
#
# We'll now take a small detour from our previous code, to implement the
# backpropagation tutorial:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
import numpy as np

# learning rate
ALPHA = 0.5

# We have 2 input neurons (+ bias = 3 total), 2 hidden neurons
# (+ bias = 3 total) and 2 output neurons. So we need, one weights matrix per
# layer (input to hidden, hidden to output), both of size 2x3 (3 input, 2
# output). We'll use the preset weights from the example:
#
# input to hidden:      w1    w2  b1     w3   w4   b1
Wxh =       np.array([[.15, .20, .35], [.25, .30, .35]])

# hidden to output:     w5   w6   b2     w7   w8   b2
Why =       np.array([[.40, .45, .60], [.50, .55, .60]])

# single instance inputs/targets from example:
x   = [.05, .10, 1.]
t   = [.01, .99]

# FORWARD PASS
# ------------
#
# the forward pass is the same as before. We want to compute the total net input
# to the hidden layer. We have two hidden neurons, h1 and h2:
net_h1 = sum(Wxh[0] * x) # = 0.3775
net_h2 = sum(Wxh[1] * x)

# run it through the sigmoid activation function
out_h1 = 1 / (1 + np.exp(-net_h1)) # = 0.593269992
out_h2 = 1 / (1 + np.exp(-net_h2)) # = 0.596884378

# compact into an array of all hidden outputs and add the new bias:
h = np.array([out_h1, out_h2, 1.])

# repeat for the output neurons, this time the input is the output from the
# hidden layer h:
net_o1 = sum(Why[0] * h) # = 1.105905967
net_o2 = sum(Why[1] * h)

out_o1 = 1 / (1 + np.exp(-net_o1)) # = 0.75136507
out_o2 = 1 / (1 + np.exp(-net_o2)) # = 0.772928465

# that's it!
o = np.array([out_o1, out_o2])


# TOTAL ERROR
# -----------
Eo1 = (t[0] - o[0]) ** 2 / 2 # = 0.274811083
Eo2 = (t[1] - o[1]) ** 2 / 2 # = 0.023560026
Etotal = Eo1 + Eo2 # = 0.298371109


# BACKWARD PASS - Output Layer
# ----------------------------
#
# Now we want to walk backwards and compute the derivatives of the total error
# w.r.t every weight in both layers. In other words, we want to know how every
# weight affects the loss function. There are several ways to go about that. One
# option is to use pertubations: basically attempt to change one weight at a
# time and measure what effect it had to the total loss. This is similar to our
# initial numeric gradient descent. The Back Propogation algorithm is much
# faster - it allows us to learn all of the derivatives using math all at once.
#
# We'll start with the output neurons, and their incoming weights (Why). These
# are exactly the same as before - we know how each weight affects the total
# error, without caring about how the inputs to this layer were produced (i.e
# the hidden layer). For example, the derivative of w5 (Why[0][0]) going from
# h1 to o1:
#
#   dEtotal   dEtotal   dout_o1   dnet_o1
#   ------- = ------- * ------- * -------
#   dw5       dout_o1   dnet_o1   dw5
#
# This makes sense: (math) it's the chain rule of all of the operations going
# from w5 all the way to our total error. Or (intuition) when we (a) change w5,
# it changes the net output of o1 (net_o1), and then (b) that change to net_o1
# also changes the output of the sigmoid (out_o1), and then (c) that change to
# out_o1 also changes Etotal. So that tiny change to w5 generated this series of
# changes - and we need to derive each one and multiply their effects.
#
# (a) We will now compute all of the components in this equation, starting with
# how the total error changes when the output of o1 changes. This is the
# derivative of the squared difference error function (dEtotal/o1):

#                 dEo1/dout_o1   +  dEo2/dout_o1 (zero - o1 doesn't change Eo2)
dEtotal_dout_o1 = -(t[0] - o[0]) + 0. # = 0.74136507

# (b) Now for the second component: how does the output o1 changes w.r.t its
# total net input (h). This is the derivative of the sigmoid activation
# function, which is: dout_o1/dnet_o1 = o1(1 - o1)
dout_o1_dnet_o1 = out_o1 * (1 - out_o1) # = 0.186815602

# (c) We're at the last component: how does the total net input to o1 changes
# w.r.t w5. That's easy, since it's the derivative of the sum of multiples
# between the input to this layer (h) and the weights. But since we only care
# about the partial derivative w.r.t to w5, it doesn't affect any of the other
# inputs to net_o1, except for its input h1:
dnet_o1_dw5 = out_h1 + 0. + 0. # = 0.593269992

# Finally - going back to the original equation. We want to multiply these
# effects because every change to one component will multiply the effect of the
# other component, until we reach the final dEtotal/dw5:
dEtotal_dw5 = dEtotal_dout_o1 * dout_o1_dnet_o1 * dnet_o1_dw5 # = 0.082167041

# Perfect. Now we want to do the same for every other weight in the output
# layer. The whole calculation is:
#
#   dEtotal/w[i] = -(t[j] - o[j]) * o[j] * (1 - o[j]) * x[i]
dEtotal_dw6 = -(t[0] - o[0]) * o[0] * (1 - o[0]) * h[1]
dEtotal_dw7 = -(t[1] - o[1]) * o[1] * (1 - o[1]) * h[0]
dEtotal_dw8 = -(t[1] - o[1]) * o[1] * (1 - o[1]) * h[1]

# Now that we have the derivative. We can update the weight:
Why[0][0] -= ALPHA * dEtotal_dw5 # = 0.35891648
Why[0][1] -= ALPHA * dEtotal_dw6 # = 0.408666186
Why[1][0] -= ALPHA * dEtotal_dw7 # = 0.511301270
Why[1][1] -= ALPHA * dEtotal_dw8 # = 0.561370121

# BACKWARD PASS - Hidden Layer
# ----------------------------
#
# Now comes the tricky bit. We similarly want to find the derivative of our
# error w.r.t Wxh - that is the weights in our hidden layer. While it appears
# more complicated at first, it's easy to see that as we've already solved some
# part of that problem: we already know how the error changes w.r.t to the net
# input (h) of every output neuron (o). So all that's left to compute is how
# each weight in Wxh affects that net input to o; For example, consider w1 that
# goes from input neuron 1 (x[0]) to hidden neuron 1 (h[0]), and finally to
# every output neuron (o):
#
#   dEtotal   dEtotal   dout_h1   dnet_h1
#   ------- = ------- * ------- * -------
#   dw1       dout_h1   dnet_h1   dw1
#
# Almost exactly like before! Super easy. The math is again the chain rule, and
# the intuition is a few steps again: (a) every change in w1 will create a
# change in the net total input to the hidden neuron h1 (net_h1), that change to
# net_h1 will create some change via the sigmoid function to output of neruon
# h1 (out_h1), and finally that change to the output will create some change to
# the final error due to how it interacts with the following output layer as
# already computed above. In other words, computing (a) and (b) is exactly the
# same as before. While computing (c) relies on informaiton we already computed
# in the top output layer. This is why it's called "Back Propogation" - we're
# going backwards from the derivatives at the top, and use those to compute the
# derivatives at the bottom.
#
# (a) we will start with the derivative of the error function w.r.t changes in
# the output of our hidden neuron h1. This is arguably the only tricky part here
# because while we're only looking at a single weight and a single output, that
# output will be used as input to every neuron in the output layer. So we need
# to sum up these effects. It means that as we change w1, the output of our
# hidden layer will change, and it will then be multiplied by many different
# weights to many different neurons in the next layer - each producing a
# different error. But we've already computed that top layer, so we just need to
# re-use our previous results:
#
#   dEtotal   dEo1      dEo2
#   ------- = ------- + ------- + ....
#   dout_h1   dout_h1   dout_h1
#
# We want to see how our output at neuron h1 affects the total error which is
# the sum of the individual errors:
#
#   dEtotal   dEtotal   dnet_o1
#   ------- = ------- * -------
#   dout_h1   dnet_o1   dout_h1
#
# If we know how our output from h1 affects the net input to o1, and how that
# net input to o1 affects the error - we're done with step (a). We'll only need
# to multiply by (b) and (c) and we're done!
