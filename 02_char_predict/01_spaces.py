# At this entire section we'll attempt to evolve our algorithm to support the
# case of character-level language modelling. The core goal of our eventual code
# will be to produce text that looks similar to text given as input, simply by
# choosing one character at a time.[1]
#
# The first step will be to use the classifier we built before to predict the
# likelihood of the next character to be a space, given the current character as
# input
#
# [1] Inspired by: https://github.com/karpathy/char-rnn
import csv
import random
import numpy as np

STEP = 1.
EPOCHS = 300

# we'll start with super predictable text, one where is upper-case char is
# followed by the same lower case char, which is then followed by a space. The
# only true randomness occurs after the space where a new upper-chase char is
# introduced. Thus this text can be predicted for 66% of its content.
data = "Ee Mm Ll Ee Mm Tt Ww Vv Vv Hh Hh Kk Uu Qq Zz Nn Dd Bb Ii Mm " +
    "Dd Bb Yy Oo Uu Ee Uu Yy Aa Gg Tt Gg Bb Uu Nn Rr Nn Bb Cc Oo " +
    "Dd Ii Tt Hh Zz Xx Xx Gg Zz Hh Oo Pp Hh Nn Uu Ww Ff Ff Tt Yy " +
    "Kk Aa Hh Tt Bb Qq Uu Hh Hh Cc Vv Gg Ss Jj Dd Dd Ll Kk Oo Ww " +
    "Rr Vv Hh Dd Nn Ii Vv Zz Cc Kk Ss Uu Ii Qq Hh Hh Cc Nn Zz Mm "

# we'll use one-of-k encoding for each input character. One neuron per character
# in the alphabet of the provided text.
ALPHABET = list(set(data)) # list of all unique characters in the input
BATCHSIZE = len(data) / 100 # 10% per batch
N = 1 + len(ALPHABET) # one neuron per character, plus bias

# data prepation. for each iteration, we'll need the current (input) and next
# (target) characters. We can do it by iterating over the indices, but it will
# keep the code cleaner if we had all of these (current, next) tuples preset.
# We'll do it here by first shifting the data by 1 character, and then zipping
# to original and the shifted data together into a single list of tuples
data = zip(data, data[1:] + " ") # extra char to keep the same length

# given a single character, return all of the alphabet neurons with the right
# one lit up
def one_of_k(c):
    x = np.zeros(N)
    x[0] = 1. # bias input fixed to 1.

    idx = ALPHABET.index(c)
    x[idx] = 1. # current character
    return x

w = np.random.random(N) - .5
for i in xrange(EPOCHS):
    remaining = data
    while len(remaining) > 0:
        minib, remaining = remaining[:BATCHSIZE], remaining[BATCHSIZE:]

        l = 0
        dw = 0
        accuracy = 0
        for d in minib:
            x = one_of_k(d[0]) # encode input
            y = sum(x * w)
            t = 1. if d[1] == "\n" else 0.

            l += (y - t) ** 2 / 2
            dw += (y - t) * x

            accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0

        dw /= len(minib)
        w += STEP * -dw # mini-batch update

        l /= len(data)
        print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))







    # start = 0
    #
    #
    # end = start + BATCHSIZE


    # while len(remaining) > 0:
    #
    #
    #     minib, remaining = remaining[:BATCHSIZE], remaining[BATCHSIZE:]
    #     dw = 0
    #
    #     for c in len(minib):
    #         x = encode(d)
    #
    #
    #
    # break

    # remaining = data
    # while len(remaining) > 0:
#         minib, remaining = remaining[:BATCHSIZE], remaining[BATCHSIZE:]
#         dw = 0
#         for d in minib:
#             x = encode(d) # encode the input features into multiple 1-of-key's
#             y = sum(x * w) # compute the prediction
#             t = float(d["Survived"]) # encode the target correct output
#
#             l += (y - t) ** 2 / 2
#             dw += (y - t) * x
#
#             accuracy += 1 if round(np.clip(y, 0, 1)) == t else 0
#
#         dw /= len(minib)
#         w += STEP * -dw # mini-batch update
#
#     l /= len(data)
#     print "%s: LOSS = %s; ACCURACY = %d of %d" % (i, l, accuracy, len(data))
