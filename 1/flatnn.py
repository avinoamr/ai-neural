# Flat Neural Network
# y = Wx + b
import random

data = open('input.txt', 'r').read()
vocab = list(set(data))
vocab_to_idx = { c: i for i, c in enumerate(vocab) }

print 'data has %d characters, %d unique' % (len(data), len(vocab))

char_to_next = { c: {} for c in vocab }

for i in xrange(len(data) - 1):
    cur = data[i]
    nxt = data[i + 1]

    char_to_next[cur].setdefault(nxt, 0)
    char_to_next[cur][nxt] += 1

def sample(c):
    candidates = char_to_next[c]
    choices = []
    for n, c in enumerate(candidates):
        choices += [c] * n

    if len(choices) == 0:
        return ' '
    # print choices
    return random.choice(choices)


# print char_to_next
out = ['I']
for i in range(1000):
    nxt = out[-1]
    v = sample(nxt)
    out.append(v)

# print out
print ''.join(out)
