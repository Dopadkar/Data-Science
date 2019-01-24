import re
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

lines = []
words = set()
with open('sentences.txt', 'r') as f:
	for l in f.readlines():
		if len(l) > 0:
			line = l.lower()
			cur_words = filter(lambda x: len(x) > 0, re.split('[^a-z]', line))
			lines.append((line, cur_words))
			words.update(cur_words)

ind = {word: i for (i, word) in enumerate(words)}
m = np.zeros((len(lines), len(words)))

for i, line in enumerate(lines):
	cnt = Counter(line[1])
	for word in cnt.keys():
		m[i, ind[word]] = cnt[word]

distances = []
for i, line in enumerate(lines):
	if i == 0:
		continue
	distances.append((i, cosine(m[0], m[i])))

distances.sort(key=lambda k: k[1])
res = '%d %d' % (distances[0][0], distances[1][0])
print res
# for d in distances:
# 	print lines[d[0]][0]
out('task2_1.txt', res)
