import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from matplotlib import pylab as plt


def out(filename, s):
	"""
	Write given string to a file
	:param filename: file name
	:param s: string to write
	:return: None
	"""
	f = open(filename, 'w')
	f.write(s)
	f.close()


def f(x):
	"""
	Calculate target function in a given point.
	:param x: argument
	:return: function value
	"""
	return np.sin(x / 5.0) * np.exp(x / 10.0) + 5 * np.exp(-x / 2.0)


def h(x):
	return int(f(x))

x0 = np.arange(1, 30.1, 0.1)
y0 = map(f, x0)

ext1 = minimize(f, [2], method='BFGS')
ext2 = minimize(f, [30], method='BFGS')
res = '%.2f %.2f' % (ext1.fun, ext2.fun)
print res
out('3_1.txt', res)

ext3 = differential_evolution(f, [(1, 30)])
res = '%.2f' % ext3.fun
print res
out('3_2.txt', res)

x1 = np.arange(1, 30.1, 0.1)
y1 = map(h, x1)

ext4 = minimize(h, [30], method='BFGS')
ext5 = differential_evolution(h, [(1, 30)])
res = '%.2f %.2f' % (ext4.fun, ext5.fun)
print res
out('3_3.txt', res)

plt.plot(x0, y0, 'green')
plt.plot(x1, y1, 'red')
plt.plot([ext1.x], [ext1.fun], marker='o')
plt.plot([ext2.x], [ext2.fun], marker='o')
plt.plot([ext3.x], [ext3.fun], marker='x')
plt.plot([ext4.x], [ext4.fun], marker='v')
plt.plot([ext5.x], [ext5.fun], marker='*')
plt.show()