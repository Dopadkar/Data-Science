import numpy as np
from scipy.linalg import solve
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


def pn(x, w_list):
	"""
	Calculate the value of a polynomial of degree n in a given point.
	Degree is equal to the amount of given coefficients.
	:param x: argument
	:param w_list: list of polynomial coefficients
	:return: function value, i.e. w0 + w1 * x + w2 * x^2 + ... + wn * x ^ n
	"""
	return sum(map(lambda w: w[1] * np.power(x, w[0]), enumerate(w_list)))


def get_matrix_a(args):
	"""
	Fill the matrix of the coefficients for a given point.
	This matrix will be used to solve the system of linear equations.
	It is a square matrix of the following form:
	1	X1		X1^2		...	X1^n
	1	X2		X2^2		...	X2^n
	...
	1	X(n+1)  X(n+1)^2	...	X(n+1)^n
	:param args: list of arguments
	:return: coefficient matrix.
	"""
	return [[np.power(arg, k) for k in xrange(len(args))] for arg in args]


def get_vector_b(args):
	"""
	Fill the vector of the constant terms for the system of linear equations
	It is a vector of the following form
	f(X1)
	f(X2)
	...
	f(X(n+1))
	:param args: list of arguments
	:return: vector of terms
	"""
	return [f(arg) for arg in args]


x0 = np.arange(1, 15.1, 0.1)
y0 = map(f, x0)

x1 = [1, 15]
res1 = solve(get_matrix_a(x1), get_vector_b(x1))
y1 = [pn(x, res1) for x in x0]

x2 = [1, 8, 15]
res2 = solve(get_matrix_a(x2), get_vector_b(x2))
y2 = [pn(x, res2) for x in x0]

x3 = [1, 4, 10, 15]
res3 = solve(get_matrix_a(x3), get_vector_b(x3))
y3 = [pn(x, res3) for x in x0]

x4 = [1, 3, 7, 11, 15]
res4 = solve(get_matrix_a(x4), get_vector_b(x4))
y4 = [pn(x, res4) for x in x0]


plt.plot(x0, y0, 'green')
plt.plot(x0, y1, 'red')
plt.plot(x0, y2, 'blue')
plt.plot(x0, y3, 'black')
plt.plot(x0, y4, 'purple')
plt.show()

res = ' '.join(['%.2f' % c for c in res3])
print res
out('task2_2.txt', res)