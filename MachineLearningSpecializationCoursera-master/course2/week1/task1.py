import pandas as pd
import numpy as np
import numpy.linalg as ln
from matplotlib import pylab as plt


def out(filename, s):
	with open(filename, 'w') as f:
		f.write(s)


def mserror(y, y_pred):
	return (1 / float(y.size)) * np.sum((y - y_pred) ** 2)


def normal_equation(X, y):
	return ln.solve(X.T.dot(X), X.T.dot(y))


def linear_prediction(X, w):
	return X.dot(w)


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
	mult = (2 * eta / float(X.shape[0])) * (y[train_ind] - X[train_ind, :].dot(w))
	w_new = np.zeros(X.shape[1])
	w_new[0] = (w[0] + mult)
	for i in xrange(1, X.shape[1]):
		w_new[i] = (w[i] + mult * X[train_ind, i])
	return w_new


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
								min_weight_dist=1e-8, seed=42, verbose=False):
	weight_dist = np.inf
	w = w_init
	errors = []
	iter_num = 0
	np.random.seed(seed)

	while weight_dist > min_weight_dist and iter_num < max_iter:
		random_ind = np.random.randint(X.shape[0])
		w_new = stochastic_gradient_step(X, y, w, random_ind, eta)

		weight_dist = np.linalg.norm(w_new - w)
		w = w_new
		errors.append(mserror(y, linear_prediction(X, w)))
		iter_num += 1

	return w, errors


data = pd.read_csv('advertising.csv')

y = data['Sales'].values
X = data.drop('Sales', axis=1).values

means, stds = np.mean(X, axis=0), np.std(X, axis=0)
X = (X - means) / stds
X = np.hstack((np.ones((X.shape[0], 1)), X))

y_pred = np.full(y.shape, np.median(y))
err = mserror(y, y_pred)
res = '%.3f' % err
out('1_1.txt', res)

norm_eq_weights = normal_equation(X, y)
res = '%.3f' % norm_eq_weights.dot(np.mean(X, axis=0))
out('1_2.txt', res)

y_pred = linear_prediction(X, norm_eq_weights)
err = mserror(y, y_pred)
res = '%.3f' % err
out('1_3.txt', res)

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros(X.shape[1]), max_iter=1e5)
plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

print stoch_grad_desc_weights
print stoch_errors_by_iter[-1]

y_pred = linear_prediction(X, stoch_grad_desc_weights)
err = mserror(y, y_pred)
res = '%.3f' % err
print res
out('1_4.txt', res)