from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pylab as plt
import numpy as np


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


def gbm_predict(X):
	if len(base_algorithms_list) == 0:
		return 0
	return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)]) for x in
			X]


def make_plot(x, y):
	plt.plot(x, y)
	plt.show()


boston = load_boston()
border = boston.data.shape[0] * 0.75
X_train = boston.data[:border, :]
X_test = boston.data[border:, :]
y_train = boston.target[:border]
y_test = boston.target[border:]

base_algorithms_list = []
coefficients_list = []

N = 50
coeff = 0.9

for i in xrange(N):
	tree = DecisionTreeRegressor(max_depth=5, random_state=42)
	coeff = 0.9 / (1.0 + i)
	tree.fit(X_train, y_train - gbm_predict(X_train))
	base_algorithms_list.append(tree)
	coefficients_list.append(coeff)

res = np.sqrt(mean_squared_error(y_test, gbm_predict(X_test)))
print res
#out('4_2_2.txt', str(res))
out('4_2_3.txt', str(res))

n_trees = [1] + range(10, 105, 5)
xgb_scoring = []
for n_tree in n_trees:
	estimator = xgb.XGBRegressor(n_estimators=n_tree)
	estimator.fit(X_train, y_train)
	score = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
	xgb_scoring.append(score)

depths = range(1, 20, 1)
xgb_scoring2 = []
for depth in depths:
	estimator = xgb.XGBRegressor(max_depth=depth)
	estimator.fit(X_train, y_train)
	score = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
	xgb_scoring2.append(score)


f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(n_trees, xgb_scoring)
ax1.set_title('MSE for various n_estimators')
ax2.plot(depths, xgb_scoring2)
ax2.set_title('MSE for various max_depth')
plt.show()

out('4_2_4.txt', '2 3')

estimator = LinearRegression()
estimator.fit(X_train, y_train)
res = np.sqrt(mean_squared_error(y_test, estimator.predict(X_test)))
print res
out('4_2_5.txt', str(res))