from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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


def predict_x(X_train, y_train, x):
	min_dist = 1e10
	min_ind = 0
	for index, val in enumerate(X_train):
		dist = euclidean(val, x)
		if dist < min_dist:
			min_dist = dist
			min_ind = index
	return y_train[min_ind]


def predict(X_train, y_train, X_test):
	res = np.array([])
	for x in X_test:
		res = np.append(res, predict_x(X_train,  y_train, x))
	return res


digits = load_digits()
border = digits.data.shape[0] * 0.75
X_train = digits.data[:border, :]
X_test = digits.data[border:, :]
y_train = digits.target[:border]
y_test = digits.target[border:]


y_pred = predict(X_train, y_train, X_test)
error_rate = 1 - accuracy_score(y_test, y_pred)
print error_rate
out('5_4.txt', str(error_rate))

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
error_rate = 1 - accuracy_score(y_test, y_pred)
print error_rate
out('5_5.txt', str(error_rate))