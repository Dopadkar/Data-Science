from sklearn.datasets import load_digits
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pylab as plt
import numpy as np

def show_digit(digits, n):
	print digits.data[n]
	print digits.target[n]
	plt.gray()
	plt.matshow(digits.images[n])
	plt.show()


def make_plot(x, y):
	plt.plot(x, y)
	plt.show()


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

digits = load_digits()

# print digits.data.shape
# print digits.target.shape
# show_digit(digits, 2)

X = digits.data
y = digits.target
tree = DecisionTreeClassifier()

res = cross_val_score(tree, X, y, cv=10).mean()
print res
out('4_1.txt', str(res))

bagging = BaggingClassifier(n_estimators=100)
res = cross_val_score(bagging, X, y, cv=10).mean()
print res
out('4_2.txt', str(res))

d = X.shape[1]
bagging = BaggingClassifier(n_estimators=100, max_features=int(np.sqrt(d)))
res = cross_val_score(bagging, X, y, cv=10).mean()
print res
out('4_3.txt', str(res))

tree = DecisionTreeClassifier(max_features=int(np.sqrt(d)))
bagging = BaggingClassifier(base_estimator=tree, n_estimators=100)
res = cross_val_score(bagging, X, y, cv=10).mean()
print res
out('4_4.txt', str(res))

rf = RandomForestClassifier()

grid_1 = {'n_estimators': np.arange(10, 140, 20)}
grid_2 = {'max_features': np.arange(0.05, 1, 0.05)}
grid_3 = {'max_depth': np.arange(2, 20, 2)}

gs1 = GridSearchCV(rf, grid_1, scoring='accuracy', cv=10)
gs2 = GridSearchCV(rf, grid_2, scoring='accuracy', cv=10)
gs3 = GridSearchCV(rf, grid_3, scoring='accuracy', cv=10)

gs1.fit(X, y)
gs2.fit(X, y)
gs3.fit(X, y)

x1 = [s.parameters['n_estimators'] for s in gs1.grid_scores_]
y1 = [s.mean_validation_score for s in gs1.grid_scores_]
make_plot(x1, y1)

x2 = [s.parameters['max_features'] for s in gs2.grid_scores_]
y2 = [s.mean_validation_score for s in gs2.grid_scores_]
make_plot(x2, y2)

x3 = [s.parameters['max_depth'] for s in gs3.grid_scores_]
y3 = [s.mean_validation_score for s in gs3.grid_scores_]
make_plot(x3, y3)