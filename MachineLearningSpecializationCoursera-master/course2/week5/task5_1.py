from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
import heapq
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


digits = load_digits()
cancer = load_breast_cancer()
digits_scores = []
cancer_scores = []

cls = BernoulliNB()
digits_scores.append(cross_val_score(cls, digits.data, digits.target).mean())
cancer_scores.append(cross_val_score(cls, cancer.data, cancer.target).mean())

cls = MultinomialNB()
digits_scores.append(cross_val_score(cls, digits.data, digits.target).mean())
cancer_scores.append(cross_val_score(cls, cancer.data, cancer.target).mean())

cls = GaussianNB()
digits_scores.append(cross_val_score(cls, digits.data, digits.target).mean())
cancer_scores.append(cross_val_score(cls, cancer.data, cancer.target).mean())

print digits_scores
print cancer_scores

#out('5_1.txt', str(max(cancer_scores)))
out('5_1.txt', str(heapq.nlargest(2, (cancer_scores))[1]))

out('5_2.txt', str(max(digits_scores)))
out('5_3.txt', '3 4')