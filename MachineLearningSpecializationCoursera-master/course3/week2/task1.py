import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score as cv_score
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import RandomizedPCA

def plot_scores(d_scores):
    n_components = np.arange(1,d_scores.size+1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()

def write_answer_1(optimal_d):
    with open("pca_answer1.txt", "w") as fout:
        fout.write(str(optimal_d))

def plot_variances(d_variances):
    n_components = np.arange(1,d_variances.size+1)
    plt.plot(n_components, d_variances, 'b', label='Component variances')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.legend(loc='upper right')
    plt.show()

def write_answer_2(optimal_d):
    with open("pca_answer2.txt", "w") as fout:
        fout.write(str(optimal_d))

def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()

def write_answer_3(list_pc1, list_pc2):
    with open("pca_answer3.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc1]))
        fout.write(" ")
        fout.write(" ".join([str(num) for num in list_pc2]))

def write_answer_4(list_pc):
    with open("pca_answer4.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc]))

def task1():
	data = pd.read_csv('data_task1.csv')
	print data.shape[1]

	d_scores = []
	for d in xrange(1, data.shape[1] + 1):
		model = PCA(n_components=d)
		model.fit(data)
		scores = cv_score(model, data)
		cur_score = scores.mean()
		d_scores.append(cur_score)

	res = np.argmax(d_scores) + 1
	write_answer_1(res)
	print res
	plot_scores(np.array(d_scores))

def task2():
	data = pd.read_csv('data_task2.csv')
	print data.shape
 	D = data.shape[1]
	model = PCA(n_components=D)
	model.fit(data)
	data_new = model.transform(data)
	d_variances = data_new.var(axis=0)
	#d_variances = sorted(d_variances, reverse=True)
	diffs = [d_variances[i] - d_variances[i+1] for i in xrange(len(d_variances) - 1)]
	res = np.argmax(diffs) + 1
	plot_variances(d_variances)
	print res
	write_answer_2(res)

def task3():
	iris = datasets.load_iris()
	data = iris.data
	target = iris.target
	target_names = iris.target_names
	feature_names = iris.feature_names
	model = PCA(n_components=data.shape[1])
	model.fit(data, target)
	data_new = model.transform(data)
	plot_iris(data_new, target, target_names)
	corrs1 = []
	corrs2 = []
	x1_new = data_new[:, 0] - data_new[:, 0].mean()
	x2_new = data_new[:, 1] - data_new[:, 1].mean()
	for i in xrange(data.shape[1]):
		x_i = data[:, i] - data[:, i].mean()
		corrs1.append(np.abs(pearsonr(x_i, x1_new)[0]))
		corrs2.append(np.abs(pearsonr(x_i, x2_new)[0]))

	list_pc1 = []
	list_pc2 = []
	for i in xrange(len(corrs1)):
		if corrs1[i] > corrs2[i]:
			list_pc1.append(i+1)
		else:
			list_pc2.append(i+1)
	write_answer_3(list_pc1, list_pc2)


def cos(x, y, xk, yk):
	s = 0
	for i in xrange(len(x)):
		s += np.power((x[i] - y[i]), 2)
	return np.power((xk - yk), 2) / s


def task4():
	data = fetch_olivetti_faces(shuffle=True, random_state=0).data
	image_shape = (64, 64)
	model = RandomizedPCA(n_components=10)
	model.fit(data)
	data_new = model.transform(data)
	mean_components = [data_new[:, i].mean() for i in xrange(data_new.shape[1])]
	influence = np.zeros((data_new.shape[0], data_new.shape[1]))
	for i in xrange(data_new.shape[0]):
		for j in xrange(data_new.shape[1]):
			influence[i, j] = cos(data_new[i, :], mean_components, np.abs(data_new[i, j]), mean_components[j])
	res = []
	for i in xrange(influence.shape[1]):
		res.append(np.argmax(influence[:, i]))
	print res
	write_answer_4(res)
	# for i in res:
	# 	plt.imshow(data[i, :].reshape(image_shape))
	# 	plt.show()

task4()