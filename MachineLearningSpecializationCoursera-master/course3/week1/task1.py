import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from numpy.linalg import norm


def prepare():
	input = open('checkins.dat', 'r')
	output = open('checkins_prepared.csv', 'w')
	for line in input:
		parts = [s for s in line.split('|')]
		if len(parts) == 6 and len(parts[3].strip()) != 0 and len(parts[4].strip()) != 0:
			#output.write(','.join(parts))
			output.write(parts[3] + ', ' +  parts[4] + '\n')
	input.close()
	output.close()


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


#prepare()
# df = pd.read_csv("checkins.dat", sep='|', skiprows=2,
#                   names=['id', 'user_id', 'venue_id', 'latitude', 'longitude', 'created_at'])
# df = df[pd.notnull(df['latitude']) & pd.notnull(df['longitude'])]

df = pd.read_csv('checkins_prepared.csv')
df = df.ix[np.random.choice(df.index, 100000, replace=False)]
# print df.head()
# print len(df.index)

ms = MeanShift(bandwidth=0.1)
ms.fit(df)

print len(ms.cluster_centers_)
clusters = {k: 0 for k in range(len(ms.cluster_centers_))}

for k in ms.labels_:
	clusters[k] += 1

selected_clusters = [k for k in clusters.keys() if clusters[k] > 15]
# print clusters
# print selected_clusters
coords = [ms.cluster_centers_[i] for i in selected_clusters]
# print coords

offices = [np.array([33.751277, -118.188740]), np.array([25.867736, -80.324116]),
		   np.array([51.503016, -0.075479]), np.array([52.378894, 4.885084]),
		   np.array([39.366487, 117.036146]), np.array([-33.868457, 151.205134])]


def get_min_dist_office(coord):
	min_dist = 1000000
	for office in offices:
		dist = norm(coord - office)
		if dist < min_dist:
			min_dist = dist
	return min_dist


min_dist = 10000000
min_coord = None
for coord in coords:
	dist = get_min_dist_office(coord)
	if dist < min_dist:
		min_dist = dist
		min_coord = coord


print min_coord, min_dist
res = '%f %f' % (min_coord[0], min_coord[1])
out('task1.txt', res)
