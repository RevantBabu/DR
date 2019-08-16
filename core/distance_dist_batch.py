import os
import sys
import numpy as np
import matplotlib.pyplot as plt

path = '..\\distances\\' + sys.argv[1] + '\\' + sys.argv[2] + '\\'
distpath = '..\\plots\\distance_distribution\\' + sys.argv[1] + '\\' + sys.argv[2] + '\\'

def plot_distance_dist(f):
	dM = np.load(path + f)
	n = dM.shape[0]
	d = []

	for i in range(0,n):
		for j in range(0,n):
			d.append(dM[i][j])

	d = np.asarray(d)

	fig = plt.figure(figsize=(9,9))
	ax = plt.subplot(111)
	ax.hist(d, bins=np.arange(d.min(), d.max()+1), align='left')
	plt.title('distance distribution')
	plt.xlabel('distance')
	plt.ylabel('frequency')
	plt.savefig(distpath + f[:-4] + ".png")


files = []
for r, d, f in os.walk(path):
    for fname in f:
    	print("------starting-------: ", fname[:-4])
    	plot_distance_dist(fname)
    	print("------finished-------: ", fname[:-4])