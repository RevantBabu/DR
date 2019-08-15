import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

dM = np.load("distance_matrix_" + sys.argv[1] + ".npy")
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
plt.savefig("../results/distance_distribution_" + sys.argv[1] + ".svg", format="svg")