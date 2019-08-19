import sys
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('../data/raw/hc_13/T1rawpos.mat')
days = mat['rawpos']

day = int(sys.argv[1])
epoch = int(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])

d = days[0][day][0][epoch][0][0]['data']
spikes = d[:, 0]
xs = d[:, 1]
ys = d[:, 2]

fct = 1
rng = int((end-start)/fct)
resX = np.zeros(rng)
resY = np.zeros(rng)

for i in range(0, spikes.size):
	resX[int((int(spikes[i])-start)/fct)] = xs[i]
	resY[int((int(spikes[i])-start)/fct)] = ys[i]


result = np.zeros(shape=(rng,rng))
for i in range(0, rng):
	print(i)
	for j in range(0, rng):
		result[i][j] = np.sqrt((resX[i]-resX[j])**2 + (resY[i]-resY[j])**2)


np.save("../distances/" + sys.argv[1] + "/" + sys.argv[2] + "/distance_matrix_spatial.npy", result)
print(result.shape)