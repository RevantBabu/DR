import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/hc_13/T26_0.csv", header=None)
spikes = df[0].values
xs = df[1].values
ys = df[2].values

fct = 1
rng = int((1595-790)/fct)
resX = np.zeros(rng)
resY = np.zeros(rng)

for i in range(0, spikes.size):
	resX[int((int(spikes[i])-790)/fct)] = xs[i]
	resY[int((int(spikes[i])-790)/fct)] = ys[i]


result = np.zeros(shape=(rng,rng))
for i in range(0, rng):
	print(i)
	for j in range(0, rng):
		result[i][j] = np.sqrt((resX[i]-resX[j])**2 + (resY[i]-resY[j])**2)

np.save("distance_matrix_spatial.npy", result)
print(result.shape)