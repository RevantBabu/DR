import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#args pos_file spikes_file start end
df1 = pd.read_csv("../data/processed/mj/" + sys.argv[1] + ".csv", header=None)
xs = df1[1].values
ys = df1[2].values

df2 = pd.read_csv("../data/processed/mj/" + sys.argv[2] + ".csv", header=None)
spikes = df2[0].values

spikes = spikes[spikes>float(sys.argv[3])]
spikes = spikes[spikes<float(sys.argv[4])]

n = spikes.size
x = np.zeros(n)
y = np.zeros(n)

for i in range(n):
	x[i] = xs[int((spikes[i]-float(sys.argv[3]))/0.04)]
	y[i] = ys[int((spikes[i]-float(sys.argv[3]))/0.04)]

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(x, y, 'o', markersize=0.4)
plt.savefig(sys.argv[2] + "_pos.png")
