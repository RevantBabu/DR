import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

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




#plt.title('Position plot')
#plt.xlabel('x position')
#plt.ylabel('y position')

# fig = plt.figure(figsize=(9,9))
# ax = plt.subplot(311)
# ax.plot(resX, resY, 'o', label="Target neuron", markersize=1)
# ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)

# ax = plt.subplot(312)
# ax.plot(resY, 'o', label="Target neuron", markersize=1)
# ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)

# ax = plt.subplot(313)
# ax.plot(resY, 'o', label="Target neuron", markersize=1)
# ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)

# plt.savefig("position_plot_spacetime.svg", format="svg")



fig = plt.figure()
ax = fig.gca(projection='3d')

# Prepare arrays x, y, z
N = 805
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(0, N, N)

ax.plot(resX, resY, z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
ax.legend()


plt.savefig("position_plot_spacetime3D.svg", format="svg")