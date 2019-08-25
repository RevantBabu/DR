import sys
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

day = sys.argv[1]
epoch = sys.argv[2]
cellNo = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])

df = pd.read_csv("../data/processed/hc_13/" + day + "/" + epoch + "/" + cellNo + ".csv", header=None)

spikes = df[0].values
xs = df[1].values
ys = df[2].values

fct = 1
rng = int((end-start)/fct)
resX = np.zeros(rng)
resY = np.zeros(rng)

for i in range(0, spikes.size):
	resX[int((int(spikes[i])-start)/fct)] = xs[i]
	resY[int((int(spikes[i])-start)/fct)] = ys[i]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# z = np.linspace(0, rng, rng)

# ax.plot(resX, resY, z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
# ax.legend()

# #plt.show()
# plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall3D.png")

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(resX, resY, 'o', label="Target neuron")
plt.title('Position plot')
plt.xlabel('x position')
plt.ylabel('y position')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/filtered_" + cellNo + ".png")


cornerX = []
cornerY = []
ncX = []
ncY = []

for i in range(0,rng):
	if (resX[i]>240 or resX[i]<110 or resY[i]>170 or resY[i]<35):
		cornerX.append(resX[i])
		cornerY.append(resY[i])
	else:
		ncX.append(resX[i])
		ncY.append(resY[i])


print(len(cornerY), len(ncY))
fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(cornerX, cornerY, 'o', label="Target neuron")
plt.title('Position plot')
plt.xlabel('x position')
plt.ylabel('y position')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/filtered_c_" + cellNo + ".png")

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(ncX, ncY, 'o', label="Target neuron")
plt.title('Position plot')
plt.xlabel('x position')
plt.ylabel('y position')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/filtered_nc_" + cellNo + ".png")