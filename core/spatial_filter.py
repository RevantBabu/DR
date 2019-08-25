import sys
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


mat = scipy.io.loadmat('../data/raw/hc_13/T1rawpos.mat')
days = mat['rawpos']

day = int(sys.argv[1])
epoch = int(sys.argv[2])
cellNo = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])

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
cornerIdx1= np.zeros(rng)
cornerIdx2 = np.zeros(rng)
cornerIdx3 = np.zeros(rng)
cornerIdx4 = np.zeros(rng)
ncX = []
ncY = []


for i in range(0,rng):
	if (resX[i]>240 or resX[i]<115 or resY[i]>170 or resY[i]<35):
		cornerX.append(resX[i])
		cornerY.append(resY[i])
		if (resX[i]<115 and resY[i]>100):
			cornerIdx1[i] = -1
		elif (resX[i]<115 and resY[i]<100):
			cornerIdx2[i] = -1
		elif (resX[i]>240 and resY[i]<100):
			cornerIdx3[i] = -1
		elif (resX[i]>240 and resY[i]>100):
			cornerIdx4[i] = -1
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

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(cornerIdx1[200:245], 'o', label="Corner1")
ax.plot(cornerIdx2[200:245], 'o', label="Corner2")
ax.plot(cornerIdx3[200:245], 'o', label="Corner3")
ax.plot(cornerIdx4[200:245], 'o', label="Corner4")
plt.title('Position plot')
plt.xlabel('x position')
plt.ylabel('y position')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/filtered_ctime_" + cellNo + ".png")