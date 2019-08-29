import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

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

fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, rng, rng)

ax.plot(resX, resY, z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
ax.legend()

#plt.show()
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall3D.png")

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(resX, resY, 'o', label="Target neuron")
plt.title('spiking-position plot (1Hz)')
plt.xlabel('x position (cm)')
plt.ylabel('y position (cm)')
plt.xlim(80, 270)
#ax.legend(loc='upper left', bbox_to_anchor=(0, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall.pdf")
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall.png")

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(xs, ys, 'o', label="Target neuron")
plt.title('spiking-position plot (25Hz)')
plt.xlabel('x position (cm)')
plt.ylabel('y position (cm)')
plt.xlim(80, 270)
#ax.legend(loc='upper left', bbox_to_anchor=(0, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall_hf.pdf")
plt.savefig("../plots/position_plot/" + sys.argv[1] + "/" + sys.argv[2] + "/overall_hf.png")