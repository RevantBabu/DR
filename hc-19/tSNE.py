import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

e = np.load("ephys.npy").item()
a =  e['TT1']['spike_amplitudes']

print(a.shape)
#a_embedded = TSNE(n_components=3).fit_transform(a)
a_embedded = TSNE(n_components=2).fit_transform(a)

fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
#ax.scatter(a_embedded[:,0], a_embedded[:,1], a_embedded[:,2])
ax.scatter(a_embedded[:,0], a_embedded[:,1])
plt.title('2d clustering tSNE')
# ax.set_xlabel('dimension 1')
# ax.set_ylabel('dimension 2')
# ax.set_zlabel('dimension 3')
# plt.savefig("tSNE_3d.png")
# plt.savefig("tSNE_3d.pdf")
plt.savefig("tSNE_2d.png")
plt.savefig("tSNE_2d.pdf")