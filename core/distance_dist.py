import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

day = sys.argv[1]
epoch = sys.argv[2]
#dM = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_spatial.npy")
d1 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T22_0_1s_20ms.npy")
d2 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T22_2_1s_20ms.npy")
d3 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T22_3_1s_20ms.npy")
d4 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T22_4_1s_20ms.npy")
d5 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T26_0_1s_20ms.npy")
d6 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_0_1s_20ms.npy")
d7 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_1_1s_20ms.npy")
d8 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_2_1s_20ms.npy")
d9 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_3_1s_20ms.npy")
d10 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_4_1s_20ms.npy")
d11 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_5_1s_20ms.npy")
d12 = np.load("../distances/" + day + "/" + epoch + "/distance_matrix_T27_6_1s_20ms.npy")

d1 = d1/np.amax(d1)
d2 = d2/np.amax(d2)
d3 = d3/np.amax(d3)
d4 = d4/np.amax(d4)
d5 = d5/np.amax(d5)
d6 = d6/np.amax(d6)
d7 = d7/np.amax(d7)
d8 = d8/np.amax(d8)
d9 = d9/np.amax(d9)
d10 = d10/np.amax(d10)
d11 = d11/np.amax(d11)
d12 = d12/np.amax(d12)
dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)

n = dM.shape[0]
d = []

for i in range(0,n):
	for j in range(0,n):
		d.append(dM[i][j])

d = np.asarray(d)*20

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.hist(d, bins=np.arange(d.min(), d.max()+1), align='left')
plt.title('distance distribution: all normalized')
plt.xlabel('distance (cm)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-4)))
plt.ylabel('frequency ($10^4$)')
plt.savefig("../plots/distance_distribution/" + day + "/" + epoch + "/all_normalized.png")
plt.savefig("../plots/distance_distribution/" + day + "/" + epoch + "/all_normalized.pdf")