import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

if sys.argv[1]=="all":
  d1 = np.load("../distances/21/1/distance_matrix_T22_0_1s_20ms.npy")
  d2 = np.load("../distances/21/1/distance_matrix_T22_2_1s_20ms.npy")
  d3 = np.load("../distances/21/1/distance_matrix_T22_3_1s_20ms.npy")
  d4 = np.load("../distances/21/1/distance_matrix_T22_4_1s_20ms.npy")
  d5 = np.load("../distances/21/1/distance_matrix_T26_0_1s_20ms.npy")
  d6 = np.load("../distances/21/1/distance_matrix_T27_0_1s_20ms.npy")
  d7 = np.load("../distances/21/1/distance_matrix_T27_1_1s_20ms.npy")
  d8 = np.load("../distances/21/1/distance_matrix_T27_2_1s_20ms.npy")
  d9 = np.load("../distances/21/1/distance_matrix_T27_3_1s_20ms.npy")
  d10 = np.load("../distances/21/1/distance_matrix_T27_4_1s_20ms.npy")
  d11 = np.load("../distances/21/1/distance_matrix_T27_5_1s_20ms.npy")
  d12 = np.load("../distances/21/1/distance_matrix_T27_6_1s_20ms.npy")
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
  #dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d10**2 + d11**2)
  dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)
else:  
  dM = np.load("../distances/21/1/distance_matrix_" + sys.argv[1] + ".npy")


mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(dM)

coords = results.embedding_

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], coords[:, 1], 'o', label="Target neurons")
plt.title('Isomap Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/' + sys.argv[1] + "_isomap.svg", format="svg")
plt.savefig('../results/21/1/mds2d.png')
