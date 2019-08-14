import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generateSimilarityMatrix(dM):
  n = dM.shape[0]
  m = dM.shape[1]
  result = np.zeros(shape=(n,m))
  for i in range(0, n):
    for j in range(0, m):
      if (i==j): result[i][j] = 0
      elif (dM[i][j]==0): result[i][j] = 0#1 #because some slots have 0 spikes?
      # is default 1 fine? if 0 used, gives imaginary eVs
      else: result[i][j] = 1/dM[i][j]
  return result

def thresholdMatrix(sM, topN):
  n = sM.shape[0]
  m = sM.shape[1]
  result = np.zeros(shape=(n,m))

  columnSorted = np.sort(sM,axis=0)
  rowSorted = np.sort(sM,axis=1)

  rowTopN = np.zeros(n)
  columnTopN = np.zeros(n)

  for i in range(0, n):
    rowTopN[i] = rowSorted[i, :][n-topN]
    columnTopN[i] = columnSorted[:, i][n-topN]


  for i in range(0, n):
    for j in range(0, m):
      if (sM[i][j]>=rowTopN[i] or sM[i][j]>=columnTopN[j]):
        result[i][j] = sM[i][j]

  return result

def generateLaplacianMatrix(tM):
  n = tM.shape[0]
  m = tM.shape[1]
  result = np.zeros(shape=(n,m))
  columnSums = np.sum(tM, axis = 0)
  for i in range(0, n):
    for j in range(0, m):
      if (i==j): result[i][j] = 1
      else: result[i][j] = -tM[i][j]/columnSums[j]
  return result

if sys.argv[1]=="all":
  d1 = np.load("distance_matrix_T22_0_1s_20ms.npy")
  d2 = np.load("distance_matrix_T22_2_1s_20ms.npy")
  d3 = np.load("distance_matrix_T22_3_1s_20ms.npy")
  d4 = np.load("distance_matrix_T22_4_1s_20ms.npy")
  d5 = np.load("distance_matrix_T26_0_1s_20ms.npy")
  d6 = np.load("distance_matrix_T27_1_1s_20ms.npy")
  d7 = np.load("distance_matrix_T27_2_1s_20ms.npy")
  d8 = np.load("distance_matrix_T27_3_1s_20ms.npy")
  d9 = np.load("distance_matrix_T27_4_1s_20ms.npy")
  d10 = np.load("distance_matrix_T27_5_1s_20ms.npy")
  d11 = np.load("distance_matrix_T27_6_1s_20ms.npy")
  dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2)
elif sys.argv[1]=="all_sparse":
  d1 = np.load("distance_matrix_T22_0_1s_20ms.npy")
  d2 = np.load("distance_matrix_T22_2_1s_20ms.npy")
  d3 = np.load("distance_matrix_T22_3_1s_20ms.npy")
  d6 = np.load("distance_matrix_T27_1_1s_20ms.npy")
  d7 = np.load("distance_matrix_T27_2_1s_20ms.npy")
  d8 = np.load("distance_matrix_T27_3_1s_20ms.npy")
  d9 = np.load("distance_matrix_T27_4_1s_20ms.npy")
  d10 = np.load("distance_matrix_T27_5_1s_20ms.npy")
  dM = np.sqrt(d1**2 + d2**2 + d3**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2)
else:  
  dM = np.load("distance_matrix_" + sys.argv[1] + ".npy")

sM = generateSimilarityMatrix(dM)
tM = thresholdMatrix(sM, 10)
lM = generateLaplacianMatrix(tM)


#print(sM)
#print(tM)
#print(lM)

w, v = np.linalg.eig(lM)
#idx = w.argsort()[::-1]
idx = w.argsort()

wSorted = w[idx]
vSorted = v[idx]
print("eigenValues")
#print(wSorted , "\n")

print("eigenVectors")
#print(vSorted, "\n")

np.savetxt('../results/' + sys.argv[1] + '_eigenValues.txt', wSorted)
np.savetxt('../results/' + sys.argv[1] + '_eigenVectors.txt', vSorted)


fig = plt.figure(figsize=(9,4))
ax = plt.subplot(111)
ax.plot(vSorted[1], vSorted[2], 'o', label="Target neurons")#, markersize=0.4)
plt.title('Diffusion map, leading dimension')
plt.xlabel('time')
plt.ylabel('Leading eV values')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig('../results/' + sys.argv[1] + "_leadingVectors.svg", format="svg")
