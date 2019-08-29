import numpy as np
import matplotlib.pyplot as plt

def vanRossumDistance(u, v, tau):
  componentU = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, u.shape[0]):
      componentU += np.exp(-abs(u[i]-u[j])/tau)

  componentV = 0;
  for i in range(0, v.shape[0]):
    for j in range(0, v.shape[0]):
      componentV += np.exp(-abs(v[i]-v[j])/tau)

  componentUV = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, v.shape[0]):
      componentUV += np.exp(-abs(u[i]-v[j])/tau)

  return componentU + componentV - 2*componentUV


def generateNSamples(N, tau, interval):
  result = []
  for i in range(0, N):
    sample = np.random.poisson(tau, 4*int(interval/tau))
    sample = np.cumsum(sample)
    sample = np.extract(np.less_equal(sample,
                np.full_like(sample, interval)), sample)
    result.append(sample.tolist())
  
  return result

def generateDistanceMatrix(samples, tau):
  n = len(samples)
  result = np.zeros(shape=(n,n))
  print("Distance")
  for i in range(0, n):
    print(i)
    for j in range(i+1, n):
      result[i][j] = vanRossumDistance(np.asarray(samples[i]), np.asarray(samples[j]), tau)
  
  for i in range(0, n):
    for j in range(0, i):
      result[i][j] = result[j][i]

  return result

def generateSimilarityMatrix(dM):
  n = dM.shape[0]
  m = dM.shape[1]
  result = np.zeros(shape=(n,m))
  for i in range(0, n):
    for j in range(0, m):
      if (i==j): result[i][j] = 0
      else: result[i][j] = 1/dM[i][j]
  return result

def thresholdMatrix(sM, topN):
  n = sM.shape[0]
  m = sM.shape[1]
  result = np.zeros(shape=(n,m))

  columnSorted = np.sort(sM,axis=0)
  rowSorted = np.sort(sM,axis=1)

  for i in range(0, n):
    for j in range(0, m):
      columnRank = n-list(columnSorted[:, j]).index(sM[i][j])
      rowRank = m-list(rowSorted[i, :]).index(sM[i][j])
      if (rowRank<=topN or columnRank<=topN):
        result[i][j] = sM[i][j]
  
  return result

def generateLaplacianMatrix(tM):
  n = tM.shape[0]
  m = tM.shape[1]
  result = np.zeros(shape=(n,m))
  columnSums = np.sum(tM, axis = 0)
  print("columnSums:", columnSums)
  for i in range(0, n):
    for j in range(0, m):
      if (i==j): result[i][j] = 1
      else: result[i][j] = -tM[i][j]/columnSums[j]
  return result

np.random.seed(42)

TAU = 15
samples = generateNSamples(5, TAU, 100)
#samples = generateNSamples(200, TAU, 1000)
distanceMatrix = generateDistanceMatrix(samples, TAU)
print("distanceMatrix:")
print(distanceMatrix, "\n")

similarityMatrix = generateSimilarityMatrix(distanceMatrix)
print("similarityMatrix:")
print(similarityMatrix, "\n")

thresholdMatrix = thresholdMatrix(similarityMatrix, 2)
#thresholdMatrix = thresholdMatrix(similarityMatrix, 5)
print("thresholdMatrix:")
print(thresholdMatrix, "\n")


laplacianMatrix = generateLaplacianMatrix(thresholdMatrix)
print("laplacianMatrix:")
print(laplacianMatrix, "\n")

w, v = np.linalg.eig(laplacianMatrix)
idx = w.argsort()[::-1]

wSorted = w[idx]
vSorted = v[idx]
print("eigenValues")
print(wSorted , "\n")

print("eigenVectors")
print(vSorted, "\n")

np.savetxt('eigenValues.txt', wSorted)
np.savetxt('eigenVectors.txt', vSorted)