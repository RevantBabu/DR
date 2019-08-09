import numpy as np

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
