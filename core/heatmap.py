import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = np.load("distance_matrix_T26_0_1s_20ms.npy")
#df = np.load("distance_matrix_spatial.npy")
df = 1/(df + 10)

sns.heatmap((df-df.mean())/df.std(), cmap="YlGnBu")

plt.show()
