import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("../data/processed/hc_13/" + sys.argv[1] + ".csv", header=None)
spikes = df1[0].values
start = int(sys.argv[2])
end = int(sys.argv[3])


n = end-start
d = np.zeros(n)
for spike in spikes:
	d[int(spike) - start] += 1

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.bar(range(n), d, label="Target neuron")
plt.title('Spiking distribution')
plt.xlabel('time')
plt.ylabel('spike count')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig('../results/' + sys.argv[1] + "_dist.png")
plt.savefig('../results/' + sys.argv[1] + "_dist.svg", format="svg")