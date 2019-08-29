import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

day = sys.argv[1]
epoch = sys.argv[2]
cellNo = sys.argv[3]


df1 = pd.read_csv("../data/processed/hc_13/" + day + "/" + epoch + "/" + cellNo + ".csv", header=None)
spikes = df1[0].values
start = int(sys.argv[4])
end = int(sys.argv[5])

n = end-start
d = np.zeros(n)
for spike in spikes:
	d[int(spike) - start] += 1

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.bar(range(n), d, label="Target neuron")
plt.title('spiking frequency : T27-3')
plt.xlabel('time (s)')
plt.ylabel('spike count')
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/frequency_plot/" + day + "/" + epoch + "/" + cellNo + ".png")
plt.savefig("../plots/frequency_plot/" + day + "/" + epoch + "/" + cellNo + ".pdf")