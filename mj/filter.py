import sys
import numpy as np
import pandas as pd


#args spikes_file start end

df = pd.read_csv("../data/processed/mj/" + sys.argv[1] + ".csv", header=None)
spikes = df[0].values

spikes = spikes[spikes>float(sys.argv[2])]
spikes = spikes[spikes<float(sys.argv[3])]

np.savetxt( sys.argv[1] + "_filtered.csv", spikes, delimiter=",", fmt="%.4f")
