import csv
import numpy as np

e = np.load("ephys.npy").item()
a =  e['TT1']['spike_amplitudes']

with open("amplitudes_TT1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(a)