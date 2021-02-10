import pandas as pd
import numpy as np
import sys
# import matplotlib.pylab as plt


path = sys.argv[1]
col = sys.argv[2]
bin = sys.argv[3]

original_data = pd.read_csv(path, sep=" ", index_col=None, header=None)
original_data.columns=["Position", "Angle", "Output"]
original_data.hist(bins=bin)
target_col = original_data[col]


boundaries = np.linspace(start=target_col.min(), stop=target_col.max(), num=bin)

bin_datas = [original_data[(target_col >= down) & (target_col <= up)] for (down, up) in zip(boundaries[:-1], boundaries[1:])]

no_replica_bin_size = int(np.median([bin_data.shape[0] for bin_data in bin_datas]))
replica_bin_size = original_data.shape[0]//bin

no_replica_bins = [bin_data.sample(n=no_replica_bin_size, replace=False) if bin_data.shape[0] > no_replica_bin_size else bin_data.sample(frac=1, replace=False) for bin_data in bin_datas]
no_replica_data = pd.concat(no_replica_bins, ignore_index = True)
no_replica_data.to_csv(path.rstrip(".dat")+f"_{col}_noreplica.dat", sep=" ", index=False, header=None)

replica_bins = [bin_data.sample(n=replica_bin_size, replace=True)  for bin_data in bin_datas if bin_data.shape[0] > 0]
replica_data = pd.concat(replica_bins, ignore_index = True)
replica_data.to_csv(path.rstrip(".dat")+f"_{col}_replica.dat", sep=" ", index=False, header=None)

# no_replica_data.hist(bins= bin*2)
# replica_data.hist(bins= bin*2)
# plt.show()
