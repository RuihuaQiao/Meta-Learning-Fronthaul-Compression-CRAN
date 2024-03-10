import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


M = 8
B = 7
Ball = 19
Nc = 2
K = 2
test_size = 2000

# method = "antenna_reduction"
# antenna_reduction = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, K, Nc, B, K))['sorted_rate']

method = "SingleCell"
SingleCell = loadmat(method+'_cell{}_M{}_Nc{}_B{}.mat'.format(Ball, M, Nc, B))['sorted_rate']

method = "EVD"
EVD = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K))['sorted_rate']

method = "EVD_beta"
EVD_beta = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K))['sorted_rate']

method = "DNN"
DNN = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K))['sorted_rate']

method = "GlobalGD"
globalGD_randini = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K))['sorted_rate']

method = "GRU_T4"
GRU_T4 = loadmat(method+'_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K))['sorted_rate']


font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)
plt.figure(figsize=(7, 5))

# plt.plot(antenna_reduction.squeeze(), np.arange(test_size) / test_size, linestyle="dotted", linewidth=2, color="tab:gray", label="Antenna_reduction")
plt.plot(SingleCell.squeeze(), np.arange(test_size) / test_size, linestyle="dotted", linewidth=2.5, color="tab:gray", label="Single-cell processing")
plt.plot(EVD.squeeze(), np.arange(test_size) / test_size, linestyle="dashdot", linewidth=2.5, color="tab:green", label="EVD w/ local CSI")
plt.plot(EVD_beta.squeeze(), np.arange(test_size) / test_size, linestyle="dashed", linewidth=2.5, color="tab:cyan", label="Modified EVD w/ local CSI")
plt.plot(DNN.squeeze(), np.arange(test_size) / test_size, linestyle="solid", linewidth=2.5, color="tab:orange", label="DNN w/ local CSI")
plt.plot(GRU_T4.squeeze(), np.arange(test_size) / test_size, linestyle=(0, (5, 3)), linewidth=2.5, color="tab:blue", label="DNN w/ local CSI + GRU (T=4)")
plt.plot(globalGD_randini.squeeze(), np.arange(test_size) / test_size, linestyle=(0, (5, 1, 1, 1, 1, 1)), linewidth=2.5, color="tab:purple", label="GD w/ global CSI")

# plt.xlim(5, 16)
# plt.xlim(1, 4.5)
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.05), borderaxespad=0)
plt.xlabel("Average per-user rate (bps/Hz)", fontsize=14)
plt.ylabel("Cumulative distribution function (CDF)", fontsize=14)
plt.grid()
plt.savefig("cell19_M{}_B{}_K{}_cdf_ul_under.pdf".format(M, B, K), format="pdf", bbox_inches="tight")
plt.show()
