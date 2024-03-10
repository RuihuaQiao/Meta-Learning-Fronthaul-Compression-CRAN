import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator

folder_path = "./"
Ball = 19
Nc = 2
test_size = 2000

M = 32
B = 17
K = 2
N = Ball * Nc
step_rnn = 7
step_GD = 7

overhead = np.array([((2 * ii) * K * N + K * N) for ii in range(step_GD + 1)]) / (M * N + K * M)

GRU = loadmat(folder_path+"GRU_M{}_B{}_K{}_step{}.mat".format(M, B, K, step_rnn))['rate_list']
# GRU_EVD = loadmat(folder_path+"GRU_EVD_M{}_B{}_K{}_step{}.mat".format(M, B, K, step_rnn))['rate_list']
GD = loadmat(folder_path + "GD_M{}_B{}_K{}_step{}.mat".format(M, B, K, step_GD))['rate_list']


font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)

fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2

### M=32, B=17, K=2
ax1.axhline(y=11.18175, color='tab:purple', linestyle='--', linewidth=2, label="GD w/ global CSI")
ax1.plot(GRU.squeeze(), marker="*", markersize=12, color="tab:blue", linewidth=2, label="DNN w/ local CSI + GRU")
# ax1.plot(GRU_EVD.squeeze(), marker=">", markersize=8, color="mediumaquamarine", label="EVD w/ local CSI + GRU")
ax1.plot(GD.squeeze(), marker="<", markersize=8, color="tab:pink", linewidth=2, label="DNN w/ local CSI + GD")
ax1.axhline(y=7.71360, color='tab:orange', linestyle='--', linewidth=2, label="DNN w/ local CSI")
ax1.legend(loc='upper left')
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

ax2.bar(np.arange(step_GD + 1), overhead, color="#C0C0C0", alpha=0.5, label="Overhead")
ax2.set_ylim(0.005, 1.05)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax1.set_xlabel("Refinement iterations", fontsize=15)
ax1.set_ylabel("Average per-user rate (bps/Hz)", fontsize=15)
ax2.set_ylabel("Communication overhead ratio to global CSI", fontsize=15)

plt.tight_layout()
plt.grid()

plt.savefig("cell19_M{}_B{}_convergence_K{}_dl.pdf".format(M, B, K), format="pdf", bbox_inches="tight")
plt.show()
