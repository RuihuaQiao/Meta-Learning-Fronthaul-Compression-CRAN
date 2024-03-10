import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


Ball = 19
Nc = 2
test_size = 2000

M = 32
B = 17
K = 2
N = Ball * Nc
step_rnn = 7
step_GD = 7

overhead = np.array([((2*ii)*K*N + K*N) for ii in range(step_GD+1)]) / (M*N + K*M)

GRU = loadmat("GRU_M{}_B{}_K{}_step{}.mat".format(M, B, K, step_rnn))['rate_list']
adam_2 = loadmat("Adam_M{}_B{}_K{}_step{}_lr0.02.mat".format(M, B, K, step_GD))['rate_list']
adam_3 = loadmat("Adam_M{}_B{}_K{}_step{}_lr0.03.mat".format(M, B, K, step_GD))['rate_list']
adam_4 = loadmat("Adam_M{}_B{}_K{}_step{}_lr0.04.mat".format(M, B, K, step_GD))['rate_list']
adam_5 = loadmat("Adam_M{}_B{}_K{}_step{}_lr0.05.mat".format(M, B, K, step_GD))['rate_list']
rmsp_2 = loadmat("RMSprop_M{}_B{}_K{}_step{}_lr0.002.mat".format(M, B, K, step_GD))['rate_list']
rmsp_3 = loadmat("RMSprop_M{}_B{}_K{}_step{}_lr0.003.mat".format(M, B, K, step_GD))['rate_list']
rmsp_5 = loadmat("RMSprop_M{}_B{}_K{}_step{}_lr0.005.mat".format(M, B, K, step_GD))['rate_list']
GD_8 = loadmat("SGD_M{}_B{}_K{}_step{}_lr0.008.mat".format(M, B, K, step_GD))['rate_list']
GD_10 = loadmat("SGD_M{}_B{}_K{}_step{}_lr0.01.mat".format(M, B, K, step_GD))['rate_list']


font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2

### M=32, B=17, K=2
ax1.axhline(y=11.18, color='tab:purple', linestyle='--', linewidth=2, label="GD w/ global CSI")
# ax1.plot(Binary.squeeze(), marker="*", markersize=12, color="tab:olive", label="Binary gradient")
ax1.plot(GRU.squeeze(), marker="*", markersize=12, color="tab:blue", linewidth=2, label="DNN w/ local CSI + GRU")
ax1.plot(adam_2.squeeze(), marker="d", color="tab:olive", linestyle='-', linewidth=2, label="DNN w/ local CSI + Adam (lr=0.02)")
# ax1.plot(adam_3.squeeze(), marker="<", markersize=8, color="tab:pink", linestyle='-.', label="DNN w/ local CSI + Adam (lr=0.03)")
ax1.plot(adam_4.squeeze(), marker="D", color="tab:green", linestyle=':', linewidth=2, label="DNN w/ local CSI + Adam (lr=0.04)")
ax1.plot(adam_5.squeeze(), marker="s", color="yellowgreen", linestyle=(5, (10, 3)), linewidth=2, label="DNN w/ local CSI + Adam (lr=0.05)")
ax1.plot(rmsp_2.squeeze(), marker="1", markersize=8, color="tab:orange", linestyle='-', linewidth=2, label="DNN w/ local CSI + RMSP (lr=0.002)")
ax1.plot(rmsp_3.squeeze(), marker="2", markersize=8, color="peru", linestyle='-', linewidth=2, label="DNN w/ local CSI + RMSP (lr=0.003)")
ax1.plot(rmsp_5.squeeze(), marker="3", markersize=8, color="sandybrown", linestyle='-.', linewidth=2, label="DNN w/ local CSI + RMSP (lr=0.005)")
ax1.plot(GD_8.squeeze(), marker="<", markersize=8, color="tab:pink", linewidth=2, label="DNN w/ local CSI + GD (lr=0.008)")
ax1.plot(GD_10.squeeze(), marker=">", markersize=8, color="deeppink", linestyle='-.', linewidth=2, label="DNN w/ local CSI + GD (lr=0.01)")
ax1.axhline(y=7.63, color='tab:orange', linestyle='--', linewidth=2, label="DNN w/ local CSI")

ax1.legend()
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])

ax1.set_xlabel("Refinement iterations", fontsize=17)
ax1.set_ylabel("Average per-user rate (bps/Hz)", fontsize=17)


plt.tight_layout()
plt.grid()

plt.savefig("cell19_M{}_B{}_convergence_K{}_ul_vertical.pdf".format(M, B, K), format="pdf", bbox_inches="tight")
plt.show()
