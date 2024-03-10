from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

Ball = 19
Nc = 2
test_size = 2000

M = 32
B = 17
K = 2
N = Ball * Nc
step_rnn = 7
step_GD = 7

# CE_err_pow = [-65]  # db
local_CSI_DNN = []
GD_CE_err = [7.289156436920166,
             7.948282241821289,
             8.490324974060059,
             8.914934158325195,
             9.234373092651367,
             9.452374458312988,
             9.602144241333008,
             9.691629409790039]
GRU_CE_err = [7.287230968475342,
              8.444761276245117,
              9.242240905761719,
              9.74718189239502,
              10.06933307647705,
              10.276883125305176,
              10.41492748260498,
              10.50661849975586]

GD = loadmat("../plot_convergence/SGD_M{}_B{}_K{}_step{}_lr0.008.mat".format(M, B, K, step_GD))['rate_list'].squeeze()

GRU = loadmat("../plot_convergence/GRU_M{}_B{}_K{}_step{}.mat".format(M, B, K, step_rnn))['rate_list'].squeeze()

plt.figure(figsize=(8, 6))
font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)
plt.axhline(y=11.18, linestyle='--', color='tab:purple', label='GD w/ global CSI (perfect CSI)')
plt.axhline(y=10.86, linestyle=':', color='tab:purple', label='GD w/ global CSI (imperfect CSI)')
plt.plot(GRU, color="tab:blue", linestyle='-', marker="*", markersize=12, label='DNN w/ local CSI + GRU (perfect CSI)')
plt.plot(GRU_CE_err, color="tab:blue", marker="*", markersize=12, linestyle=':', label='DNN w/ local CSI + GRU (imperfect CSI)')
plt.plot(GD, color="tab:pink", linestyle='-', marker="<", markersize=8, label='DNN w/ local CSI +GD (perfect CSI)')
plt.plot(GD_CE_err, color="tab:pink", linestyle=':', marker="<", markersize=8, label='DNN w/ local CSI + GD (imperfect CSI)')
plt.axhline(y=7.65299, linestyle='--', color='tab:orange', label='DNN w/ local CSI (perfect CSI)')
plt.axhline(y=7.27746, linestyle=':', color='tab:orange', label='DNN w/ local CSI (imperfect CSI)')
plt.axhline(y=7.06, linestyle='--', color='tab:green', label='EVD w/ local CSI (perfect CSI)')
plt.axhline(y=7.05, linestyle=':', color='tab:green', label='EVD w/ local CSI (imperfect CSI)')

for line in plt.gca().get_lines():
    line.set_linewidth(2)

plt.legend(loc='lower right')
# plt.legend()
plt.grid()
plt.xlabel("Refinement iterations", fontsize=17)
plt.ylabel("Average per-user rate (bps/Hz)", fontsize=17)
plt.tight_layout()
plt.savefig('convergence_CE_err_-65dbm.pdf', bbox_inches="tight")
plt.show()
