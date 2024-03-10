import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

Q_bits = [6, 7, 8, 9, 10]

EVD = [5.96411, 6.5909, 6.94144, 7.07685, 7.1201]
EVD_beta = [6.29294, 6.922306, 7.27156, 7.41248, 7.45614]
DNN_local = [6.60386, 7.22121, 7.56042, 7.71266, 7.76107]
GRU_T4 = [7.72426, 9.05968, 10.06645, 10.70723, 10.99004]
global_CSI = [8.01969, 9.60787, 10.91321, 11.79762, 12.2288]

font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)
plt.figure(figsize=(7, 5))

plt.plot(Q_bits, global_CSI, color='tab:purple', marker="s", markersize=8, label="GD w/ global CSI")
plt.plot(Q_bits, GRU_T4, color='tab:blue', marker="*", markersize=12, label="DNN w/ local CSI + GRU (T=4)")
plt.plot(Q_bits, DNN_local, color='tab:orange', marker="o", markersize=8, label="DNN w/ local CSI")
plt.plot(Q_bits, EVD_beta, color='tab:cyan', marker="^", markersize=8, label="Modified EVD w/ local CSI")
plt.plot(Q_bits, EVD, color='tab:green', marker=">", markersize=8, label="EVD w/ local CSI")

plt.axhline(y=12.50534, color='tab:purple', linestyle='--')
plt.axhline(y=11.20161, color='tab:blue', linestyle='--')
plt.axhline(y=7.82555, color='tab:orange', linestyle='--')
plt.axhline(y=7.57274, color='tab:cyan', linestyle='--')
plt.axhline(y=7.2773, color='tab:green', linestyle='--')
plt.axhline(y=6.28415, color='tab:gray', linestyle='--', label="Single-cell processing")
# plt.axhline(y=4.46, color='tab:gray', linestyle='--', label="compress-forward")

plt.xticks(Q_bits)
plt.ylim(5.5, 15)
plt.xlabel("Bits per dimension", fontsize=14)
plt.ylabel("Average per-user rate (bps/Hz)", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("cell19_quant_dl.pdf", format="pdf", bbox_inches="tight")

plt.show()
