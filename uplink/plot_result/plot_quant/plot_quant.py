import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

Q_bits = [6, 7, 8, 9, 10]

EVD = [6.2038, 6.70009, 6.93455, 7.02354, 7.0498]
EVD_beta = [6.46197, 6.988234, 7.23294, 7.32616, 7.35546]
DNN_local = [6.56507, 7.16389, 7.44603, 7.573009, 7.60278]
GRU_T4 = [7.79263, 9.02981, 9.8242, 10.23388, 10.39213]
global_CSI = [7.67015, 9.10061, 10.15307, 10.75434, 10.99304]


font = {'family': 'Times New Roman',
        'size': 12}
plt.rc('font', **font)
plt.figure(figsize=(7, 5))

plt.plot(Q_bits, global_CSI, color='tab:purple', marker="s", markersize=8, label="GD w/ global CSI")
plt.plot(Q_bits, GRU_T4, color='tab:blue', marker="*", markersize=12, label="DNN w/ local CSI + GRU (T=4)")
plt.plot(Q_bits, DNN_local, color='tab:orange', marker="o", markersize=8, label="DNN w/ local CSI")
plt.plot(Q_bits, EVD_beta, color='tab:cyan', marker="^", markersize=8, label="Modified EVD w/ local CSI")
plt.plot(Q_bits, EVD, color='tab:green', marker=">", markersize=8, label="EVD w/ local CSI")


plt.axhline(y=11.18224, color='tab:purple', linestyle='--')
plt.axhline(y=10.47112, color='tab:blue', linestyle='--')
plt.axhline(y=7.63758, color='tab:orange', linestyle='--')
plt.axhline(y=7.37209, color='tab:cyan', linestyle='--')
plt.axhline(y=7.06561, color='tab:green', linestyle='--')
plt.axhline(y=6.3610, color='tab:gray', linestyle='--', label="Single-cell processing")

plt.xticks(Q_bits)
plt.ylim(6, 13.5)
plt.xlabel("Bits per dimension", fontsize=14)
plt.ylabel("Average per-user rate (bps/Hz)", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("cell19_quant_ul.pdf", format="pdf", bbox_inches="tight")

plt.show()
