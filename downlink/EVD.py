import matplotlib.pyplot as plt
import numpy as np
import torch
from funcs import *
import torch.nn as nn
import torch.optim as optim  # for gradient descent
from torch import linalg as LA
import time
import math
from scipy.io import loadmat

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# torch.set_num_threads(1)


M = 32
B = 17  # size of Theta_n
K = 2
Nc = 2  # number of UEs per cell
Ball = 19  # total number of RRH
Nall = Ball * Nc  # total number of UEs
Q_bits = 10
n_UELocSamples = 50

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 30  # dBm
sigmax2 = db2pow(UEpow-30)/2
sigmaz2 = db2pow(bg_noise-30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

use_cuda = 1
device = 'cuda' if torch.cuda.is_available() and use_cuda == 1 else 'cpu'
print(f'Using {device} device')

batch_size = 2000

torch.manual_seed(1)
start_time = time.time()


testset = loadmat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed42.mat".
                  format(batch_size, M, Ball, B, Nc, n_UELocSamples))
UELocs = torch.tensor(testset['UELocs']).to(device)
dist_set = torch.tensor(testset['dist_set']).to(device)  # (bs, Ball, Nall)
Theta_set = torch.tensor(testset['Theta_set']).to(device)  # (bs, Nall, B)
H_set = torch.tensor(testset['H_set']).to(device)  # (bs, M, Nall)
# plot_cell_UEs(B2Bdist, plot_UE=True, UE_1_loc=UELocs[0, ...])

##### EVD
W_EVD_set = {}
for bb in range(Ball):
    Hb = H_set[:, bb, :, :]
    diag_M = torch.diag_embed(torch.ones((batch_size, M)).to(device))

    Sigma_ybyb = Hb @ Hb.mH + sigma2_zx_ratio * diag_M  # (bs, M, M)
    U, S, V = torch.svd(Sigma_ybyb)
    W_bb = U[:, :, 0:K].mH  # (bs, K, M)
    # W_bb = torch.randn(batch_size, K, M).to(device)
    W_EVD_set[bb] = gramschmidt(W_bb)
    # W_EVD_set[bb] = diag_M
W_EVD_set = func_EVD(H_set, sigma2_zx_ratio, K)
######


###### Modified EVD (beta)
# Phi_set = {}
# for bb in range(Ball):
#     Phi_b = torch.zeros(batch_size, Nall).to(device)
#     for ii in range(Nall):
#         d_bi = dist_set[:, bb, ii].view(-1, 1)  # (bs, 1)
#         beta_bi_db = 180 - 147.5 + 20 * np.log10(2.9) + 29 * torch.log10(d_bi * 1000)
#         beta_bi = db2pow(-beta_bi_db) / scaling_factor
#         Phi_b[:, ii] = beta_bi.squeeze()
#         Phi_set[bb] = torch.diag_embed(Phi_b)
#
# W_EVD_set = {}
# for bb in range(Ball):
#     Hb = H_set[:, bb, :, :]
#     diag_Nall = torch.diag_embed(torch.ones((batch_size, Nall)).to(device))
#     Phi_sum = torch.zeros_like(Phi_set[bb])
#     for jj in range(Ball):
#         if jj != bb:
#             Phi_sum += Phi_set[jj]
#     Sigma_new = Hb @ LA.inv(Phi_sum + sigma2_zx_ratio*diag_Nall) @ Hb.mH  # (bs, M, M)
#     U, S, V = torch.svd(Sigma_new)
#     W_bb = U[:, :, 0:K].mH  # (bs, K, M)
#     # W_bb = torch.randn(batch_size, K, M).to(device)
#     W_EVD_set[bb] = gramschmidt(W_bb)
#####

Cn_set, Fn_bar_set = compute_Cn_Fnbar(H_set, W_EVD_set, Theta_set, sigma2_zx_ratio)
C_mtx_fullsize = expand_C_mtx(Theta_set, Cn_set)
# W_EVD_set_tilde, S_set, C_mtx_fullsize = compute_W_tilde(W_EVD_set, H_set, C_mtx_fullsize, sigma2_zx_ratio)

gamma = torch.tensor([3], requires_grad=False, device=device)
# bits_EVD = bits_allocation(S_set, Q_bits, K)  # water-filling allocation
bits_EVD = torch.ones((K, )).to(device) * Q_bits  # equal allocation

with torch.no_grad():
    rate_EVD = compute_rate(H_set, W_EVD_set, Theta_set, Cn_set, Fn_bar_set, sigma2_zx_ratio)
    rate_EVD_quant = compute_rate_quant(H_set, W_EVD_set, Theta_set, C_mtx_fullsize,
                                        sigma2_zx_ratio, bits_EVD, gamma, if_test=1)
print("%.5f" % rate_EVD.item())
print("%.5f" % rate_EVD_quant.item())
print('Time used: {}min{}s'
      .format("%.0f" % ((time.time() - start_time) / 60), "%.0f" % ((time.time() - start_time) % 60)))

quant_range = np.arange(1, 4, 0.1)
rate_quant = []
for ii in quant_range:
    quant_scaling_ = torch.tensor(ii)
    rate = compute_rate_quant(H_set, W_EVD_set, Theta_set, C_mtx_fullsize, sigma2_zx_ratio, bits_EVD, quant_scaling_, if_test=1).item()
    print(quant_scaling_.item(), rate)
    rate_quant.append(rate)
plt.plot(quant_range, rate_quant)
plt.grid()
plt.show()


print('end')
