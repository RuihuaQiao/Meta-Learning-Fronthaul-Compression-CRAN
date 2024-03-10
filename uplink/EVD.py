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
n_UELocSamples = 50

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 23  # dBm
sigmax2 = db2pow(UEpow-30)
sigmaz2 = db2pow(bg_noise-30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

use_quant = True
Q_bits = 6
quant_scaling = 3

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

###
# ------------ EVD beta ----------------
# Phi_set = {}
# for bb in range(Ball):
#     Phi_b = torch.zeros(batch_size, Nall).to(device)
#     for ii in range(Nall):
#         d_bi = dist_set[:, bb, ii].view(-1, 1)  # (bs, 1)
#         beta_bi_db = 180 - 147.5 + 20 * np.log10(2.9) + 29 * torch.log10(d_bi * 1000)
#         beta_bi = db2pow(-beta_bi_db) / scaling_factor
#         Phi_b[:, ii] = beta_bi.squeeze() * M
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

# ----------------------------------
# ------------ EVD ----------------
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
# W_EVD_set = func_EVD(H_set, sigma2_zx_ratio, K)
# -------------------------------------

with torch.no_grad():
    rate_EVD = compute_rate(H_set, W_EVD_set, Theta_set, sigma2_zx_ratio)
    print(rate_EVD.item())
    if use_quant:
        rate_EVD_q = compute_rate_quant(H_set, W_EVD_set, Theta_set, sigma2_zx_ratio, Q_bits, quant_scaling, if_test=1)
        print(rate_EVD_q.item())

print('Time used: {}min{}s'
      .format("%.0f" % ((time.time() - start_time) / 60), "%.0f" % ((time.time() - start_time) % 60)))

print('end')
