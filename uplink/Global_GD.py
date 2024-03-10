from funcs import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # for gradient descent
from torch import linalg as LA
import time
import math
from scipy.io import loadmat
import pathlib
import os

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
batch_size = 2000
n_UELocSamples = 50

step_size = 0.001
num_iter = 2

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 23  # dBm
sigmax2 = db2pow(UEpow - 30)
sigmaz2 = db2pow(bg_noise - 30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

use_quant = False
Q_bits = 6
quant_scaling = 3

print(f'Using {device} device')

ini_local = 1  # start with local CSI based DNN
load_W = 1  # load trained W


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.Dense_1 = nn.Linear(M * Nall, 2048)
        self.Dense_2 = nn.Linear(2048, 512)
        self.Dense_3 = nn.Linear(512, K * M)

    def forward(self, H):
        """
        compute W from local CSI
        :param H: (bs, B, M, Nall)
        :return: W_set: Ball * (bs, K, M)
        """
        W_set = {}
        for bbb in range(Ball):
            hi_flat = nn.Flatten()(H[:, bbb, :, :])  # (bs, M*Nall)
            layer_1 = torch.tanh(self.Dense_1(hi_flat))
            layer_2 = torch.tanh(self.Dense_2(layer_1))
            Wi_flat = self.Dense_3(layer_2)  # (bs, K*M)
            Wi = Wi_flat.view(-1, K, M)  # (bs, K, M)
            Wi_normed = Wi / LA.vector_norm(Wi, dim=-1, keepdims=True)  # normalize each row
            W_set[bbb] = gramschmidt(Wi_normed)
        return W_set


torch.manual_seed(1)
start_time = time.time()

##### load dataset
testset = loadmat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed42.mat".
                  format(batch_size, M, Ball, B, Nc, n_UELocSamples))
UELocs = torch.tensor(testset['UELocs']).to(device)
dist_set = torch.tensor(testset['dist_set']).to(device)  # (bs, Ball, Nall)
Theta_set = torch.tensor(testset['Theta_set']).to(device)  # (bs, Nall, B)
H_set = torch.tensor(testset['H_set']).to(device)  # (bs, Ball, M, Nall)
H_set_sorted = torch.tensor(testset['H_set_sorted']).to(device)  # (bs, Ball, M, Nall)

### initialization
if load_W == 1:  # load trained W
    W_load_dict = loadmat("./saved_model/trained_W/W_global_localini{}_bs{}_Nc{}_B{}_M{}_K{}.mat".
                     format(ini_local, batch_size, Nc, B, M, K))
    W_global_set = {}
    for bb in range(Ball):
        W_global_set[bb] = torch.tensor(W_load_dict[str(bb)], requires_grad=True, device=device)
elif ini_local == 1:  # initialize with local CSI
    model_path = './saved_model/UserCentricDNN_cell{}_Nc{}_M{}_B{}_K{}'.format(Ball, Nc, M, B, K)
    DNN_local = FCNN().to(device)
    DNN_local.load_state_dict(torch.load(model_path + "/DNN_local.zip"))
    with torch.no_grad():
        W_localCSI_dict = DNN_local(H_set_sorted)
    W_global_set = {}
    for bb in range(Ball):
        W_global_set[bb] = W_localCSI_dict[bb].clone().detach().requires_grad_(True)
else:  # randomly initialize
    W_global_set = {}
    for bb in range(Ball):
        W_global_set[bb] = torch.randn((batch_size, K, M), requires_grad=True, device=device)


params = []
for bb in range(Ball):
    params = params + [W_global_set[bb]]
optimizer = optim.SGD(params, lr=step_size)


rate_convergence, rate_q_convergence = [], []
for ee in range(num_iter+1):  # first iteration is ini
    with torch.no_grad():
        for bb in range(Ball):
            W_global_set[bb].data = gramschmidt(W_global_set[bb])

    if not use_quant:  # do not add quantization
        rate_iter = compute_rate(H_set, W_global_set, Theta_set, sigma2_zx_ratio)
        rate_convergence.append(rate_iter.item())
        print('Iter{}, Rate: {} | {}min{}s'.format(ee, "%.5f" % rate_iter.item(),
                                                   "%.0f" % ((time.time() - start_time) // 60),
                                                   "%.0f" % ((time.time() - start_time) % 60)))

        loss_train = -rate_iter * batch_size * Nall * np.log(2)

    else:  # add quantization
        with torch.no_grad():
            rate_iter = compute_rate(H_set, W_global_set, Theta_set, sigma2_zx_ratio)
        bits_alloc = torch.ones((K,)).to(device) * Q_bits  # equal allocation
        rate_q_iter = compute_rate_quant(H_set, W_global_set, Theta_set, sigma2_zx_ratio, bits_alloc, Q_bits, if_test=1)

        rate_convergence.append(rate_iter.item())
        rate_q_convergence.append(rate_q_iter.item())
        print('Iter{}, Rate: {} | R_q: {} | {}min{}s'.format(ee, "%.5f" % rate_iter.item(), "%.5f" % rate_q_iter.item(),
                                                             "%.0f" % ((time.time() - start_time) // 60),
                                                             "%.0f" % ((time.time() - start_time) % 60)))
        loss_train = -rate_q_iter * batch_size * Nall * np.log(2)

    loss_train.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (ee+1) % 10 == 0:
        ### save grad-trained W
        save_dict = {}
        for bb in range(Ball):
            save_dict[str(bb)] = W_global_set[bb].detach().cpu().numpy()
        savemat("./saved_model/trained_W/W_global_localini{}_bs{}_Nc{}_B{}_M{}_K{}_UEpow{}.mat".
                format(ini_local, batch_size, Nc, B, M, K, UEpow), save_dict)
        print("trained W is saved")


plt.title("Global CSI Convergence K={}, lr={}".format(K, step_size))
plt.plot(rate_convergence)
plt.xlabel("Iterations")
plt.ylabel("Rate (bps/Hz)")
plt.grid()
plt.show()


result_dict = {'rate_list': rate_convergence}
savemat("./plot_result/plot_convergence/GD_M{}_B{}_K{}_step{}.mat".format(M, B, K, num_iter), result_dict)

print('end')
