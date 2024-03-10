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

if_quant = 0
Q_bits = 6

step_size = 0.001
num_iter = 7

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 30  # dBm
sigmax2 = db2pow(UEpow - 30)/2
sigmaz2 = db2pow(bg_noise - 30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

print(f'Using {device} device')

ini_local = 1  # 1: initialize with local CSI; 0: random initialization
load_W = 0  # load trained W


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
    model_path = './saved_model/UserCentric_DL_DNN_cell{}_Nc{}_M{}_B{}_K{}'.format(Ball, Nc, M, B, K)
    DNN_local = FCNN().to(device)
    DNN_local.load_state_dict(torch.load(model_path + "/DNN_local.zip", map_location=torch.device(device)))
    # W_localCSI_dict = {}
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

rate_convergence = []
for ee in range(num_iter+1):  # first iteration is ini
    with torch.no_grad():
        for bb in range(Ball):
            W_global_set[bb].data = gramschmidt(W_global_set[bb])
    Cn_set, Fn_bar_set = compute_Cn_Fnbar(H_set, W_global_set, Theta_set, sigma2_zx_ratio)
    rate_iter = compute_rate(H_set, W_global_set, Theta_set, Cn_set, Fn_bar_set, sigma2_zx_ratio)
    rate_convergence.append(rate_iter.item())
    print('Iter{}, Rate: {} | {}min{}s'.format(ee, "%.5f" % rate_iter.item(),
                                               "%.0f" % ((time.time() - start_time) // 60),
                                               "%.0f" % ((time.time() - start_time) % 60)))

    loss_train = -rate_iter * batch_size * Nall * np.log(2)
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (ee+1) % 50 == 0:
        ### save grad-trained W
        save_dict = {}
        for bb in range(Ball):
            save_dict[str(bb)] = W_global_set[bb].detach().cpu().numpy()
        # save_dict["power"] = pow_alloc.detach().cpu().numpy()
        savemat("./saved_model/trained_W/W_global_localini{}_bs{}_Nc{}_B{}_M{}_K{}.mat".
                format(ini_local, batch_size, Nc, B, M, K), save_dict)
        print("trained W is saved")

if num_iter > 0:
    plt.title("Global CSI Convergence K={}, lr={}".format(K, step_size))
    plt.plot(rate_convergence)
    plt.xlabel("Iterations")
    plt.ylabel("Rate (bps/Hz)")
    plt.grid()
    plt.show()


# test final rate
with torch.no_grad():
    ### compute rate
    Cn_set_test, Fn_bar_set_test = compute_Cn_Fnbar(H_set, W_global_set, Theta_set, sigma2_zx_ratio)
    rate_test = compute_rate(H_set, W_global_set, Theta_set, Cn_set_test, Fn_bar_set_test, sigma2_zx_ratio)
    print("Test rate: %.5f" % rate_test.item())

    if if_quant:
        C_mtx_fullsize_test = expand_C_mtx(Theta_set, Cn_set_test)
        gamma = torch.tensor([3], requires_grad=False, device=device)
        bits_test = torch.ones((K,)).to(device) * Q_bits  # equal allocation

        rate_test_quant = compute_rate_quant(H_set, W_global_set, Theta_set, C_mtx_fullsize_test,
                                             sigma2_zx_ratio, bits_test, gamma, if_test=1)
        print("Test rate quant: %.5f" % rate_test_quant.item())

        quant_range = np.arange(1, 4, 0.1)
        rate_quant = []
        for ii in quant_range:
            quant_scaling_ = torch.tensor(ii)
            rate = compute_rate_quant(H_set, W_global_set, Theta_set,C_mtx_fullsize_test,
                                      sigma2_zx_ratio, Q_bits, quant_scaling_, if_test=1).item()
            print(quant_scaling_.item(), rate)
            rate_quant.append(rate)
        plt.plot(quant_range, rate_quant)
        plt.grid()
        plt.show()
print('end')
