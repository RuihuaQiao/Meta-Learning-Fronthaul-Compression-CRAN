import numpy as np
from funcs import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim  # for gradient descent
from torch import linalg as LA
import time
import pathlib
from scipy.io import loadmat
import os

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# torch.set_num_threads(1)

### paramaters
B2Bdist = 0.15
center_radius = 0
RRH_height = 0.03  # km
M = 32  # number of antennas per RRH
B = 17  # size of Theta_n
K = 2  # compression dimension
Nc = 2  # number of UEs per cell
Ball = 19  # total number of RRH
Nall = Ball * Nc  # total number of UEs
n_UELocSamples = 50  # to simulate the user location in hexagonal grid

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

### training configs
print(f'Using {device} device')
learning_rate = 1e-4
batch_size = 500
val_size = 2000
test_size = 2000
batch_per_epoch = 50
num_epochs = 0
initial_run = 0  # 0: continue to train, 1: start from scratch


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.Dense_1 = nn.Linear(M * Nall, 2048)
        self.Dense_2 = nn.Linear(2048, 512)
        self.Dense_3 = nn.Linear(512, K * M)

    def forward(self, H):
        """
        compute W from H
        :param H: (bs, B, M, Nall)
        :return: W_set: Ball * (bs, K, M)
        """
        W_set = {}
        # H_log = torch.sign(H) * torch.sqrt(torch.abs(H))
        for bbb in range(Ball):
            hi_flat = nn.Flatten()(H[:, bbb, :, :])  # (bs, M*Nall)
            # hi_flat = nn.Flatten()(H_cov[:, bbb, :, :])  # (bs, M*M)
            layer_1 = torch.tanh(self.Dense_1(hi_flat))
            # layer_1_normed = self.BatchNorm_1(layer_1)
            layer_2 = torch.tanh(self.Dense_2(layer_1))
            # layer_2_normed = self.BatchNorm_2(layer_2)
            Wi_flat = self.Dense_3(layer_2)  # (bs, K*M)
            Wi = Wi_flat.view(-1, K, M)  # (bs, K, M)
            Wi_normed = Wi / LA.vector_norm(Wi, dim=-1, keepdims=True)  # normalize each row
            W_set[bbb] = gramschmidt(Wi_normed)
        return W_set


### model path
model_path = './saved_model/UserCentricDNN_cell{}_Nc{}_M{}_B{}_K{}'.format(Ball, Nc, M, B, K)
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

DNN_local = FCNN().to(device)
print("DNN Local CSI: ", DNN_local)

if initial_run == 0:  # continue to train
    DNN_local.load_state_dict(torch.load(model_path + "/DNN_local.zip", map_location=torch.device(device)))

### generate user location grid
UELocs_discrete, in_cell = gen_discrete_UEs(B2Bdist, center_radius, n_UELocSamples)  # user location grid (in_cell, )

##### training
torch.manual_seed(1)
np.random.seed(1)
params = list(DNN_local.parameters())
optimizer = optim.Adam(params, lr=learning_rate)  # weight_decay=1e-6
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

UELocs_val = scheduling(UELocs_discrete, val_size, Nall)  # (bs, Nall)
dist_set_val = compute_dist_set(B2Bdist, UELocs_val, RRH_height, Nc)  # (bs, Ball, Nall)
Theta_set_val = gen_Theta_set(UELocs_val, B2Bdist, Nc, B)  # (bs, Nall, B)
H_set_val, H_set_val_sorted = gen_channel(UELocs_val, dist_set_val, M, scaling_factor)  # (bs, B, M, Nall)
### DNN forward prop
with torch.no_grad():
    W_set_val = DNN_local(H_set_val_sorted)  # dict: Ball * (bs, K, M)

### compute rate
rate_val = compute_rate(H_set_val, W_set_val, Theta_set_val, sigma2_zx_ratio)
best_rate_val = rate_val
print("Initial val rate: %.5f" % best_rate_val.item())

### training
print('Start training')
train_losses, val_losses = [], []
loss_train = 0
for ee in range(num_epochs):
    start_time = time.time()
    for ii in range(batch_per_epoch):
        UELocs_train = scheduling(UELocs_discrete, batch_size, Nall)  # (bs, Nall)
        dist_set_train = compute_dist_set(B2Bdist, UELocs_train, RRH_height, Nc)  # (bs, Ball, Nall)
        Theta_set_train = gen_Theta_set(UELocs_train, B2Bdist, Nc, B)  # (bs, Nall, B)
        H_set_train, H_set_train_sorted = gen_channel(UELocs_train, dist_set_train, M, scaling_factor)  # (bs, M, Nall)
        ### DNN forward prop
        W_set_train = DNN_local(H_set_train_sorted)  # dict: Ball * (bs, K, M)

        ### compute rate
        rate_train = compute_rate(H_set_train, W_set_train, Theta_set_train, sigma2_zx_ratio)

        loss_train = -rate_train
        ### model update
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        train_losses.append(loss_train.item())

        W_set_val = DNN_local(H_set_val_sorted)  # dict: Ball * (bs, K, M)

        rate_val = compute_rate(H_set_val, W_set_val, Theta_set_val, sigma2_zx_ratio)
        val_losses.append(-rate_val.item())
        scheduler.step(val_losses[-1])
        print("learning rate: ", optimizer.param_groups[0]["lr"])

    print('Epoch{}, Train rate: {} | Val rate: {} | {}min{}s'
          .format(ee, "%.5f" % -train_losses[-1], "%.5f" % -val_losses[-1],
                  "%.0f" % ((time.time() - start_time) / 60), "%.0f" % ((time.time() - start_time) % 60)))

    if (ee + 1) % 5 == 0 and rate_val > best_rate_val:
        best_rate_val = rate_val
        print("Best val rate: {}".format("%.5f" % best_rate_val))
        ### save the best model
        torch.save(DNN_local.state_dict(), model_path + "/DNN_local.zip")
        # torch.save(DNN_pow.state_dict(), model_path + "/DNN_pow.zip")

### plot training curve
if num_epochs != 0:
    plt.title("DNN (User Centric), K=%.0f" % K)
    plt.plot([-element for element in train_losses], label="Train")
    plt.plot([-element for element in val_losses], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Rate (bps/Hz)")
    # plt.yscale("log")
    plt.legend(loc='best')
    plt.show()

### testing
testset = loadmat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed42.mat".
                  format(test_size, M, Ball, B, Nc, n_UELocSamples))
UELocs_test = torch.tensor(testset['UELocs']).to(device)
dist_set_test = torch.tensor(testset['dist_set']).to(device)  # (bs, Ball, Nall)
Theta_set_test = torch.tensor(testset['Theta_set']).to(device)  # (bs, Nall, B)
H_set_test = torch.tensor(testset['H_set']).to(device)  # (bs, M, Nall)
H_set_test_sorted = torch.tensor(testset['H_set_sorted']).to(device)  # (bs, M, Nall)
bits_test = torch.ones((K,)).to(device) * Q_bits  # equal allocation

### DNN forward prop
with torch.no_grad():
    W_set_test = DNN_local(H_set_test_sorted)  # dict: Ball * (bs, K, M)
    ### compute rate
    rate_test = compute_rate(H_set_test, W_set_test, Theta_set_test, sigma2_zx_ratio)
    print("Test rate: {}".format("%.5f" % rate_test.item()))
    if use_quant:
        rate_test_q = compute_rate_quant(H_set_test, W_set_test, Theta_set_test, sigma2_zx_ratio, bits_test, quant_scaling, if_test=1)
        print("Test rate after quantization: {}".format("%.5f" % rate_test_q.item()))

print('end')
