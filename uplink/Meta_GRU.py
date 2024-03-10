import matplotlib.pyplot as plt
import numpy as np
import torch
from funcs import *
from funcs_autograd import *
# from funcs_grad import compute_grad
import torch.nn as nn
import torch.optim as optim  # for gradient descent
from torch import linalg as LA
import time
import pathlib
import os
from scipy.io import loadmat


# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# torch.set_num_threads(1)

### paramaters
B2Bdist = 0.15
center_radius = 0
RRH_height = 0.03  # km
M = 32
B = 17  # size of Theta_n
K = 2
Nc = 2  # number of UEs per cell
Ball = 19  # total number of RRH
Nall = Ball * Nc  # total number of UEs
n_UELocSamples = 50

### SNR configs
bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 23  # dBm
sigmax2 = db2pow(UEpow - 30)
sigmaz2 = db2pow(bg_noise - 30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

### training configs
print(f'Using {device} device')
learning_rate = 1e-4
batch_size = 200
val_size = 2000
test_size = 2000
batch_per_epoch = 50
num_epochs = 0
initial_run = 0

use_quant = False
Q_bits = 8
quant_scaling = 3
bits_alloc = torch.ones((K,)).to(device) * Q_bits  # equal allocation

gru_hs = 2*K*M
time_step = 7
time_step_load = 7


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


class GRU_meta(nn.Module):
    def __init__(self):
        super(GRU_meta, self).__init__()
        self.gru_cell_rrh = nn.GRUCell(input_size=K*M, hidden_size=gru_hs)
        self.Dense_1 = nn.Linear(gru_hs, 2*K*M)
        self.Dense_2 = nn.Linear(2*K*M, K*M)

    def forward(self, W_set, H_set, Theta_set):
        """
        meta learning
        :param W_set: dict: Ball * (bs, K, M)
        :param H_set: (bs, M, Nall)
        :param Theta_set: (bs, Nall, B)
        :return: updated W_set: dict: Ball * (bs, K, M)
        """
        data_size = H_set.shape[0]

        R_ini = compute_rate(H_set, W_set, Theta_set, sigma2_zx_ratio)
        R_list, Rq_list = [R_ini.item()], []  # list without grad info
        if use_quant:
            Rq_ini = compute_rate_quant(H_set, W_set, Theta_set, sigma2_zx_ratio, bits_alloc, quant_scaling, if_test=1)
            Rq_list.append(Rq_ini.item())
        R_sum_time = 0  # tensor with grad info

        ### initialize hidden unit
        hn_set = {bbb: torch.zeros(data_size, gru_hs).to(device) for bbb in range(Ball)}

        grad_abs = np.zeros((Ball, time_step, K * M))
        delta_W_abs = np.zeros((Ball, time_step, K * M))

        for eee in range(time_step):
            ### compute gradient of MSE_tilde w.r.t. Wi
            W_np = {bb: W_set[bb].cpu().detach().numpy() for bb in range(Ball)}
            # gamma_np = np.array(3)
            # W_grad, _ = autograd(W_np, gamma_np, H_set, Theta_set, sigma2_zx_ratio, grad_gamma=False, grad_W=True)
            W_grad = autograd(W_np, H_set, Theta_set, sigma2_zx_ratio, grad_W=True)
            # print(np.mean([torch.norm(W_grad[bb]).detach().cpu().numpy() for bb in range(19)]))

            for bbb in range(Ball):
                grad_abs[bbb, eee] = np.mean(torch.abs(W_grad[0]).detach().cpu().numpy(), axis=0).flatten()

                ## normalization
                Wb_grad = W_grad[bbb] / LA.vector_norm(W_grad[bbb], dim=(1, 2), keepdims=True)
                ### an GRU cell
                input_flat_RRHb = nn.Flatten()(Wb_grad)  # (bs, K*M)
                hn_set[bbb] = self.gru_cell_rrh(input_flat_RRHb, hn_set[bbb])  # (bs, hs)
                ### use hn_rrhi to produce output
                Delta_Wb_flat = self.Dense_2(torch.relu(self.Dense_1(hn_set[bbb])))  # (bs, KM)

                delta_W_abs[bbb, eee] = np.mean(torch.abs(Delta_Wb_flat).detach().cpu().numpy(), axis=0).flatten()

                ### update W_set
                W_set[bbb] = W_set[bbb] + Delta_Wb_flat.view(data_size, K, M)  # (bs, K, M)
                W_set[bbb] = gramschmidt(W_set[bbb])  # orthogonalize

            R_new = compute_rate(H_set, W_set, Theta_set, sigma2_zx_ratio)  # weighted sum: (ee+1) *
            R_sum_time = R_sum_time + R_new
            R_list.append(R_new.item())

            if use_quant:
                R_new_q = compute_rate_quant(H_set, W_set, Theta_set, sigma2_zx_ratio,
                                             bits_alloc, quant_scaling, if_test=1)

                Rq_list.append(R_new_q.item())

        R_avg_time = R_sum_time / time_step

        return W_set, R_avg_time, R_list, Rq_list


### model path
model_path = './saved_model/UserCentricDNN_GRU_cell{}_Nc{}_M{}_B{}_K{}_step{}'.format(Ball, Nc, M, B, K, time_step_load)
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

DNN_local = FCNN().to(device)
print("DNN Local CSI: ", DNN_local)
DNN_local.load_state_dict(torch.load(model_path + "/DNN_local.zip", map_location=torch.device(device)))
gru_meta = GRU_meta().to(device)
print('GRU_meta', gru_meta)


if initial_run == 0:  # continue to train
    gru_meta.load_state_dict(torch.load(model_path + "/GRU.zip", map_location=torch.device(device)))

### generate user location grid
UELocs_discrete, in_cell = gen_discrete_UEs(B2Bdist, center_radius, n_UELocSamples)  # user location grid (in_cell, )


##### training
torch.manual_seed(1)
optimizer = optim.Adam(gru_meta.parameters(), lr=learning_rate)  # weight_decay=1e-6
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

UELocs_val = scheduling(UELocs_discrete, val_size, Nall)  # (bs, Nall)
dist_set_val = compute_dist_set(B2Bdist, UELocs_val, RRH_height, Nc)  # (bs, Ball, Nall)
Theta_set_val = gen_Theta_set(UELocs_val, B2Bdist, Nc, B)  # (bs, Nall, B)
H_set_val, H_set_val_sorted = gen_channel(UELocs_val, dist_set_val, M, scaling_factor)  # (bs, M, Nall)
### DNN forward prop
with torch.no_grad():
    W_set_val = DNN_local(H_set_val_sorted)  # dict: Ball * (bs, K, M)
    # W_set_val = func_EVD(H_set_val, sigma2_zx_ratio, K)
    _, _, rate_val_list, _ = gru_meta(W_set_val, H_set_val, Theta_set_val)
best_rate_val = rate_val_list[-1]
print("Initial rate: %.5f" % rate_val_list[0], "| %.5f" % rate_val_list[-1])  # loss bf/af lstm

print('Start training')
train_losses, val_losses = [], []
rate_train_list = []
for ee in range(num_epochs):
    start_time = time.time()
    for ii in range(batch_per_epoch):

        UELocs_train = scheduling(UELocs_discrete, batch_size, Nall)  # (bs, Nall)
        dist_set_train = compute_dist_set(B2Bdist, UELocs_train, RRH_height, Nc)  # (bs, Ball, Nall)
        Theta_set_train = gen_Theta_set(UELocs_train, B2Bdist, Nc, B)  # (bs, Nall, B)
        H_set_train, H_set_train_sorted = gen_channel(UELocs_train, dist_set_train, M, scaling_factor)  # (bs, M, Nall)
        ### DNN forward prop
        with torch.no_grad():
            W_set_train = DNN_local(H_set_train_sorted)  # dict: Ball * (bs, K, M)
            # W_set_train = func_EVD(H_set_train_sorted, sigma2_zx_ratio, K)
        _, rate_train_avg, rate_train_list, _ = gru_meta(W_set_train, H_set_train, Theta_set_train)

        loss_train = -rate_train_avg
        ### model update
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        train_losses.append(-rate_train_list[-1])

        W_set_val = DNN_local(H_set_val_sorted)  # dict: Ball * (bs, K, M)
        # W_set_val = func_EVD(H_set_val, sigma2_zx_ratio, K)
        _, _, rate_val_list, _ = gru_meta(W_set_val, H_set_val, Theta_set_val)
        val_losses.append(-rate_val_list[-1])
        scheduler.step(-rate_val_list[-1])

    print("Epoch", ee, ": %.5f" % rate_val_list[0], "| %.5f" % rate_val_list[-1])  # loss bf/af lstm
    print('Epoch{}, Train avg R_tilde: {} | Val R: {} | {}min{}s'
          .format(ee, "%.5f" % -train_losses[-1], "%.5f" % -val_losses[-1],
                  "%.0f" % ((time.time() - start_time) / 60), "%.0f" % ((time.time() - start_time) % 60)))

    if (ee+1) % 5 == 0 and rate_val_list[-1] > best_rate_val:
        best_rate_val = rate_val_list[-1]
        print("Best val rate: {}".format("%.5f" % best_rate_val))
        ### save the best model
        # torch.save(DNN_local.state_dict(), model_path + "/DNN_local.zip")
        torch.save(gru_meta.state_dict(), model_path + '/GRU.zip')


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
with torch.no_grad():
    testset = loadmat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed42.mat".
                      format(test_size, M, Ball, B, Nc, n_UELocSamples))
    UELocs_test = torch.tensor(testset['UELocs']).to(device)
    dist_set_test = torch.tensor(testset['dist_set']).to(device)  # (bs, Ball, Nall)
    Theta_set_test = torch.tensor(testset['Theta_set']).to(device)  # (bs, Nall, B)
    H_set_test = torch.tensor(testset['H_set']).to(device)  # (bs, M, Nall)
    H_set_test_sorted = torch.tensor(testset['H_set_sorted']).to(device)  # (bs, M, Nall)
    ### DNN forward prop
    W_set_test = DNN_local(H_set_test_sorted)  # dict: Ball * (bs, K, M)
    # W_set_test = func_EVD(H_set_test, sigma2_zx_ratio, K)
    _, _, rate_test_list, rate_quant_test_list = gru_meta(W_set_test, H_set_test, Theta_set_test)

    print("Test rate:")
    for ii in range(len(rate_test_list)):
        print("Iter {}, rate {}".format(ii, "%.5f" % rate_test_list[ii]))
    ### save results
    result_dict = {'rate_list': rate_test_list}
    savemat("./plot_result/plot_convergence/GRU_M{}_B{}_K{}_step{}.mat".format(M, B, K, time_step, UEpow), result_dict)


print('end')
