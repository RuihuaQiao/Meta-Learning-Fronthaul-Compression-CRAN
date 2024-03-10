import warnings

import torch
from funcs import *
from torch import linalg as LA
from scipy.io import loadmat

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# torch.set_num_threads(1)


M = 32
B = 17  # size of Theta_n
Nc = 2  # number of UEs per cell
Ball = 19  # total number of RRH
Nall = Ball * Nc  # total number of UEs
batch_size = 2000
n_UELocSamples = 50

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
UEpow = 26  # dBm
sigmax2 = db2pow(UEpow-30)/2
sigmaz2 = db2pow(bg_noise-30) * bandwidth
scaling_factor = sigmaz2 * 100
sigmaz2 = sigmaz2 / scaling_factor
sigma2_zx_ratio = sigmaz2 / sigmax2

print(f'Using {device} device')

torch.manual_seed(1)

##### load dataset
testset = loadmat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed42.mat".
                  format(batch_size, M, Ball, B, Nc, n_UELocSamples))
H_set = torch.tensor(testset['H_set'], dtype=torch.double).to(device)  # (bs, Ball, M, Nall)
diag_M = torch.diag_embed(torch.ones((batch_size, M), dtype=torch.double).to(device))

##### local processing
sum_rate = torch.zeros((batch_size, 1), dtype=torch.double).to(device)

Cn_set = {}  # transmit beamformer for each user
for nn in range(Nall):
    bb = nn // Nc  # determine the serving RRH of UE nn
    Hb = H_set[:, bb, :, :]
    hnn = H_set[:, bb, :, nn].view(batch_size, M, 1)

    ### local transmit beamforming vector
    Cn = LA.inv(Hb @ Hb.mH + sigma2_zx_ratio * diag_M) @ hnn  # (bs, M, 1)
    Cn = Cn / LA.vector_norm(Cn, dim=-2, keepdims=True)  # (bs, M, 1)
    Cn_set[nn] = Cn

for nn in range(Nall):
    bb = nn // Nc  # determine the serving RRH of UE nn
    hnn = H_set[:, bb, :, nn].view(batch_size, M, 1)
    numerator = (torch.abs(hnn.mH @ Cn_set[nn]) ** 2).view(batch_size, 1)  # (bs, 1)

    power_all = 0
    for jj in range(Nall):
        bb_j = jj // Nc  # serving RRH for UE-j
        hnj = H_set[:, bb_j, :, nn].view(batch_size, M, 1)
        power_all = power_all + (torch.abs(hnj.mH @ Cn_set[jj]) ** 2).view(batch_size, 1)
    if torch.amax(power_all < numerator):  # throw warning when at least one sample satisfies condition
        warnings.warn("numerical issue in computing power_all")
    interference = power_all - numerator
    denominator = interference + sigma2_zx_ratio * torch.ones(batch_size, 1).to(device)  # (bs, 1)

    Rn = torch.log(1 + numerator/denominator / db2pow(6)) / np.log(2)
    sum_rate = sum_rate + Rn/Nall  # (bs, 1)

mean_sum_rate = torch.mean(sum_rate)  # average over samples in a batch

##### save & plot cdf curve
"""
sorted_rate = (torch.sort(sum_rate.squeeze())[0]).cpu().detach().numpy()  # x-axis
plt.plot(sorted_rate, np.arange(batch_size) / batch_size)
plt.show()
result_dict = {'sorted_rate': sorted_rate}
folder_path = "./plot_result/plot_cdf/"
method = "SingleCell"
savemat(folder_path + method + '_cell{}_M{}_Nc{}_B{}.mat'.format(Ball, M, Nc, B), result_dict)
"""

print(mean_sum_rate.item())

print("end")
