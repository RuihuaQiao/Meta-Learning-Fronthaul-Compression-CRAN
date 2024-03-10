"""
Generate test UE locations and channel data
"""

import numpy as np
from funcs import *
from scipy.io import savemat, loadmat

test_size = 2000
M = 8
B = 7
B2Bdist = 0.15
center_radius = 0
RRH_height = 0.03

Ball = 19
Nc = 2
Nall = Ball * Nc
random_seed = 42
n_UELocSamples = 50

bg_noise = -169  # dBm/Hz
bandwidth = 20e6  # Hz
sigmaz2 = db2pow(bg_noise-30) * bandwidth
scaling_factor = sigmaz2 * 100

np.random.seed(random_seed)
torch.manual_seed(1)

UELocs_discrete, in_cell = gen_discrete_UEs(B2Bdist, center_radius, n_UELocSamples)  # user location grid (in_cell, )
UELocs = scheduling(UELocs_discrete, test_size, Nall)  # (bs, Nall)
# plot_cell_UEs(B2Bdist, plot_UE=True, UE_1_loc=UELocs[0, ...])
plot_cell_UEs(B2Bdist, plot_UE=True, UE_1_loc=UELocs)

dist_set = compute_dist_set(B2Bdist, UELocs, RRH_height, Nc)  # (bs, Ball, Nall)

### check minimum and maximum pathloss power
print(db2pow(124 + 27 * torch.log10(torch.max(dist_set))))
print(db2pow(124 + 27 * torch.log10(torch.min(dist_set))))

Theta_set = gen_Theta_set(UELocs, B2Bdist, Nc, B)  # (bs, Nall, B)
H_set, H_set_sorted = gen_channel(UELocs, dist_set, M, scaling_factor)


save_dict = {"UELocs": UELocs.cpu().numpy(),
             "dist_set": dist_set.cpu().numpy(),
             "Theta_set": Theta_set.cpu().numpy(),
             "H_set": H_set.cpu().numpy(),
             "H_set_sorted": H_set_sorted.cpu().numpy()}

savemat("test_bs{}_M{}_Ball{}_B{}_Nc{}_samples{}_seed{}.mat".
        format(test_size, M, Ball, B, Nc, n_UELocSamples, random_seed),
        save_dict)


print('saved')
