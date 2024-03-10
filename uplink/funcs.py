import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import torch
from torch import linalg as LA
import warnings
from scipy.io import savemat
from scipy.io import loadmat


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pow2db(y):
    return 10 * np.log10(y)


def db2pow(ydb):
    return 10 ** (ydb / 10)


def gen_discrete_UEs(B2Bdist, center_radius, n_UELocSamples):
    """
    generate discrete UE location grid
    :param center_radius: schedule UEs in the cell edge
    :param B2Bdist:
    :param n_UELocSamples: num of samples in x & y directions.
    :return: (in_cell, ) discrete grid of relative UE loctions in a cell
    """

    def is_in_cell(point, center_to_edge, center_radius):
        x_abs = torch.abs(point.real)
        y_abs = torch.abs(point.imag)
        center_to_vertex = 2 / np.sqrt(3) * center_to_edge

        if y_abs >= center_to_edge or x_abs >= center_to_vertex or torch.sqrt(x_abs ** 2 + y_abs ** 2) < center_radius:
            return False
        elif x_abs <= center_to_vertex / 2:
            return True
        else:
            nearby_center = np.sqrt(3) * center_to_edge + 1j * center_to_edge
            dist_point_center = torch.abs(point) ** 2
            dist_point_nearby_center = (x_abs - nearby_center.real) ** 2 + (y_abs - nearby_center.imag) ** 2
            if dist_point_center < dist_point_nearby_center:
                return True
            else:
                return False

    randLocs_standard = (torch.arange(n_UELocSamples) + 1) / (n_UELocSamples + 1)  # eg. 0.01-0.99
    randLocs_x = B2Bdist * (randLocs_standard * np.sqrt(3) * 2 / 3 - np.sqrt(3) / 3)  # (n, )
    randLocs_y = B2Bdist * (randLocs_standard - 0.5)  # (n, )
    randLocs = (randLocs_x.view(-1, 1) + 1j * randLocs_y.view(1, -1)).view(n_UELocSamples ** 2)  # (n^2, )

    UELocs_discrete = (torch.zeros(n_UELocSamples ** 2) + 1j * torch.zeros(n_UELocSamples ** 2)).to(device)
    in_cell = 0
    for ii in range(n_UELocSamples ** 2):
        if is_in_cell(randLocs[ii], B2Bdist / 2, center_radius):
            UELocs_discrete[in_cell] = randLocs[ii]

            in_cell = in_cell + 1

    UELocs_discrete = UELocs_discrete[torch.nonzero(UELocs_discrete)].squeeze()  # (in_cell, )

    return UELocs_discrete, in_cell


def scheduling(UELocs_discrete, bs, Nall):
    """
    randomly schedule UEs from discrete grid in a timeslot
    :param UELocs_discrete: (in_cell, ) discrete grid of relative UE loctions in a cell
    :param bs:
    :param Nall:
    :return: (bs, Nall) relative UE locations in a scheduling timeslot
    """
    in_cell = UELocs_discrete.shape[0]
    # UELocs = torch.zeros((bs, Nall))
    idx = torch.randint(low=0, high=in_cell, size=(bs * Nall,)).to(device)
    UELocs = torch.gather(input=UELocs_discrete, dim=-1, index=idx)
    UELocs = UELocs.view(bs, Nall)
    return UELocs


def plot_cell_UEs(B2Bdist, plot_UE, UE_1_loc):
    """
    19 cell -- plot RRH locations, cell grids and UE locations for one realization
    :param B2Bdist:
    :param plot_UE: bool
    :param UE_1_loc: (Ball, )
    :return:
    """
    BSLocs = 0.5 * B2Bdist * np.array(
        [0, np.sqrt(3) + 1j, 2 * 1j, -np.sqrt(3) + 1j, -np.sqrt(3) - 1j, -2 * 1j, np.sqrt(3) - 1j,
         np.sqrt(12), np.sqrt(12) + 2j, np.sqrt(3) + 3j, 4j, -np.sqrt(3) + 3j, -np.sqrt(12) + 2j,
         -np.sqrt(12), -np.sqrt(12) - 2j, -np.sqrt(3) - 3j, -4j, np.sqrt(3) - 3j, np.sqrt(12) - 2j])  # 19 cells

    BSLocs_x = np.real(BSLocs)
    BSLocs_y = np.imag(BSLocs)

    labels = np.arange(19)

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    for x, y, l in zip(BSLocs_x, BSLocs_y, labels):
        hex_cell = RegularPolygon((x, y),
                                  numVertices=6,
                                  radius=np.sqrt(3) / 3 * B2Bdist,
                                  orientation=np.pi / 6,  # rad
                                  alpha=1,
                                  facecolor='w',
                                  edgecolor='k')

        ax.add_patch(hex_cell)
        ax.text(x, y, str(l), ha='center', va='center', size=10)

    # add scatter points in hexagon centers
    ax.scatter(BSLocs_x, BSLocs_y, color='k', s=1)

    if plot_UE:
        UE_1_loc = UE_1_loc.reshape(19, -1)
        for cc in range(19):
            loc = BSLocs[cc] + UE_1_loc[cc, :].cpu().numpy()
            ax.scatter(loc.real, loc.imag, c='tab:blue', marker='*')

    plt.savefig("plot_result/topology_19cell.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def gen_Theta_set(UELocs, B2Bdist, Nc, B):
    """
    Generate the set of B RRHs closest to each user
    :param UELocs: (bs, Nall)
    :param B2Bdist:
    :param Nc:
    :param B: size of RRH cluster for each UE
    :return: (bs, Nall, B)
    """
    bs, Nall = UELocs.shape
    Ball = 19
    BSLocs = 0.5 * B2Bdist * np.array(
        [0, np.sqrt(3) + 1j, 2 * 1j, -np.sqrt(3) + 1j, -np.sqrt(3) - 1j, -2 * 1j, np.sqrt(3) - 1j,
         np.sqrt(12), np.sqrt(12) + 2j, np.sqrt(3) + 3j, 4j, -np.sqrt(3) + 3j, -np.sqrt(12) + 2j,
         -np.sqrt(12), -np.sqrt(12) - 2j, -np.sqrt(3) - 3j, -4j, np.sqrt(3) - 3j, np.sqrt(12) - 2j])  # 19 cells

    dist_set = torch.zeros(bs, Nall, Ball).to(device)
    for ii in range(Nall):
        for bb in range(Ball):
            # distance between the i-th user to b-th relative RRH
            dist_set[:, ii, bb] = torch.abs(UELocs[:, ii] - BSLocs[bb])
    dist_sorted, dist_sort_idx = torch.sort(dist_set, dim=-1, stable=True)

    Theta_relative = dist_sort_idx[:, :, 0:B]
    Theta_real = torch.zeros_like(Theta_relative).to(device)
    wrap_cell_order = wrap_around_7cell()
    for ii in range(Nall):
        cell_idx = ii // Nc  # which cell this UE is in
        for bb in range(B):
            wrap_order_cell_ii = wrap_cell_order[cell_idx, :]
            Theta_real[:, ii, bb] = torch.gather(wrap_order_cell_ii, -1, Theta_relative[:, ii, bb])
    Theta_real, _ = torch.sort(Theta_real, dim=-1)
    return Theta_real


def wrap_around_7cell():
    wrap_cell_order = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                    [1, 8, 9, 2, 0, 6, 7, 13, 12, 16, 15, 10, 11, 3, 4, 5, 17, 18, 14],
                                    [2, 9, 10, 11, 3, 0, 1, 8, 16, 15, 14, 18, 17, 12, 13, 4, 5, 6, 7],
                                    [3, 2, 11, 12, 13, 4, 0, 1, 9, 10, 18, 17, 16, 8, 7, 14, 15, 5, 6],
                                    [4, 0, 3, 13, 14, 15, 5, 6, 1, 2, 11, 12, 8, 7, 18, 10, 9, 16, 17],
                                    [5, 6, 0, 4, 15, 16, 17, 18, 7, 1, 2, 3, 13, 14, 10, 9, 8, 12, 11],
                                    [6, 7, 1, 0, 5, 17, 18, 14, 13, 8, 9, 2, 3, 4, 15, 16, 12, 11, 10],
                                    [7, 13, 8, 1, 6, 18, 14, 4, 3, 12, 16, 9, 2, 0, 5, 17, 11, 10, 15],
                                    [8, 12, 16, 9, 1, 7, 13, 3, 11, 17, 5, 15, 10, 2, 0, 6, 18, 14, 4],
                                    [9, 16, 15, 10, 2, 1, 8, 12, 17, 5, 4, 14, 18, 11, 3, 0, 6, 7, 13],
                                    [10, 15, 14, 18, 11, 2, 9, 16, 5, 4, 13, 7, 6, 17, 12, 3, 0, 1, 8],
                                    [11, 10, 18, 17, 12, 3, 2, 9, 15, 14, 7, 6, 5, 16, 8, 13, 4, 0, 1],
                                    [12, 11, 17, 16, 8, 13, 3, 2, 10, 18, 6, 5, 15, 9, 1, 7, 14, 4, 0],
                                    [13, 3, 12, 8, 7, 14, 4, 0, 2, 11, 17, 16, 9, 1, 6, 18, 10, 15, 5],
                                    [14, 4, 13, 7, 18, 10, 15, 5, 0, 3, 12, 8, 1, 6, 17, 11, 2, 9, 16],
                                    [15, 5, 4, 14, 10, 9, 16, 17, 6, 0, 3, 13, 7, 18, 11, 2, 1, 8, 12],
                                    [16, 17, 5, 15, 9, 8, 12, 11, 18, 6, 0, 4, 14, 10, 2, 1, 7, 13, 3],
                                    [17, 18, 6, 5, 16, 12, 11, 10, 14, 7, 1, 0, 4, 15, 9, 8, 13, 3, 2],
                                    [18, 14, 7, 6, 17, 11, 10, 15, 4, 13, 8, 1, 0, 5, 16, 12, 3, 2, 9]]).to(device)

    return wrap_cell_order


def gramschmidt(vv):
    def projection(u, v):
        return (v * u).sum(dim=-1, keepdims=True) / (u * u).sum(dim=-1, keepdims=True) * u

    n_row = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0, :] = vv[:, 0, :].clone()
    for kk in range(1, n_row):
        vk = vv[:, kk, :].clone()
        uk = torch.zeros_like(vk, device=vk.device)
        for jj in range(0, kk):
            uj = uu[:, jj, :].clone()
            uk = uk + projection(uj, vk)
        uu[:, kk] = vk - uk
    uu = uu / LA.vector_norm(uu, dim=-1, keepdims=True)
    return uu


def get_Wn_bar(Theta_n, W_dict):
    """
    Based on Theta_n, select some Wb's from W_dict to get Wn_bar
    :param Theta_n: (bs, B)
    :param W_dict: dict, Ball entries, each (bs, K, M)
    :return: Wn_bar: # (bs, BK, BM)
    """
    Ball = 19
    B = Theta_n.shape[-1]
    bs, K, M = W_dict[0].shape
    W_set = torch.cat([W_dict[bb].view(bs, 1, K, M) for bb in range(Ball)], dim=1)  # (bs, Ball, K, M)
    Theta_n_gather_idx = Theta_n.view(bs, B, 1, 1).expand(bs, B, K, M)  # used in torch.gather
    W_set_gathered = torch.gather(input=W_set, dim=1, index=Theta_n_gather_idx)  # (bs, B, K, M)
    Wn_bar = torch.zeros((bs, B * K, B * M)).to(device)
    for bb in range(B):
        Wn_bar[:, (bb * K):((bb + 1) * K), (bb * M):((bb + 1) * M)] = W_set_gathered[:, bb, :, :]  # (bs, K, M)

    return Wn_bar  # (bs, BK, BM)


def compute_dist_set(B2Bdist, UELocs, RRH_height, Nc):
    """
    Compute dist between all RRHs and all UEs
    :param B2Bdist:
    :param UELocs: (bs, Nall) UE relative location. the relative location of users to its centering RRH
    :param RRH_height:
    :param Nc:
    :return: (bs, Ball, Nall)
    """
    Ball = 19
    BSLocs = 0.5 * B2Bdist * np.array(
        [0, np.sqrt(3) + 1j, 2 * 1j, -np.sqrt(3) + 1j, -np.sqrt(3) - 1j, -2 * 1j, np.sqrt(3) - 1j,
         np.sqrt(12), np.sqrt(12) + 2j, np.sqrt(3) + 3j, 4j, -np.sqrt(3) + 3j, -np.sqrt(12) + 2j,
         -np.sqrt(12), -np.sqrt(12) - 2j, -np.sqrt(3) - 3j, -4j, np.sqrt(3) - 3j, np.sqrt(12) - 2j])  # 19 cells
    d = B2Bdist / 2
    rrh_shift = torch.tensor(
        [0, -3 * np.sqrt(3) * d + 7j * d, 3 * np.sqrt(3) * d - 7j * d, -2 * np.sqrt(3) * d - 8j * d,
         2 * np.sqrt(3) * d + 8j * d, -5 * np.sqrt(3) * d - 1j * d, 5 * np.sqrt(3) * d + 1j * d]).to(device)
    bs, Nall = UELocs.shape

    dist_set = torch.zeros((bs, Ball, Nall)).to(device)
    for bb in range(Ball):
        for ii in range(Nall):
            UE_i_rrh_idx = ii // Nc  # which cell UE ii is in
            Loc_UE_i = UELocs[:, ii] + BSLocs[UE_i_rrh_idx]  # (bs, )
            Loc_rrh_b_possible = BSLocs[bb] + rrh_shift  # (7, ), 7 posssible RRH locations
            dist_7 = torch.abs(Loc_UE_i.view(-1, 1) - Loc_rrh_b_possible)  # (bs, 7)
            d_bi_horizontal = torch.amin(dist_7, dim=-1, keepdim=True)  # (bs, 1)
            BS_height = torch.tensor(RRH_height).to(device)
            d_bi = torch.sqrt(d_bi_horizontal ** 2 + BS_height ** 2)
            dist_set[:, bb, ii] = d_bi.squeeze()

    return dist_set


def gen_channel(UELocs, dist_set, M, scaling_factor):
    """
    generate channel matrix & sort
    :param UELocs: (bs, Nall), UE relative location
    :param dist_set: (bs, Ball, Nall), dist between all RRHs and all UEs
    :param M:
    :param scaling_factor:
    :return: (bs, Ball, M, Nall), original & sorted channel matrix
    """
    bs, Nall = UELocs.shape
    Ball = dist_set.shape[1]

    beta_set = torch.zeros(bs, Ball, Nall).to(device)
    Hc = torch.zeros((bs, Ball, M, Nall)).to(device)
    # shad_set = torch.normal(0, 2.9, size=(bs, Ball, Nall)).to(device)

    for ii in range(Nall):
        for bb in range(Ball):
            d_bi = dist_set[:, bb, ii].view(-1, 1)  # (bs, 1)
            beta_bi = 180 - 147.5 + 20 * np.log10(2.9) + 29 * torch.log10(d_bi * 1000)
            Hc[:, bb, :, ii] = torch.sqrt(db2pow(-beta_bi)) / np.sqrt(scaling_factor) * torch.randn(bs, M).to(device)
            beta_set[:, bb, ii] = beta_bi.squeeze()

    ### sorting
    _, dist_order = torch.sort(beta_set, dim=-1)  # (bs, Ball, Nall)
    dist_order_gather_idx = dist_order.view(bs, Ball, 1, Nall).expand(bs, Ball, M, Nall)
    Hc_sorted = torch.gather(input=Hc, dim=-1, index=dist_order_gather_idx)  # (bs, Ball, M, Nall)

    return Hc, Hc_sorted  # both: (bs, Ball, M, Nall)


def func_EVD(H_set, sigma2_zx_ratio, K):
    W_EVD_set = {}
    bs, Ball, M, _ = H_set.shape
    for bb in range(19):
        Hb = H_set[:, bb, :, :]
        diag_M = torch.diag_embed(torch.ones((bs, M)).to(device))

        Sigma_ybyb = Hb @ Hb.mH + sigma2_zx_ratio * diag_M  # (bs, M, M)
        U, S, V = torch.svd(Sigma_ybyb)
        W_bb = U[:, :, 0:K].mH  # (bs, K, M)
        # W_bb = torch.randn(batch_size, K, M).to(device)
        W_EVD_set[bb] = gramschmidt(W_bb)
    return W_EVD_set


def compute_rate(H_set, W_set, Theta_set, sig2r):
    """
    compute averaged sum rate
    :param H_set: (bs, Ball, M, Nall)
    :param W_set: dict, Ball * (bs, K, M)
    :param Theta_set: (bs, Nall, B)
    :param sig2r: simaz2 / sigmax2
    :return: mean per UE rate (scalar, averaged over batch)
    """
    bs, Ball, M, Nall = H_set.shape
    K = W_set[0].shape[1]
    B = Theta_set.shape[-1]
    Nc = Nall // Ball
    diag_BK = torch.diag_embed(torch.ones((bs, B * K)).to(device))

    sum_rate = torch.zeros(bs, 1).to(device)
    stats_set = np.zeros((Nall, 2))

    for nn in range(Nall):
        Theta_n = Theta_set[:, nn, :]  # (bs, B)
        Theta_n_gather_idx = Theta_n.view(bs, B, 1, 1).expand(bs, B, M, Nall)  # used in torch.gather

        Hn_set = torch.gather(input=H_set, dim=1, index=Theta_n_gather_idx)  # (bs, B, M, Nall)
        Hn_bar = torch.cat([Hn_set[:, bb, :, :] for bb in range(B)], dim=1)  # (bs, BM, Nall)
        Wn_bar = get_Wn_bar(Theta_n, W_set)  # (bs, BK, BM)

        Fn_bar = Wn_bar @ Hn_bar  # (bs, BK, Nall)
        fnn = Fn_bar[..., nn].view(bs, B * K, 1)  # (bs, BK, 1)

        # print(torch.max(LA.cond(Fn_bar @ Fn_bar.mH + sig2r * Wn_bar @ Wn_bar.mH)).item())
        # print(torch.min(LA.matrix_rank(Fn_bar @ Fn_bar.mH + sig2r * Wn_bar @ Wn_bar.mH)).item())
        # if torch.min(LA.matrix_rank(Fn_bar @ Fn_bar.mH + sig2r * Wn_bar @ Wn_bar.mH)).item() < B * K:
        #     warnings.warn("matrix low rank")

        # print(torch.max(LA.cond(Fn_bar @ Fn_bar.mH + sig2r * diag_BK)))

        # Cn = LA.inv(Fn_bar @ Fn_bar.mH + sig2r * Wn_bar @ Wn_bar.mH) @ fnn  # (bs, BK, 1)
        Cn = LA.inv(Fn_bar @ Fn_bar.mH + sig2r * diag_BK) @  fnn  # (bs, BK, 1)
        # Cn = LA.inv(Fn_bar @ Fn_bar.mH) @  fnn  # (bs, BK, 1)

        Cn = Cn / LA.vector_norm(Cn, dim=-2, keepdims=True)  # (bs, BK, 1)

        numerator = (torch.abs(fnn.mH @ Cn) ** 2).view(bs, 1)  # (bs, 1)
        power_all = 0
        for jj in range(Nall):
            fjn = Fn_bar[..., jj].view(bs, B * K, 1)  # (bs, BK, 1)
            power_all = power_all + (torch.abs(fjn.mH @ Cn) ** 2).view(bs, 1)
        if torch.amax(power_all < numerator):  # throw warning when at least one sample satisfies condition
            warnings.warn("numerical issue in computing power_all")
        denominator = power_all - numerator + sig2r * torch.ones(bs, 1).to(device)  # (bs, 1)
        # denominator = power_all - numerator + sig2r * LA.vector_norm((Wn_bar.mH @ Cn).squeeze(), dim=-1).view(bs, 1) ** 2  # (bs, 1)
        Rn = torch.log(1 + numerator / denominator / db2pow(6)) / np.log(2)
        sum_rate = sum_rate + Rn  # (bs, 1)
        stats_set[nn, :] = np.array([torch.mean(numerator).item(), torch.mean(power_all - numerator).item()])

    # stats_mean = np.mean(stats_set, axis=0)  # (3,)
    # print(stats_mean)

    per_UE_rate = sum_rate / Nall
    mean_per_UE_rate = torch.mean(per_UE_rate)  # average over samples in a batch

    ##### save & plot cdf curve
    # sorted_rate = (torch.sort(per_UE_rate.squeeze())[0]).cpu().detach().numpy()  # x-axis
    # plt.plot(sorted_rate, np.arange(bs) / bs)
    # plt.show()
    # result_dict = {'sorted_rate': sorted_rate}
    # folder_path = "./plot_result/plot_cdf/"
    # method = "GlobalGD"
    # savemat(folder_path + method + '_cell{}_M{}_Nc{}_B{}_K{}.mat'.format(Ball, M, Nc, B, K), result_dict)

    return mean_per_UE_rate


def bits_allocation(S_vec, Q_bits, K):
    bs, Ball, _ = S_vec.shape
    # S_vec = S_vec - sig2r  # only signal power left
    total_bits = Q_bits
    bits_allocate = torch.zeros((bs, Ball, K)).to(device)  # (bs, Ball, K)
    # for ss in range(bs):
    for bb in range(Ball):
        quant_noise = S_vec[:, bb, :].clone().detach()  # (bs, K)
        for ii in range(total_bits):
            jj = torch.argmax(quant_noise, dim=-1).unsqueeze(-1)  # (bs, 1) from 0 - K-1
            jj_scatter = torch.scatter(input=torch.zeros((bs, K), device=device),
                                       src=torch.ones((bs, 1), device=device),
                                       index=jj, dim=-1)
            bits_allocate[:, bb, :] = bits_allocate[:, bb, :] + jj_scatter
            quant_noise = torch.scatter(input=quant_noise,
                                        src=torch.gather(input=quant_noise/4, index=jj, dim=-1),
                                        index=jj, dim=-1)
            # quant_noise[jj] = quant_noise[jj] / 4
    return bits_allocate  # (bs, Ball, K)


def compute_rate_quant(H_set, W_set, Theta_set, sig2r, quant_bits, gamma, if_test):
    """
    compute per user rate considering uniform quantization
    :param H_set: (bs, Ball, M, Nall)
    :param W_set: dict, Ball * (bs, K, M)
    :param Theta_set: (bs, Nall, B)
    :param sig2r: simaz2 / sigmax2
    :param quant_bits: (bs, Ball, K), bits allocated for each dimension
    :param gamma: scaling factor
    :param if_test
    :return: per user rate considering uniform quantization
    """
    bs, Ball, M, Nall = H_set.shape
    K = W_set[0].shape[1]
    B = Theta_set.shape[-1]
    Nc = Nall // Ball
    diag_BK = torch.diag_embed(torch.ones((bs, B * K)).to(device))

    ### compute quantization noise varaince per channel realization (per RRH per dim)
    S_set = torch.zeros((bs, Ball, K)).to(device)
    for bb in range(Ball):
        Fb = W_set[bb] @ H_set[:, bb, :, :]  # (bs, K, N)
        Sigma_vb = Fb @ Fb.mH  # (bs, K, K)
        _, S, _ = LA.svd(Sigma_vb)  # S (bs, K), U (bs, K, K)
        S_set[:, bb, :] = S

    clip_range = gamma * torch.sqrt(S_set)  # (bs, Ball, K)
    delta_q = 2 * clip_range / (2 ** quant_bits)  # (bs, Ball, K)
    S_quant = delta_q ** 2 / 12  # (bs, Ball, K)

    if if_test:
        x = torch.normal(0, 1, size=(bs, Nall, 1)).to(device)
        mask_set = torch.zeros((bs, Ball, K), dtype=torch.bool).to(device)
        for bb in range(Ball):
            z_b = torch.normal(0, sig2r, size=(bs, M, 1)).to(device)
            y_b = H_set[:, bb, :, :] @ x + z_b  # (bs, M, 1)
            v_b = (W_set[bb] @ y_b).squeeze()  # (bs, K)
            clip_b = clip_range[:, bb, :]  # (bs, K)
            tmp_clip = - clip_b + torch.relu(v_b + clip_b) - torch.relu(v_b - clip_b)  # (bs, K)
            delta_q_b = delta_q[:, bb, :]
            v_b_Q = - clip_b + torch.floor(torch.abs(tmp_clip - (-clip_b)) / delta_q_b) * delta_q_b + delta_q_b / 2
            err_v_b = v_b_Q - v_b
            mask_set[:, bb, :] = torch.gt(torch.abs(err_v_b.squeeze()), (delta_q_b / 2))  # entry=1 if out of clipping

    # diag_BK = torch.diag_embed(torch.ones((bs, B * K)).to(device))
    sum_rate = torch.zeros(bs, 1).to(device)

    stats_set = np.zeros((Nall, 3))

    for nn in range(Nall):
        Theta_n = Theta_set[:, nn, :]  # (bs, B)
        Theta_n_gather_idx = Theta_n.view(bs, B, 1, 1).expand(bs, B, M, Nall)  # used in torch.gather
        Hn_set = torch.gather(input=H_set, dim=1, index=Theta_n_gather_idx)  # (bs, B, M, Nall)
        Hn_bar = torch.cat([Hn_set[:, bb, :, :] for bb in range(B)], dim=1)  # (bs, BM, Nall)
        Wn_bar = get_Wn_bar(Theta_n, W_set)  # (bs, BK, BM)

        Theta_n_gather_idx = Theta_n.view(bs, B, 1).expand(bs, B, K)  # used in torch.gather
        # Sn_quant_set = torch.gather(input=S_quant, dim=1, index=Theta_n_gather_idx)  # (bs, B, K)
        Sn_quant_set = torch.gather(input=S_quant.expand(bs, Ball, K), dim=1, index=Theta_n_gather_idx)  # (bs, B, K)
        Sn_quant_bar = torch.cat([Sn_quant_set[:, bb, :] for bb in range(B)], dim=1)  # (bs, BK)
        Sn_quant_bar = torch.diag_embed(Sn_quant_bar)  # (bs, BK, BK)

        Fn_bar = Wn_bar @ Hn_bar  # (bs, BK, Nall)
        fnn = Fn_bar[..., nn].view(bs, B * K, 1)  # (bs, BK, 1)

        Cn = LA.inv(Fn_bar @ Fn_bar.mH + sig2r * diag_BK + Sn_quant_bar) @ fnn  # (bs, BK, 1)
        Cn = Cn / LA.vector_norm(Cn, dim=-2, keepdims=True)  # (bs, BK, 1)
        if if_test:
            ### delete overload entries
            mask_set_bar = torch.gather(input=mask_set, dim=1, index=Theta_n_gather_idx)  # (bs, B, K)
            Cn_mask = Cn.masked_fill(mask_set_bar.view(bs, B*K, 1), 0)  # (bs, BK, 1), set overload entries to 0
            Cn = Cn_mask

        numerator = (torch.abs(fnn.mH @ Cn) ** 2).view(bs, 1)  # (bs, 1)
        # power_all = torch.sum(torch.abs(Fn_bar.mH @ Cn)**2, dim=1)  # (bs, 1)
        power_all = 0
        for jj in range(Nall):
            fjn = Fn_bar[..., jj].view(bs, B * K, 1)  # (bs, BK, 1)
            power_all = power_all + (torch.abs(fjn.mH @ Cn) ** 2).view(bs, 1)
        if torch.amax(power_all < numerator):  # throw warning when at least one sample satisfies condition
            warnings.warn("numerical issue in computing power_all")
        # pow_noise = LA.vector_norm(Wn_bar.mH @ Cn, dim=1).view(bs, 1) ** 2  # (bs, 1)
        pow_quant = (Cn.mH @ Sn_quant_bar @ Cn).view(bs, 1)  # (bs, 1)
        denominator = power_all - numerator + sig2r * torch.ones(bs, 1).to(device) + pow_quant  # (bs, 1)
        Rn = torch.log(1 + numerator / denominator / db2pow(6)) / np.log(2)
        sum_rate = sum_rate + Rn  # (bs, 1)

        stats_set[nn, :] = np.array([torch.mean(numerator).item(), torch.mean(power_all - numerator).item(),
                                     torch.mean(pow_quant).item()])

    stats_mean = np.mean(stats_set, axis=0)  # (3,)
    # print(stats_mean)
    per_UE_rate = sum_rate / Nall
    mean_per_UE_rate = torch.mean(per_UE_rate)  # average over samples in a batch

    return mean_per_UE_rate
