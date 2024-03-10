from funcs import *
import torch
import torch.optim as optim  # for gradient descent


@torch.enable_grad()
def autograd(W_np, H_set, Theta_set, sig2r):
    """
    :param W_np: dict, numpy array, Ball * (bs, K, M)
    # :param gamma_np: numpy array, (1,)
    :param H_set: tensor, (bs, Ball, M, Nall)
    :param Theta_set: tensor, (bs, Nall, B)
    :param sig2r: simaz2 / sigmax2
    # :param grad_gamma: bool
    :param grad_W: bool
    :return:
    """
    bs, Ball, M, Nall = H_set.shape

    ### convert W to tensor
    W_set = {}
    for bb in range(Ball):
        W_set[bb] = torch.tensor(W_np[bb], dtype=torch.float32, requires_grad=True, device=device)

    ### forward propagation
    Cn_set, Fn_bar_set = compute_Cn_Fnbar(H_set, W_set, Theta_set, sig2r)
    rate = compute_rate_complete(H_set, W_set, Theta_set,  Cn_set, Fn_bar_set, sig2r)
    ### autograd
    rate.backward()

    W_grad = {}
    for bb in range(Ball):
        W_grad[bb] = W_set[bb].grad.detach() * bs * Nall * np.log(2)
        # W_grad[bb] = torch.clamp(W_grad[bb], min=-30, max=30)

    # torch.cuda.empty_cache()

    return W_grad
