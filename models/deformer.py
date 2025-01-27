import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from .mlp import ImplicitNetwork


def broyden(g, x_init, J_inv_init, max_steps=50, cvg_thresh=1e-5, dvg_thresh=1, eps=1e-6):
    """Find roots of the given function g(x) = 0.
    This function is impleneted based on https://github.com/locuslab/deq.

    Tensor shape abbreviation:
        N: number of points
        D: space dimension
    Args:
        g (function): the function of which the roots are to be determined. shape: [N, D, 1]->[N, D, 1]
        x_init (tensor): initial value of the parameters. shape: [N, D, 1]
        J_inv_init (tensor): initial value of the inverse Jacobians. shape: [N, D, D]

        max_steps (int, optional): max number of iterations. Defaults to 50.
        cvg_thresh (float, optional): covergence threshold. Defaults to 1e-5.
        dvg_thresh (float, optional): divergence threshold. Defaults to 1.
        eps (float, optional): a small number added to the denominator to prevent numerical error. Defaults to 1e-6.

    Returns:
        result (tensor): root of the given function. shape: [N, D, 1]
        diff (tensor): corresponding loss. [N]
        valid_ids (tensor): identifiers of converged points. [N]
    """

    # initialization
    x = x_init.clone().detach()
    J_inv = J_inv_init.clone().detach()

    ids_val = torch.ones(x.shape[0]).bool()

    gx = g(x, mask=ids_val)
    update = -J_inv.bmm(gx)

    x_opt = x
    gx_norm_opt = torch.linalg.norm(gx.squeeze(-1), dim=-1)

    delta_gx = torch.zeros_like(gx)
    delta_x = torch.zeros_like(x)

    ids_val = torch.ones_like(gx_norm_opt).bool()

    for _ in range(max_steps):

        # update paramter values
        delta_x[ids_val] = update
        x[ids_val] += delta_x[ids_val]
        delta_gx[ids_val] = g(x, mask=ids_val) - gx[ids_val]
        gx[ids_val] += delta_gx[ids_val]

        # store values with minial loss
        gx_norm = torch.linalg.norm(gx.squeeze(-1), dim=-1)
        ids_opt = gx_norm < gx_norm_opt
        gx_norm_opt[ids_opt] = gx_norm.clone().detach()[ids_opt]
        x_opt[ids_opt] = x.clone().detach()[ids_opt]

        # exclude converged and diverged points from furture iterations
        ids_val = (gx_norm_opt > cvg_thresh) & (gx_norm < dvg_thresh)
        if ids_val.sum() <= 0:
            break

        # compute paramter update for next iter
        vT = (delta_x[ids_val]).transpose(-1, -2).bmm(J_inv[ids_val])
        a = delta_x[ids_val] - J_inv[ids_val].bmm(delta_gx[ids_val])
        b = vT.bmm(delta_gx[ids_val])
        b[b >= 0] += eps
        b[b < 0] -= eps
        u = a / b
        J_inv[ids_val] += u.bmm(vT)
        update = -J_inv[ids_val].bmm(gx[ids_val])

    return {'result': x_opt, 'diff': gx_norm_opt, 'valid_ids': gx_norm_opt < cvg_thresh}



def hierarchical_softmax(x, model_type):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    if model_type == 'smpl':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        
        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))
        
        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        if n_dim >21:
            prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
            prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

            prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
            prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))
            
    elif model_type == 'smplh':
        prob_all = torch.ones(n_batch * n_point, n_dim-3, device=x.device)

        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-3]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-3]]))
        
        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

        prob_all[:, [22, 25, 28, 31, 34]] = prob_all[:, [20]] * sigmoid(x[:, [n_dim-2]]) * softmax(x[:, [22, 25, 28, 31, 34]])
        prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid(x[:, [n_dim-2]]))
        
        prob_all[:, [23, 26, 29, 32, 35]] = prob_all[:, [22, 25, 28, 31, 34]] * (sigmoid(x[:, [23, 26, 29, 32, 35]]))
        prob_all[:, [22, 25, 28, 31, 34]] = prob_all[:, [22, 25, 28, 31, 34]] * (1- sigmoid(x[:, [23, 26, 29, 32, 35]]))

        prob_all[:, [24, 27, 30, 33, 36]] = prob_all[:, [23, 26, 29, 32, 35]] * (sigmoid(x[:, [24, 27, 30, 33, 36]]))
        prob_all[:, [23, 26, 29, 32, 35]] = prob_all[:, [23, 26, 29, 32, 35]] * (1- sigmoid(x[:, [24, 27, 30, 33, 36]]))

        prob_all[:, [37, 40, 43, 46, 49]] = prob_all[:, [21]]  * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [37, 40, 43, 46, 49]])
        prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [38, 41, 44, 47, 50]] = prob_all[:, [37, 40, 43, 46, 49]] * (sigmoid(x[:, [38, 41, 44, 47, 50]]))
        prob_all[:, [37, 40, 43, 46, 49]] = prob_all[:, [37, 40, 43, 46, 49]] * (1 - sigmoid(x[:, [38, 41, 44, 47, 50]]))

        prob_all[:, [39, 42, 45, 48, 51]] = prob_all[:, [38, 41, 44, 47, 50]] * (sigmoid(x[:, [39, 42, 45, 48, 51]]))
        prob_all[:, [38, 41, 44, 47, 50]] = prob_all[:, [38, 41, 44, 47, 50]] * (1 - sigmoid(x[:, [39, 42, 45, 48, 51]]))
    
    elif model_type == 'smplx':
        prob_all = torch.ones(n_batch * n_point, n_dim-4, device=x.device)

        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-4]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-4]]))

        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

        prob_all[:, [22, 23, 24]] = prob_all[:, [15]] * sigmoid(x[:, [n_dim-3]]) * softmax(x[:, [22, 23, 24]])
        prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid(x[:, [n_dim-3]]))

        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [20]] * sigmoid(x[:, [n_dim-2]]) * softmax(x[:, [25, 28, 31, 34, 37]])
        prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid(x[:, [n_dim-2]]))

        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [25, 28, 31, 34, 37]] * (sigmoid(x[:, [26, 29, 32, 35, 38]]))
        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [25, 28, 31, 34, 37]] * (1- sigmoid(x[:, [26, 29, 32, 35, 38]]))

        prob_all[:, [27, 30, 33, 36, 39]] = prob_all[:, [26, 29, 32, 35, 38]] * (sigmoid(x[:, [27, 30, 33, 36, 39]]))
        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [26, 29, 32, 35, 38]] * (1- sigmoid(x[:, [27, 30, 33, 36, 39]]))

        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [21]]  * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [40, 43, 46, 49, 52]])
        prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [40, 43, 46, 49, 52]] * (sigmoid(x[:, [41, 44, 47, 50, 53]]))
        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [40, 43, 46, 49, 52]] * (1 - sigmoid(x[:, [41, 44, 47, 50, 53]]))

        prob_all[:, [42, 45, 48, 51, 54]] = prob_all[:, [41, 44, 47, 50, 53]] * (sigmoid(x[:, [42, 45, 48, 51, 54]]))
        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [41, 44, 47, 50, 53]] * (1 - sigmoid(x[:, [42, 45, 48, 51, 54]]))
    
    elif model_type == 'mano':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        prob_all[:, [1, 4, 7, 10, 13]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 4, 7, 10, 13]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [2, 5, 8, 11, 14]] = prob_all[:, [1, 4, 7, 10, 13]] * (sigmoid(x[:, [2, 5, 8, 11, 14]]))
        prob_all[:, [1, 4, 7, 10, 13]] = prob_all[:, [1, 4, 7, 10, 13]] * (1 - sigmoid(x[:, [2, 5, 8, 11, 14]]))

        prob_all[:, [3, 6, 9, 12, 15]] = prob_all[:, [2, 5, 8, 11, 14]] * (sigmoid(x[:, [3, 6, 9, 12, 15]]))
        prob_all[:, [2, 5, 8, 11, 14]] = prob_all[:, [2, 5, 8, 11, 14]] * (1 - sigmoid(x[:, [3, 6, 9, 12, 15]]))
    
    elif model_type == 'flame':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))
    
    else:
        raise NotImplementedError
    
    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    
    return prob_all



class ForwardDeformer(nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, opt, **kwargs):
        super().__init__()

        self.opt = opt

        self.lbs_network = ImplicitNetwork(**self.opt['network'])

        self.soft_blend = 20

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]

    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        xc_init = self.init(xd, tfs)

        xc_opt, others = self.search(xd, xc_init, cond, tfs, eval_mode=eval_mode)

        if eval_mode:
            return xc_opt, others

        # compute correction term for implicit differentiation during training

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        # reshape to [B,?,D] for network query
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_opt = xc_opt.reshape((n_batch, n_point * n_init, n_dim))

        xd_opt = self.forward_skinning(xc_opt, cond, tfs)

        grad_inv = self.gradient(xc_opt, cond, tfs).inverse()

        correction = xd_opt - xd_opt.detach()
        correction = einsum("bnij,bnj->bni", -grad_inv.detach(), correction)

        # trick for implicit diff with autodiff:
        # xc = xc_opt + 0 and xc' = correction'
        xc = xc_opt + correction

        # reshape back to [B,N,I,D]
        xc = xc.reshape(xc_init.shape)

        return xc, others

    def init(self, xd, tfs):
        """Transform xd to canonical space for initialization

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_init (tensor): gradients. shape: [B, N, I, D]
        """
        n_batch, n_point, _ = xd.shape
        _, n_joint, _, _ = tfs.shape

        xc_init = []
        for i in self.init_bones:
            w = torch.zeros((n_batch, n_point, n_joint), device=xd.device)
            w[:, :, i] = 1
            xc_init.append(skinning(xd, w, tfs, inverse=True))

        xc_init = torch.stack(xc_init, dim=2)

        return xc_init

    def search(self, xd, xc_init, cond, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        # reshape to [B,?,D] for other functions
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_batch, n_point * n_init, n_dim)
        xd_tgt = xd.repeat_interleave(n_init, dim=1)

        # compute init jacobians
        if not eval_mode:
            J_inv_init = self.gradient(xc_init, cond, tfs).inverse()
        else:
            w = self.query_weights(xc_init, cond, mask=None)
            J_inv_init = einsum("bpn,bnij->bpij", w, tfs)[:, :, :3, :3].inverse()

        # reshape init to [?,D,...] for boryden
        xc_init = xc_init.reshape(-1, n_dim, 1)
        J_inv_init = J_inv_init.flatten(0, 1)

        # construct function for root finding
        def _func(xc_opt, mask=None):
            # reshape to [B,?,D] for other functions
            xc_opt = xc_opt.reshape(n_batch, n_point * n_init, n_dim)
            xd_opt = self.forward_skinning(xc_opt, cond, tfs, mask=mask)
            error = xd_opt - xd_tgt
            # reshape to [?,D,1] for boryden
            error = error.flatten(0, 1)[mask].unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_batch, n_point, n_init, n_dim)
        result["valid_ids"] = result["valid_ids"].reshape(n_batch, n_point, n_init)

        return xc_opt, result

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        w = self.query_weights(xc, cond, mask=mask)
        xd = skinning(xc, w, tfs, inverse=False)
        return xd

    def query_weights(self, xc, cond, mask=None):
        """Get skinning weights in canonical space

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): valid indices. shape: [B, N]

        Returns:
            w (tensor): skinning weights. shape: [B, N, J]
        """

        w = self.lbs_network(xc, cond, mask)
        w = self.soft_blend * w

        if self.opt['softmax_mode'] == "hierarchical":
            w = hierarchical_softmax(w)
        else:
            w = F.softmax(w, dim=-1)

        return w

    def gradient(self, xc, cond, tfs):
        """Get gradients df/dx

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        xc.requires_grad_(True)

        xd = self.forward_skinning(xc, cond, tfs)

        grads = []
        for i in range(xd.shape[-1]):
            d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=xd,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, tfs)
        x_h = einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :3]
