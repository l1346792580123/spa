import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class ColorMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, weight_norm=True):
        super().__init__()
        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
           
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

class ColorAE(nn.Module):
    def __init__(self, latent_feature, width, depth, if_vae=True):
        super().__init__()
        self.if_vae = if_vae
        self.latent_feature = latent_feature
        if if_vae:
            self.encoder = ColorMLP(3, latent_feature*2, width, depth//2)
        else:
            self.encoder = ColorMLP(3, latent_feature, width, depth//2)
        self.decoder = ColorMLP(latent_feature, 3, width, depth//2)
        
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        if self.if_vae:
            tmp = self.encoder(input)
            mu, logvar = torch.split(tmp, self.latent_feature, dim=-1)
            latent = self.reparameterize(mu, logvar)
        else:
            latent = self.encoder(input)
        x = self.decoder(latent)
        x = self.sigmoid(x)
        if self.if_vae:
            return x, mu, logvar
        else:
            return x, latent

""" MLP for neural implicit shapes. The code is based on https://github.com/lioryariv/idr with adaption. """
class ImplicitNetwork(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        geometric_init=True,
        bias=1.0,
        weight_norm=True,
        multires=0,
        skip_layer=[],
        cond_layer=[],
        cond_dim=69,
        dim_cond_embed=-1,
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.cond_layer = cond_layer
        self.cond_dim = cond_dim

        self.dim_cond_embed = dim_cond_embed
        if dim_cond_embed > 0:
            self.lin_p0 = torch.nn.Linear(self.cond_dim, dim_cond_embed)
            self.cond_dim = dim_cond_embed

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in self.cond_layer:
                lin = torch.nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)

    def forward(self, input, cond, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """


        n_batch, n_point, n_dim = input.shape

        if n_batch * n_point == 0:
            return input

        # reshape to [N,?]
        input = input.reshape(n_batch * n_point, n_dim)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        if len(self.cond_layer):
            n_batch, n_cond = cond.shape
            input_cond = cond.unsqueeze(1).expand(n_batch, n_point, n_cond)
            input_cond = input_cond.reshape(n_batch * n_point, n_cond)

            if mask is not None:
                input_cond = input_cond[mask]

            if self.dim_cond_embed > 0:
                input_cond = self.lin_p0(input_cond)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        return x_full.reshape(n_batch, n_point, -1)





class ColorNet(nn.Module):
    def __init__(self, d_in, width, depth, weight_norm=True, skip_layer=[]):
        super().__init__()
        dims = [d_in] + [width] * depth + [3]
        self.num_layers = len(dims)

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
           
            if l in self.skip_layer:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        x = self.sigmoid(x)

        return x