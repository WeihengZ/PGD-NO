import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class Physics_Attention_seg(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.token_linear = nn.Linear(dim_head, dim_head, bias=False)

        self.cross_to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.cross_to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.cross_to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, seg_matrix, return_attention=False):
        '''
        x: (B, N, C)
        seg_matrix: (B, S, N), where S is number of physics tokens
        return_attention: If True, also return attention scores
        '''
        # B N C
        B, N, C = x.shape
        
        # compute the slice tokens
        query_x = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_token = torch.einsum("bsn,bhnc->bhsc", seg_matrix, x_mid)  # B H S C
        out_slice_token = self.token_linear(slice_token)

        # cross attention on the slice tokens and the data coordinates
        q_x = self.cross_to_q(query_x)
        k_slice_token = self.cross_to_k(out_slice_token)
        v_slice_token = self.cross_to_v(out_slice_token)
        
        if return_attention:
            # Compute attention scores manually
            dots = torch.matmul(q_x, k_slice_token.transpose(-1, -2)) * self.scale  # B H N S
            attn_coord_to_token = self.softmax(dots)  # B H N S (attention from coordinates to slice tokens)
            attn_coord_to_token = self.dropout(attn_coord_to_token)
            out_x = torch.matmul(attn_coord_to_token, v_slice_token)  # B H N C
        else:
            out_x = F.scaled_dot_product_attention(q_x, k_slice_token, v_slice_token)

        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        
        if return_attention:
            return self.to_out(out_x), attn_coord_to_token
        else:
            return self.to_out(out_x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_plus_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_seg(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, seg_matrix, return_attention=False):
        if return_attention:
            # When extracting attention, don't use checkpointing
            attn_output = self.Attn(self.ln_1(fx), seg_matrix, return_attention=True)
            out_x, attn_coord_to_token = attn_output
            fx = out_x + fx
        else:
            if self.training:
                fx = checkpoint(self.Attn, self.ln_1(fx), seg_matrix, use_reentrant=True) + fx
            else:
                fx = self.Attn(self.ln_1(fx), seg_matrix) + fx
        
        if self.training and not return_attention:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        
        if self.last_layer:
            result = self.mlp2(self.ln_3(fx))
            if return_attention:
                return result, attn_coord_to_token
            else:
                return result
        else:
            if return_attention:
                return fx, attn_coord_to_token
            else:
                return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(3, n_hidden)
        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=False)
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.Cp_decoder = nn.Linear(n_hidden, out_dim)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data, return_attention=False):
        x, seg_matrix = data

        fx, T = None, None
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]    # (B, N, C)
        
        all_attentions = []
        
        # geometry information processing
        for i, block in enumerate(self.blocks):
            if return_attention:
                fx, attn_coord_to_token = block(fx, seg_matrix, return_attention=True)
                all_attentions.append(attn_coord_to_token)
            else:
                fx = block(fx, seg_matrix)    # (B, N, F)
        
        fx = self.Cp_decoder(fx)

        if return_attention:
            return fx, all_attentions
        else:
            return fx
    
    def extract_attention_scores(self, data):
        """
        Extract attention scores between coordinates (query nodes) and slice tokens across all layers.
        
        Args:
            data: Tuple of (x, seg_matrix) where x is (B, N, F) and seg_matrix is (B, S, N)
            
        Returns:
            dict: Dictionary containing:
                - 'attention_scores': List of (B, H, N, S) numpy arrays, one per layer,
                  representing attention from coordinates (query nodes) to slice tokens
                  where B=batch, H=heads, N=num_query_nodes, S=num_tokens
        """
        self.eval()
        with torch.no_grad():
            _, all_attentions = self.forward(data, return_attention=True)
            
            # Convert to numpy arrays
            attention_scores = [attn.cpu().numpy() for attn in all_attentions]
            
            return {
                'attention_scores': attention_scores,  # List of (B, H, N, S) arrays
            }
