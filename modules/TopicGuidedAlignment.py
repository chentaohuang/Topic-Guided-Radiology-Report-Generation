from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import math
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# ---- attention models for FDT
class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='sparsemax', pool_type='max'):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of FDT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        # activation
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature

        # map patch/text tokens to codebook (query) spaces
        # ---note that we donot use mapping for FDT

        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )

    def forward(self, ft, sd, mask=None, return_token_att=False):

        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [FDT_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing FDT.
        Returns:

        '''

        # map image/text token to query space
        q = self.q_map(ft)  # bacth, token_num, dim

        k = sd  # code_num, sd_dim
        k = k.unsqueeze(0)  # [1, code_num, sd_dim]
        k = k.transpose(2, 1)  # [1,sd_dim, sd_num]

        # -----calculate inner dot
        inner_dot = torch.matmul(q, k)  # [bacth, token_num, code_num]

        if return_token_att:  # cosine sim
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim)  # scale dot norm

        if mask is not None:  # mask paded tokens
            inf_mask = mask

            assert mask.shape == q.shape[:2]
            mask = (mask == 1) * 1  # 0 --> 1, inf --> 0
            inf_mask = (inf_mask == 0) * 1 * -1e6
            inner_dot = inner_dot * mask.unsqueeze(-1) + inf_mask.unsqueeze(-1) # sigmod(-inf) = 0, softmax(-inf) = 0


            if return_token_att:  # if has pad, return maksed
                token_att = inner_dot

        # temptural norm
        inner_dot = inner_dot / self.temperature  # [bacth, token_num, code_num]

        # pooling
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1)  # mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        # ----get attention weights
        att_weight = self.att_activation(inner_dot)  # normaliztion

        # ----calculate weighted sum of v
        # v = self.ln_v(ft) #map to v_space

        att_ft = att_weight @ sd  # [bacth, dictory_size] * [dictory_size, dim]  ---> [bacth, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)

        if return_token_att:
            return token_att, att_ft
        return att_weight, att_ft



class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input