import torch
import numpy as np
from ..convertors import F2S
import multiprocessing
import math

def mux_selector(inputs):
    no_channels, seq_len = inputs.shape
    random_idx = torch.randint(low=0, high=no_channels, size=(seq_len,))
    return inputs[random_idx, torch.arange(seq_len)]

def mul(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return torch.logical_not(torch.logical_xor(x,y))

def scaled_add(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    x, y = torch.broadcast_tensors(x, y)

    select_x = F2S(seq_len=x.shape[-1], mode='unipolar').generate(torch.full(size=x.shape[:-1], fill_value=0.5))
    select_y = torch.logical_not(select_x)

    out = torch.logical_or(torch.logical_and(select_x, x), torch.logical_and(select_y, y))

    return out

def mean(*inputs):
    return mux_selector(torch.stack(inputs, dim=0))

def scaled_dot(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return mux_selector(mul(x,y))


def scaled_matmul(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    x_shape, y_shape = x.shape, y.shape
    seq_len = x_shape[-1]
    assert seq_len == y_shape[-1] and x_shape[-2] == y_shape[0]
    x_reshape = x.view(-1,x_shape[-2],seq_len)
    y_reshape = y.view(y_shape[0], -1, seq_len)
    
    out = torch.zeros(x_reshape.shape[0], y_reshape.shape[1],seq_len)
    out_shape = x.shape[:-2] + y.shape[1:-1] + (seq_len,)


    for i, x_slice in enumerate(x_reshape):
        for j, y_slice in enumerate(y_reshape.transpose(0,1)):
            out[i,j] = scaled_dot(x_slice, y_slice)
    out = out.view(out_shape)
    return out



#tanh(xN/2)
def tanh_fsm(x: torch.tensor, N) -> torch.tensor:
    pass

def counter(x):
    return torch.sum(x, dim=-2, dtype=torch.int16)


class ap_counter():
    def __init__(self, in_features=16, num_au_layers=1):
        self.num_au_layers = num_au_layers
        self.in_features = int(math.ceil(in_features/4**num_au_layers) * 4**num_au_layers)

    def approx_unit(self, x: torch.tensor) -> torch.tensor:
        seq_len = x.shape[-1]
        assert x.shape[-2] <= self.in_features
        if x.shape[-2] < self.in_features:
            x = torch.cat([x, torch.zeros(x.shape[:-2]+(self.in_features-x.shape[-2],seq_len,), dtype=torch.bool)],dim=-2)

        num_units = int(self.in_features/4)

        grp1,grp2,grp3,grp4 = x[...,:num_units,:], x[...,num_units: 2*num_units,:], x[...,2*num_units: 3*num_units,:], x[...,3*num_units: 4*num_units,:]
        
        out = torch.concat([torch.logical_or(grp3,grp4), torch.logical_and(grp1,grp2)], dim=-2)
        
        return out
    
    def counter(self, x):
        return torch.sum(x, dim=-2, dtype=torch.int16)
    
    def __call__(self, x):
        for i in range(self.num_au_layers):
            print(x.shape)
            x = self.approx_unit(x)
            print(x.shape)
        return 2**self.num_au_layers*self.counter(x)
 

def get_btanh_scalar(in_features, num_states):
    q = 1.835*(2*in_features)**(-0.5552)
    return (1-q)/(2*(in_features-1))*(num_states-2*in_features)+1

def get_btanh_states(in_features, scalar):
    q = 1.835*(2*in_features)**(-0.5552)
    r_prime = 2*in_features + 2*(scalar-1)*(in_features-1)/(1-q)
    return int(round(r_prime/2.)*2)

#tanh(1/s * sum(inputs))
def apc_btanh(inputs, r: int, num_au_layers=1):
    in_features, seq_len = inputs.shape[-2], inputs.shape[-1]
    s_max = r-1
    s_half = (r-1)/2


    s = torch.full(inputs.shape[:-2], s_half, dtype=torch.int16)
    v = ap_counter(in_features=in_features, num_au_layers=num_au_layers)(inputs)*2 - in_features

    out = torch.zeros(inputs.shape[:-2]+(seq_len,), dtype=torch.bool)

    for i in range(seq_len):
        s = s + v[...,i]
        overflow_idx = s > s_max
        underflow_idx = s < 0
        s[overflow_idx] = s_max
        s[underflow_idx] = 0
        out[..., i] = s > s_half
    return out

