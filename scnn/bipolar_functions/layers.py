import torch
from operations import *
from ..convertors import F2S


class MUXScaled_Linear():
    def __init__(self, 
                 in_features,
                 out_features,
                 seq_len,
                 bias=False) -> None:
        super().__init__()

        self.down_scale = in_features
        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = bias
        self.weight = None
        self.bias = None

    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias']/self.in_features)

    def forward(self, inputs):
        assert self.weight is not None
        out = scaled_matmul(inputs, self.weight)

        out = scaled_add(out, self.bias)

        return out


class APCLinear():
    def __init__(self, 
                 in_features,
                 out_features,
                 seq_len,
                 btanh_scalar,
                 bias=False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = bias
        self.weight = None
        self.bias = None
        self.btanh_scalar =  btanh_scalar
        self.states = self.get_btanh_states()

    def get_btanh_states(self,):
        q = 1.835*(2*self.in_features)**(-0.5552)
        r_prime = 2*self.in_features + 2*(self.btanh_scalar-1)*(self.in_features-1)/(1-q)
        return int(round(r_prime/2.)*2)

    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias']/self.in_features)

    def forward(self, inputs):
        out = torch.zeros(inputs.shape[:-2] + (self.out_features,self.seq_len,))

        for i in range(self.out_features):
            out[...,i,:] = apc_btanh(mul(inputs, self.weight[:,i]), self.states)

        if self.bias:
            out = scaled_add(out,self.bias)

        return out
