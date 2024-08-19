import torch
from .operations import *
from ..convertors import F2S


class MUXScaled_Linear():
    def __init__(self, 
                 in_features,
                 out_features,
                 seq_len,
                 is_bias=False) -> None:
        super().__init__()

        self.down_scale = in_features
        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = is_bias
        self.weight = None
        self.bias = None

    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias'])
            if len(self.bias.shape) <=2 : 
                self.bias = self.bias.unsqueeze(dim=0)
            self.weight = torch.cat([self.weight, self.bias], dim=0)

    def forward(self, inputs):
        assert self.weight is not None
        if self.is_bias:
            bias_input = torch.ones(1,self.seq_len, dtype=torch.bool).expand(inputs.shape[:-2]+(1,-1,))
            inputs = torch.concat([inputs, bias_input],dim=-2)

        out = scaled_matmul(inputs, self.weight)

        return out

class MUXScaled_LinearAct():
    def __init__(self, 
                 scalar,
                 in_features,
                 out_features,
                 seq_len,
                 is_bias=False) -> None:
        super().__init__()
        if scalar < 1/in_features: 
            raise NotImplementedError
        self.scalar = scalar
        self.down_scale = in_features
        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = is_bias
        self.weight = None
        self.bias = None

        
        
    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias'])
            if len(self.bias.shape) <=2 : 
                self.bias = self.bias.unsqueeze(dim=0)
            self.weight = torch.cat([self.weight, self.bias], dim=0)

    def forward(self, inputs):
        assert self.weight is not None
        if self.is_bias:
            bias_input = torch.ones(1,self.seq_len, dtype=torch.bool).expand(inputs.shape[:-2]+(1,-1,))
            inputs = torch.concat([inputs, bias_input],dim=-2)

        out = scaled_matmul(inputs, self.weight)
        
        out = stanh(out, r= int(self.scalar*2*self.in_features))
        return out


class APCLinearAct():
    def __init__(self, 
                 in_features,
                 out_features,
                 seq_len,
                 scalar=1.,
                 num_au_layers=1,
                 is_bias=False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = is_bias
        self.weight = None
        self.bias = None
        self.scalar = scalar
        self.num_au_layers = num_au_layers
        self.states = self.get_btanh_states()

    def get_btanh_states(self,):
        q = 1.835*(2*self.in_features)**(-0.5552)
        r_prime = 2*self.in_features + 2*(self.scalar-1)*(self.in_features-1)/(1-q)
        return int(torch.round(r_prime/2.)*2)

    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias'])
            if len(self.bias.shape) <=2 : 
                self.bias = self.bias.unsqueeze(dim=0)
            self.weight = torch.cat([self.weight, self.bias], dim=0)

    def forward(self, inputs):
        out = torch.zeros(inputs.shape[:-2] + (self.out_features,self.seq_len,))

        if self.is_bias:
            bias_input = torch.ones(1,self.seq_len, dtype=torch.bool).expand(inputs.shape[:-2]+(1,-1,))
            inputs = torch.concat([inputs, bias_input],dim=-2)

        for i in range(self.out_features):
            out[...,i,:] = apc_btanh(mul(inputs, self.weight[:,i]), self.states, self.num_au_layers)


        return out


class APCLinear():
    def __init__(self, 
                 in_features,
                 out_features,
                 seq_len,
                 scalar=1.,
                 num_au_layers=1,
                 is_bias=False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.is_bias = is_bias
        self.weight = None
        self.bias = None
        self.scalar = scalar
        self.num_au_layers = num_au_layers

    def load_weight(self, data):
        assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

        self.weight = F2S(seq_len=self.seq_len)(data['weight'])

        if self.is_bias:
            assert data['bias'].shape == torch.Size([self.out_features])
            self.bias = F2S(seq_len=self.seq_len)(data['bias'])
            if len(self.bias.shape) <=2 : 
                self.bias = self.bias.unsqueeze(dim=0)
            self.weight = torch.cat([self.weight, self.bias], dim=0)

    def forward(self, inputs):
        out = torch.zeros(inputs.shape[:-2] + (self.out_features,))

        if self.is_bias:
            bias_input = torch.ones(1,self.seq_len, dtype=torch.bool).expand(inputs.shape[:-2]+(1,-1,))
            inputs = torch.concat([inputs, bias_input],dim=-2)

        apc = ap_counter(in_features=inputs.shape[-2], num_au_layers=self.num_au_layers)
        for i in range(self.out_features):
            out[...,i] = self.scalar * (apc(mul(inputs, self.weight[:,i])).sum(dim=-1) / (self.seq_len) * 2 -self.in_features)
        return out



# class ORLinear():
#     def __init__(self, 
#                  in_features,
#                  out_features,
#                  seq_len,
#                  is_bias=False) -> None:
#         super().__init__()

#         self.in_features = in_features
#         self.out_features = out_features
#         self.seq_len = seq_len
#         self.is_bias = is_bias
#         self.weight = None
#         self.bias = None

#     def load_weight(self, data):
#         assert data['weight'].shape == torch.Size([self.in_features, self.out_features])

#         self.weight = F2S(seq_len=self.seq_len)(data['weight'])

#         if self.is_bias:
#             assert data['bias'].shape == torch.Size([self.out_features])
#             self.bias = F2S(seq_len=self.seq_len)(data['bias'])
#             if len(self.bias.shape) <=2 : 
#                 self.bias = self.bias.unsqueeze(dim=0)
#             self.weight = torch.cat([self.weight, self.bias], dim=0)

#     def forward(self, inputs):
#         out = torch.zeros(inputs.shape[:-2] + (self.out_features,self.seq_len),dtype=torch.bool)

#         if self.is_bias:
#             bias_input = torch.ones(1,self.seq_len, dtype=torch.bool).expand(inputs.shape[:-2]+(1,-1,))
#             inputs = torch.concat([inputs, bias_input],dim=-2)


#         for i in range(self.out_features):
#             out[...,i,:] = torch.any(mul(inputs, self.weight[:,i]),dim=-2)
#         return out