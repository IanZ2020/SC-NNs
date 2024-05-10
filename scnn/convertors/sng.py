import torch
import numpy as np

class F2S:
    def __init__(self, seq_len, mode='bipolar'):
        self.seq_len = seq_len
        self.mode = mode

    def generate(self, x_float: torch.tensor) -> torch.tensor:
        rand_num_seq = torch.rand(x_float.shape + (self.seq_len,))
        if self.mode == 'bipolar':
            return x_float.add(1).mul(0.5).unsqueeze(dim=-1) > rand_num_seq
        elif self.mode == 'unipolar':
            return x_float.unsqueeze(dim=-1) > rand_num_seq
        else:
            raise NotImplementedError(f"SNG mode {self.mode} not implemented")
            
class S2F:
    def __init__(self, mode='bipolar'):
        self.mode = mode

    def evaluate(self, x_bitstream: torch.tensor) -> torch.tensor:
        if self.mode == 'bipolar':
            return torch.sum(x_bitstream, dim=-1).div(x_bitstream.shape[-1]).mul(2.0).add(-1.0)
        elif self.mode == 'unipolar':
            return torch.sum(x_bitstream, dim=-1).div(x_bitstream.shape[-1])
        

# x = SNG(seq_len=100)
# print(torch.sum(x.generate(torch.tensor([[0.5,0.5],[0.5,0.5]]))))

# x = SNG(seq_len=100,mode='unipolar')
# print(torch.sum(x.generate(torch.tensor([[0.5,0.5],[0.5,0.5]]))))