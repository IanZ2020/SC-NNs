import torch
import numpy as np

class SNG:
    def __init__(self, seq_len, mode='bipolar'):
        self.seq_len = seq_len
        self.mode = mode

    def generate(self,x_float: torch.tensor) -> torch.tensor:
        rand_num_seq = torch.rand(x_float.shape + (self.seq_len,))
        if self.mode == 'bipolar':
            return x_float.add(1).mul(0.5).unsqueeze(dim=-1) > rand_num_seq
        elif self.mode == 'unipolar':
            return x_float.unsqueeze(dim=-1) > rand_num_seq
        else:
            raise NotImplementedError(f"SNG mode {self.mode} not implemented")
            

# x = SNG(seq_len=100)
# print(torch.sum(x.generate(torch.tensor([[0.5,0.5],[0.5,0.5]]))))

# x = SNG(seq_len=100,mode='unipolar')
# print(torch.sum(x.generate(torch.tensor([[0.5,0.5],[0.5,0.5]]))))