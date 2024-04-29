import torch

from ..gen import SNG

def mul(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    return torch.logical_xnot(torch.logical_xor(x,y))


def add(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    
