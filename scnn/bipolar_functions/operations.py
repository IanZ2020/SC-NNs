import torch

from ..convertors import F2S

class operator:
    pass

class mul(operator):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: torch.tensor, y: torch.tensor) -> torch.Any:
        return torch.logical_xnot(torch.logical_xor(x,y))

class scaled_add(operator):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: torch.tensor, y: torch.tensor, upscale_factor: torch.tensor) -> torch.Any:
        rand_select_x = F2S(seq_len=x.shape[-1], mode='unipolar').generate(torch.full(size=x.shape[:-1], fill_value=0.5))


        rand_select_y = torch.logical_not(rand_select_x)

        output = torch.logical_or(torch.logical_and(rand_select_x, x), torch.logical_and(rand_select_y, y))

        return output


class dot(operator):
    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def __call__(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        pass

class matmul(operator):
    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def __call__(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        pass