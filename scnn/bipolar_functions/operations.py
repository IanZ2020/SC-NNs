import torch
import numpy as np
from ..convertors import F2S
import multiprocessing

def mux_selector(inputs):
    no_channels, seq_len = inputs.shape
    random_idx = torch.randint(low=0, high=no_channels, size=(seq_len,))
    return inputs[random_idx, torch.arange(seq_len)]

def mul(x: torch.tensor, y: torch.tensor) -> torch.Any:
    return torch.logical_not(torch.logical_xor(x,y))

def scaled_add(x: torch.tensor, y: torch.tensor) -> torch.Any:
    x, y = torch.broadcast_tensors(x, y)

    select_x = F2S(seq_len=x.shape[-1], mode='unipolar').generate(torch.full(size=x.shape[:-1], fill_value=0.5))
    select_y = torch.logical_not(select_x)

    out = torch.logical_or(torch.logical_and(select_x, x), torch.logical_and(select_y, y))

    return out

def mean(*inputs):
    return mux_selector(torch.stack(inputs, dim=0))

def scaled_dot(x: torch.tensor, y: torch.tensor) -> torch.Any:
    return mux_selector(mul(x,y))


def scaled_matmul(x: torch.tensor, y: torch.tensor) -> torch.Any:
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


def p_scaled_matmul(x: torch.tensor, y: torch.tensor) -> torch.Any:
    x_shape, y_shape = x.shape, y.shape
    seq_len = x_shape[-1]
    assert seq_len == y_shape[-1] and x_shape[-2] == y_shape[0]
    x_reshape = x.view(-1,x_shape[-2],seq_len)
    y_reshape = y.view(y_shape[0], -1, seq_len)
    
    out = torch.zeros(x_reshape.shape[0], y_reshape.shape[1],seq_len)
    out_shape = x.shape[:-2] + y.shape[1:-1] + (seq_len,)

    procs = []
    queue = multiprocessing.Queue()

    def helper(row_idx, x_slice, queue):
        try:
            loc_res = []
            for j, y_slice in enumerate(y_reshape.transpose(0,1)):
                print(f'doing row no.{row_idx} column no.{j}')
                print(x_slice)
                loc_res.append(scaled_dot(x_slice, y_slice))
                print(f'done row no.{row_idx} column no.{j}')
            tpl = (row_idx,loc_res)
            print(f'done row no.{row_idx}')
            queue.put(tpl)
        except Exception as e:
            print(e)

    for row_idx, x_slice in enumerate(x_reshape):
        proc = multiprocessing.Process(target=helper,args=(row_idx, x_slice, queue))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    
    for _ in range(x_reshape.shape[0]):
        row_idx, loc_res = queue.get()
        out[row_idx] = torch.stack(loc_res)

    out = out.view(out_shape)
    return out