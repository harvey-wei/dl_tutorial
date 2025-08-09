import torch
import unittest
import math


def pos_encoding_(x: torch.tensor):
    '''
    :parem x is of shape (B, N, dim) with B is batch_size, N is the seq length. dim is even
    : even and odd dim share the same frequence but use sin and cos, respectively
     freq = 1 / (10000^(2i/d)) = 10000^(-(2i)/d) = exp(-(2i) / d *log(10000))
    '''
    assert x.dim() == 3

    B, N, d_model = x.shape

    assert d_model % 2 == 0

    # 2i = 0, 2, 3, .... d_model -1, i.e. torch.arange(0, d_model, 2)
    # shape (d_model/2,)
    freq = -torch.arange(0, d_model, 2) * math.log(10000.0) / d_model
    freq = torch.exp(freq) # must use torch.expa instead of math.log because we need freq to be tensor for broadcasting
    freq = freq[None, :] # shape (1, d_model / 2)

    # pos_enc: shape(N, d_model)
    pos_enc = torch.zeros(N, d_model)

    # pos shape: (N, 1)
    pos = torch.arange(0, N).unsqueeze(1)

    # For broadcasting of element-wise product
    pos_enc[:, 0::2] = torch.sin(pos * freq)
    pos_enc[:, 1::2] = torch.cos(pos * freq)

    # torch.expand returns a new view with singleton dim expaned to a larger size.
    # -1 not changing the size of that dimension
    pos_enc = pos_enc[None, :, :].expand(B, -1, -1)

    return pos_enc

def pos_encoding(x: torch.tensor):
    '''
    :param x, input tensor is of shape (B, N, C)
    :return pos_enc, torch.tensor of shape (B, N, C)
    :method
        dim 2i and 2i + 1 share the frequence freq = 1 / (10000.0 ^((2i)/C))
        = 10000.0^(-(2i)/ C)
        a^b is not safe, i.e. overflow or underflow a^b = exp(bloga)
        freq = exp(-(2i)/ C) log(10000.0))
    '''
    assert 3 == x.dim()
    B, N, C = x.shape

    assert C % 2 == 0

    # [0, C) == 0 - (C-1)
    # freq shape : (C/2,)
    freq = torch.exp(- torch.arange(0, C, 2) * math.log(10000.0) / C)
    freq = freq[None, :] # (1, C/2 )

    # shape(N, 1)
    pos = torch.arange(N)[:, None]
    pos_enc = torch.zeros(N, C)

    # for even dimension
    pos_enc[:, 0::2] = torch.sin(pos * freq)

    # for odd dimension
    pos_enc[:, 1::2] = torch.cos(pos * freq)



    return pos_enc[None, :, :].expand(B, -1, -1)
