import torch
import math


def pos_encoding(x: torch.tensor):
    '''
    :param x, input tensor of shape B, N, C
    :return pos_enc, tensor of shape B, N, C
    :method
    both even (2i) and odd (2i + 1) dimension shares the freq = 1 / (10000^{2i} / C)
    to avoid numerical issue in a^b
    freq = exp((-2i) * log(10000) / C)
    2i = 0, 2, 4, ... C -2 with C is even nubmer.
    '''
    assert 3 == x.dim()

    B, N, C = x.shape

    # frea shape: (1, C/2)
    freq = torch.exp(-torch.arange(0, C, 2) * math.log(10000.0) / C)[None, :]

    # pos shape: (N, 1)
    pos = torch.arange(N)[:, None]

    # pos_enc shape: (N, C)
    pos_enc = torch.zeros(N, C)

    # Both pos_enc and freq broadcast to a matrix of shape (N, C/2)
    pos_enc[:, 0::2] = torch.sin(pos * freq)
    pos_enc[:, 1::2] = torch.cos(pos * freq)

    return pos_enc[None, :, :].expand(B, -1, -1)


if __name__ == '__main__':
    pass
