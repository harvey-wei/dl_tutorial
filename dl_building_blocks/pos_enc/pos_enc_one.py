import torch
import math


def get_pos_encoding(x: torch.Tensor):
    '''
    (2i, 2i+1) share the same frequence but use sin and cos,respectively
    2i is even dim while 2i +1 is odd dim.
    trick 1/a = exp(-lna)
    frequence(2i) = frequence(2i+1) = 1/(10000^(2i/d_model)) = exp(-ln(10000) * 2i/d_model)
    No learned weights -> can deal with varialbe length or resolution
    '''
    assert x.dim() == 3

    batch_size, seq_len, hidden_dim = x.size()

    assert hidden_dim % 2 == 0

    # pos_enc is of shape (seq_len, hidden_dim)
    # torch.arange is from [start, end) with step
    # Both odd and even dim shares the same frequency


    # freq is of shape (hidden_dim/2,)
    # math.log instead of torch.log because it is a scalar
    freq = torch.arange(0, hidden_dim, 2).float() * torch.log(10000.0) / hidden_dim
    freq = torch.exp(-freq)

    pos_enc = torch.zeros(seq_len, hidden_dim) # Shape (seq_len, hidden_dim)

    # None is used to add a singleton dimension. It is equivalent to unsqueeze
    pos = torch.arange(0, seq_len).float()[:, None] # Shape (seq_len, 1) for broadcasting to (seq_len, hidden_dim / 2)

    # (seq_lent, 1) * (hidden_dim / 2) -> (seq_len, 1) * (1, hidden_dim / 2) -> (seq_len, hidden_dim / 2)
    pos_enc[:, 0::2] = torch.sin(pos * freq)
    pos_enc[:, 1::2] = torch.cos(pos * freq)

    # torch.tensor expands only the singleton dimension
    # -1 means not changing the size of that dimension
    pos_enc = pos_enc[None, :, :].expand(batch_size, -1, -1)

    return pos_enc


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    pos_enc = get_pos_encoding(x)
    print(pos_enc.size())
    print(pos_enc)
    print(pos_enc[0, :, :])
    print(pos_enc[1, :, :])

    # Check if the pos_enc is correct
    for i in range(3):
        for j in range(4):
            print(pos_enc[0, i, j], pos_enc[1, i, j])
            assert torch.allclose(pos_enc[0, i, j], pos_enc[1, i, j], atol=1e-5)
    print("Pass")
