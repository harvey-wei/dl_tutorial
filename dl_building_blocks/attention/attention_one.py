import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim ,num_heads) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "d_model must be divisble by num_heads"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # We simply split QKV_projection into multiple head
        self.head_dim = embed_dim // num_heads

        # Linear projection for Q, K, V
        # Like Conv, Attention can deal with variable length or variable resolution
        # What really matters is the feature dimension.
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # self._reset_parameters()

    def forward(self, x: torch.tensor):
        assert 3 == x.dim(), 'input tensor x must have 3 dimensions'

        B, N, C = x.shape

        assert C == self.input_dim, 'feature dimension of x must equal to input_dim in network'

        qkv = self.qkv_proj(x) # (B, N, 3 * embed_dim)

        # q k v are all of shape (B, N, embed_dim)
        q = qkv[:, :, 0:self.embed_dim]
        k = qkv[:, :, self.embed_dim:(2 * self.embed_dim)]
        v = qkv[:, :, (2 * self.embed_dim):]

        # split along embed_dim to num_heads
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)

        # sim_matrix computation
        # q:(B, N, H, head_dim), v:(B, N, H, head_dim) = (B, H, N, N)
        # sim_{b, h, i, j } = sum_{d} q_{b, i, h, d}v_{b, j, h, d}
        # einsum ignores summation notation, sum over repeated indices and indices not in the output
        attn_scores = torch.einsum('bihd,bjhd->bhij', q, k) # (B, H, N, N)

        attn_scores = F.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)

        output = torch.einsum('bhik, bkhd->bhid', attn_scores, v) # (B, H, N, head_dim)
        output = output.permute(0, 2, 1, 3).reshape(B, N, self.embed_dim) # (B, N, embed_dim)

        output = self.out_proj(output)

        return output

if __name__ == '__main__':
    mha = MultiHeadSelfAttention(input_dim=128, embed_dim=128, num_heads=8)
    x = torch.randn(2, 10, 128)
    out = mha(x)
    print(out.shape)  # (2, 10, 128)
