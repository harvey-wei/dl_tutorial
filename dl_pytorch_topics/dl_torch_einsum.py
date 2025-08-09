import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class AttBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_groups = num_groups

        self.group_norm = nn.GroupNorm(self.num_groups, in_ch)
        self.q_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.v_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.out_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()


    def forward(self, x: torch.tensor):
        assert 4 == x.dim(), "Input tensor x to AttBlock must be of 4 dimensional."

        N, C, H, W = x.shape
        assert C == self.in_ch

        h = self.group_norm(x) # [N, C, H, W], x is kept for residual connection.

        q = self.q_proj(h) # [N, out_ch, H, W]
        k = self.k_proj(h) # [N, out_ch, H, W]
        v = self.v_proj(h) # [N, out_ch, H, W]

        q = q.reshape(N, self.out_ch, H * W)
        k = k.reshape(N, self.out_ch, H * W)
        v = v.reshape(N, self.out_ch, H * W)

        # att_scores[n, i, j] = sum_{c} q[n, c, i] k[n, c, j]
        att_scores = torch.einsum('nci, ncj->nij', q, k)
        att_scores = att_scores * (self.out_ch) **(-0.5)
        att_scores = F.softmax(att_scores, dim=-1)

        # out[n, c, i] = sum_{j} att_scores[n, i, j] v[n, c, j]
        out = torch.einsum('nij, ncj->nci', att_scores, v) # [N, out_ch, H * W]
        out = out.reshape(N, self.out_ch, H, W)

        out = self.out_proj(out)

        return out + self.skip_proj(x)


class TestAttBlock(unittest.TestCase):
    '''
    test method name must start with test_ for unittest to identify it.
    '''
    def test_forward_pass_same_channels(self):
        block = AttBlock(in_ch=32, out_ch=32, num_groups=4)
        x = torch.randn(8, 32, 16, 16)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_pass_different_channels(self):
        block = AttBlock(in_ch=32, out_ch=64, num_groups=4)
        x = torch.randn(4, 32, 8, 8)
        y = block(x)
        self.assertEqual(y.shape, (4, 64, 8, 8))

    def test_gradient_flow(self):
        block = AttBlock(in_ch=16, out_ch=16, num_groups=4)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        y = block(x)
        y.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
