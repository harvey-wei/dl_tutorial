import unittest
import torch
import math
from pos_enc import pos_encoding  # 👈 import your function

class TestPosEncodingValues(unittest.TestCase):

    def test_known_values(self):
        test_cnt = 100000
        for _ in range(test_cnt):
            B, N, d_model = 1, 2, 4
            x = torch.randn(B, N, d_model)

            pos_enc = pos_encoding(x)  # shape: (1, 2, 4)

            # Manually compute expected values for 2 positions, 4 dimensions
            expected = torch.zeros(N, d_model)

            for pos in range(N):  # 0, 1
                for i in range(0, d_model, 2):  # even: sin, odd: cos
                    angle_rate = 1 / (10000 ** (i / d_model))
                    angle = pos * angle_rate
                    expected[pos, i] = math.sin(angle)
                    expected[pos, i+1] = math.cos(angle)

            expected = expected.unsqueeze(0)  # shape: (1, 2, 4)

            # Use allclose for floating point comparison
            self.assertTrue(torch.allclose(pos_enc, expected, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
