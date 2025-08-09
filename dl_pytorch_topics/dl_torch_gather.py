import torch
import torch.nn as nn

# https://medium.com/analytics-vidhya/understanding-torch-gather-function-in-pytorch-f90db58b3c51
# Note that output tensor shape is the same as index tensor shape
# The element of index tensor tells which to choose along dimension dim
# The position of element in index tensor tells which to choose except dimension dim
# input and index must have the same number of dimensions.
# input and index must be aligned in all dimensions except the specified dim
# https://machinelearningknowledge.ai/how-to-use-torch-gather-function-in-pytorch-with-examples/
# https://pytorch.org/docs/stable/generated/torch.gather.html
# https://stackoverflow.com/questions/50999977/what-does-gather-do-in-pytorch-in-layman-terms
# gather will create a 

tensor1 = torch.arange(9).reshape(3, 3)

print(tensor1)
print(f'\nShape: {tensor1.shape}')

# specify row, align column
rows_tensor1 = torch.gather(input=tensor1, dim=0, index=torch.tensor([[0, 1, 2],
                                                       [2, 1, 0]]))

# specify col, align 0th, 1th row
col_tensor1 =  torch.gather(input=tensor1, dim=1, index=torch.tensor([[2, 1, 0],
                                                                      [0, 2, 1]]))


class CrossEntropyLoss(nn.Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len), GT integer token IDs
        Caveat: we must subtract the max value of logits before applying softmax for numerical stability
        and cancel out log and exp whenever possible
        '''
        # Subtract the max value of logits
        max_val, _ = torch.max(logits, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        logits -= max_val # (batch_size, seq_len, vocab_size)

        # log of sum of exp of logits
        log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1)) # (batch_size, seq_len)

        # negative log likelihood
        # Advanced indexing is restricted by the number of dimensions
        # batch_idx = torch.arange(logits.shape[0], device=logits.device)[:, None] # (batch_size, 1)
        # seq_idx = torch.arange(logits.shape[1], device=logits.device)[None, :] # (1, seq_len)
        # gahter along the last dimenison not limited by the number of dimensions
        neg_log_likelihood = log_sum_exp - logits.gather(dim=-1, index=targets[..., None]).squeeze(-1) # (batch_size, seq_len)

        return torch.mean(neg_log_likelihood)
