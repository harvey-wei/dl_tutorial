import torch

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # self.register_buffer() is a method in PyTorch's nn.Module class that registers a tensor as a buffer in the model. Buffers are persistent tensors that are not considered model parameters (i.e., they are not updated during backpropagation), but are part of the module’s state. Buffers are typically used to store things like running statistics or constants that should be saved when you save the model's state dict but should not be optimized during training.
        # Registering 'betas' as a buffer
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())

        # Calculating alphas and other constants
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Registering sqrt_alphas_bar and sqrt_one_minus_alphas_bar as buffers
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
