import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Standard module with parameters (weights + bias)
        self.linear = nn.Linear(input_dim, output_dim)

        # Manually registered parameter (learnable)
        self.scale = nn.Parameter(torch.ones(1))  # requires_grad=True

        # Buffer (not learnable, but saved and moved with model)
        self.register_buffer("running_max", torch.zeros(1))  # requires_grad=False

    def forward(self, x):
        out = self.linear(x) * self.scale

        # Update buffer during forward pass (example logic)
        self.running_max = torch.maximum(self.running_max, out.max().detach())

        return out



if __name__ == '__main__':
    model = ExampleModel(4, 2)

    # List parameters
    print("\nParameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # List buffers
    print("\nBuffers:")
    for name, buf in model.named_buffers():
        print(f"{name}: {buf.shape}")

    # List full state_dict (parameters + buffers)
    print("\nState dict:")
    for name, tensor in model.state_dict().items():
        print(f"{name}: {tensor.shape}")
