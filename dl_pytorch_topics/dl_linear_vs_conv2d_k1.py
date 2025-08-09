# https://src.koda.cnrs.fr/mmdc/mtan_s1s2_classif/-/issues/31
import torch
from einops import rearrange
import time

nb_features = 128
batch_size = 500
width = 64
conv_layer = torch.nn.Conv2d(nb_features, nb_features, 1, device="cuda")
linear_layer = torch.nn.Linear(nb_features, nb_features, device="cuda")

nb_conv_params = sum([len(params) for params in conv_layer.parameters()])
nb_linear_params = sum([len(params) for params in linear_layer.parameters()])

print(f"{nb_conv_params=}")
print(f"{nb_linear_params=}")

conv_times = []
linear_times = []

for t in range(0, 100):
    in_data = torch.rand((batch_size, nb_features, width, width), device="cuda")
    # Conv layer
    '''
    Why to use torch.cuda.synchronize() if timing?
    It forces the CPU to wait until all pending CUDA operations are completed.
    By default, the actual GPU computation happens in parallel, leading to fail to measure time

    '''
    torch.cuda.synchronize()
    conv_start = time.time()
    _ = conv_layer(in_data)
    torch.cuda.synchronize()
    conv_stop = time.time()
    conv_times.append(conv_stop - conv_start)

    # Linear layer
    torch.cuda.synchronize()
    linear_start = time.time()
    # nn.Linear assume the last dimension is feature dimension
    _ = linear_layer(rearrange(in_data, "b c w h -> (b w h) c"))
    torch.cuda.synchronize()
    linear_stop = time.time()
    linear_times.append(linear_stop - linear_start)

avg_conv_time = sum(conv_times) / len(conv_times)
avg_linear_time = sum(linear_times) / len(linear_times)

percent = [100 * l / c for (l, c) in zip(linear_times, conv_times)]

mean_percent = sum(percent) / len(percent)

print(f"{avg_linear_time=:.4f} s")
print(f"{avg_conv_time=:.4f} s")
print(
    f"Speed-up: min={min(percent):.2f}%, max={max(percent):.2f}%, mean={mean_percent:.2f}%"
)
