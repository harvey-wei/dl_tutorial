import torch

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

def run_case(title, build, mutate, expect_error: bool):
    print(f"\n=== {title} ===")
    # Build graph
    x_list, out, loss = build()
    print("grad_fn(out):", type(out.grad_fn).__name__ if out.grad_fn is not None else None)

    # Mutate something (possibly in-place)
    mutate(x_list, out)

    # Backward
    try:
        loss.backward()
        if expect_error:
            print("❌ Expected a RuntimeError, but backward succeeded.")
        else:
            print("✅ Backward succeeded (as expected).")
    except RuntimeError as e:
        if expect_error:
            print("✅ Raised RuntimeError (as expected):")
            print("   ", e)
        else:
            print("❌ Unexpected RuntimeError:")
            print("   ", e)

# ---------------------------
# 1) Sigmoid saves OUTPUT (y)
# Mutating y in-place -> error
# Mutating a clone of y -> fine
# ---------------------------

def build_sigmoid():
    x = torch.randn(4, requires_grad=True)
    y = torch.sigmoid(x)              # SigmoidBackward0 saves y (output)
    loss = (y**2).sum()
    return [x, y], y, loss

def mutate_sigmoid_break(x_list, y):
    # In-place mutation on saved OUTPUT
    y[...] = 0.0

def mutate_sigmoid_safe(x_list, y):
    # Out-of-place: mutate a clone instead of y
    y_clone = y.clone()
    y_clone[...] = 0.0  # OK; y remains intact

run_case("Sigmoid: mutate OUTPUT (should fail)",
         build_sigmoid, mutate_sigmoid_break, expect_error=True)

run_case("Sigmoid: mutate CLONE of OUTPUT (should succeed)",
         build_sigmoid, mutate_sigmoid_safe, expect_error=False)


# ---------------------------
# 2) ReLU saves INPUT (x)
# Mutating x in-place -> error
# Mutating a clone of x -> fine
# ---------------------------

def build_relu():
    x = torch.randn(4, requires_grad=True)
    y = torch.relu(x)                 # ReluBackward0 needs input sign (saves x/input-mask)
    loss = (y**2).sum()
    return [x, y], y, loss

def mutate_relu_break(x_list, y):
    x, _ = x_list
    x.add_(1.0)                       # in-place on saved INPUT

def mutate_relu_safe(x_list, y):
    x, _ = x_list
    x_clone = x.clone()
    x_clone.add_(1.0)                 # mutate clone; original x intact

run_case("ReLU: mutate INPUT (should fail)",
         build_relu, mutate_relu_break, expect_error=True)

run_case("ReLU: mutate CLONE of INPUT (should succeed)",
         build_relu, mutate_relu_safe, expect_error=False)


# ---------------------------
# 3) Matmul saves BOTH inputs (A, B)
# Mutating either A or B in-place -> error
# Mutating clones -> fine
# ---------------------------

def build_matmul():
    A = torch.randn(3, 5, requires_grad=True)
    B = torch.randn(5, 4, requires_grad=True)
    C = A @ B                          # MmBackward saves inputs
    loss = C.pow(2).sum()
    return [A, B, C], C, loss

def mutate_matmul_break_A(x_list, C):
    A, B, _ = x_list
    A.add_(0.1)                        # in-place on saved INPUT A

def mutate_matmul_break_B(x_list, C):
    A, B, _ = x_list
    B.mul_(1.1)                        # in-place on saved INPUT B

def mutate_matmul_safe(x_list, C):
    A, B, _ = x_list
    A2, B2 = A.clone(), B.clone()      # mutate clones only
    A2.add_(0.1); B2.mul_(1.1)

run_case("Matmul: mutate INPUT A (should fail)",
         build_matmul, mutate_matmul_break_A, expect_error=True)

run_case("Matmul: mutate INPUT B (should fail)",
         build_matmul, mutate_matmul_break_B, expect_error=True)

run_case("Matmul: mutate CLONES of inputs (should succeed)",
         build_matmul, mutate_matmul_safe, expect_error=False)


# ---------------------------
# 4) Add typically doesn't need inputs (meta only)
# Mutating x AFTER forward usually won't error
# (But still bumps version; safe here because AddBackward doesn't need x/y)
# ---------------------------

def build_add():
    x = torch.randn(4, requires_grad=True)
    y = x + 3.0                        # AddBackward0
    loss = y.sum()
    return [x, y], y, loss

def mutate_add_ok(x_list, y):
    x, _ = x_list
    x.add_(1.0)                        # in-place on leaf input; AddBackward won't need x

run_case("Add: mutate INPUT after forward (should succeed)",
         build_add, mutate_add_ok, expect_error=False)


# ---------------------------
# 5) Softmax saves OUTPUT
# Mutating y in-place -> error
# ---------------------------

def build_softmax():
    x = torch.randn(5, requires_grad=True)
    y = torch.softmax(x, dim=0)        # SoftmaxBackward0 saves output
    loss = (y**2).sum()
    return [x, y], y, loss

def mutate_softmax_break(x_list, y):
    y[0] = 0.0                         # in-place on saved OUTPUT

run_case("Softmax: mutate OUTPUT (should fail)",
         build_softmax, mutate_softmax_break, expect_error=True)

print("\nAll cases done.")
