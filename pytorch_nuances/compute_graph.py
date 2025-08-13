import torch

# ----------------------------
# Utilities
# ----------------------------
def node_name(fn):
    if fn is None:
        return "None"
    return type(fn).__name__

def print_graph(fn, depth=0, seen=None, max_depth=50):
    """DFS over autograd graph starting from a grad_fn."""
    if fn is None or depth > max_depth:
        return
    if seen is None:
        seen = set()
    if fn in seen:
        print("  " * depth + f"- {node_name(fn)} (…revisited)")
        return
    seen.add(fn)

    print("  " * depth + f"- {node_name(fn)}")
    # Each next_functions item is (next_fn, _)
    for nxt, _ in getattr(fn, "next_functions", []):
        print_graph(nxt, depth + 1, seen, max_depth)

def show_storage_and_version(t, label):
    print(f"{label}: data_ptr={t.data_ptr()}, _version={t._version}")

# ----------------------------
# Case A: Safe (clone)
# ----------------------------
print("\n===== CASE A: clone (safe) =====")
x = torch.randn(2, 3, requires_grad=True)
y = torch.sigmoid(x)              # y.grad_fn = SigmoidBackward0

# Inspect forward result
show_storage_and_version(y, "y (before clone)")

y2 = y.clone()                    # separate storage
show_storage_and_version(y2, "y2 (clone)")
print(f"Same storage? {y2.data_ptr() == y.data_ptr()}")

# Modify the clone; original y is intact
y2[:, 0] = 0
show_storage_and_version(y, "y (after y2 in-place)")
show_storage_and_version(y2, "y2 (after in-place)")

loss = (y2 ** 2).sum()
print("\nAutograd graph for CASE A (loss.grad_fn):")
print_graph(loss.grad_fn)

loss.backward()                   # ✅ works
print("CASE A backward: OK")
print(f"x.grad shape: {x.grad.shape}\n")


# ----------------------------
# Case B: Unsafe (alias)
# ----------------------------
print("===== CASE B: alias (unsafe) =====")
x = torch.randn(2, 3, requires_grad=True)
y = torch.sigmoid(x)              # y will be saved for backward by SigmoidBackward0

show_storage_and_version(y, "y (before alias)")

y2 = y                            # alias (same storage)
show_storage_and_version(y2, "y2 (alias)")
print(f"Same storage? {y2.data_ptr() == y.data_ptr()}")

# Mutate through alias (this mutates y's storage)
y2[:, 0] = 0
show_storage_and_version(y,  "y (after alias in-place)")
show_storage_and_version(y2, "y2 (after in-place)")

loss = (y2 ** 2).sum()
print("\nAutograd graph for CASE B (loss.grad_fn):")
print_graph(loss.grad_fn)

print("\nAttempting backward for CASE B:")
try:
    loss.backward()               # ❌ will raise due to version counter mismatch
except RuntimeError as e:
    print("RuntimeError:", e)


'''
In PyTorch, grad_fn is a reference to the autograd Function object that created a given tensor during the forward pass — it’s essentially a pointer to the node in the computation graph that will be used during backward.

1. The purpose of grad_fn
When requires_grad=True and you perform an operation on a tensor, PyTorch:

Runs the forward computation for that op.

Creates an internal Function object for the backward computation (e.g., AddBackward0, SigmoidBackward0, MmBackward).

Stores a reference to that backward object in the grad_fn attribute of the output tensor.

This is how PyTorch knows how to propagate gradients back to the inputs.

2. When grad_fn is None
If a tensor:

Was created by the user directly (e.g., torch.randn() without ops), or

Was computed with requires_grad=False everywhere in its ancestry, or

Was detached from the graph via .detach()


'''