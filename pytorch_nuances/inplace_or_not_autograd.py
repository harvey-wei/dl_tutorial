import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

def print_tensor_info(tensor, name=""):
    """Print detailed information about a tensor including gradients."""
    print(f"\n{name}:" if name else "\nTensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Content: {tensor}")
    print(f"  Requires grad: {tensor.requires_grad}")
    if tensor.grad is not None:
        print(f"  Grad: {tensor.grad}")
    else:
        print(f"  Grad: None")

def demonstrate_inplace_operations_break_autograd():
    """Demonstrate how in-place operations break backward propagation."""
    print("=" * 80)
    print("IN-PLACE OPERATIONS BREAK AUTOGRAD")
    print("=" * 80)
    
    # Example 1: Basic in-place operation breaking autograd
    print("\nExample 1: Basic in-place operation breaking autograd")
    
    # Create tensors that require gradients
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    print_tensor_info(x, "Original x")
    print_tensor_info(y, "Original y")
    
    # Out-of-place operation (works correctly)
    print("\n--- Out-of-place operation (CORRECT) ---")
    z_out = x + y
    loss_out = z_out.sum()
    print_tensor_info(z_out, "z_out = x + y")
    
    loss_out.backward()
    print_tensor_info(x, "x after backward (out-of-place)")
    print_tensor_info(y, "y after backward (out-of-place)")
    
    # Reset gradients
    x.grad.zero_()
    y.grad.zero_()
    
    # In-place operation (breaks autograd)
    print("\n--- In-place operation (BREAKS AUTOGRAD) ---")
    x_inplace = x.clone()  # Clone to avoid modifying original
    y_inplace = y.clone()
    
    # In-place addition
    x_inplace.add_(y_inplace)  # This modifies x_inplace in-place
    loss_inplace = x_inplace.sum()
    print_tensor_info(x_inplace, "x_inplace after add_(y_inplace)")
    
    try:
        loss_inplace.backward()
        print_tensor_info(x_inplace, "x_inplace after backward (in-place)")
        print_tensor_info(y_inplace, "y_inplace after backward (in-place)")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("In-place operations break the computational graph!")

def demonstrate_compound_operators():
    """Demonstrate how compound operators (+=, -=, etc.) break autograd."""
    print("\n" + "=" * 80)
    print("COMPOUND OPERATORS BREAK AUTOGRAD")
    print("=" * 80)
    
    # Example 1: Compound assignment operators
    print("\nExample 1: Compound assignment operators")
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    
    print_tensor_info(x, "Original x")
    print_tensor_info(y, "Original y")
    
    # Out-of-place compound operation (works correctly)
    print("\n--- Out-of-place compound operation (CORRECT) ---")
    x_out = x + y
    loss_out = x_out.sum()
    loss_out.backward()
    print_tensor_info(x, "x after backward (out-of-place)")
    print_tensor_info(y, "y after backward (out-of-place)")
    
    # Reset gradients
    x.grad.zero_()
    y.grad.zero_()
    
    # In-place compound operation (breaks autograd)
    print("\n--- In-place compound operation (BREAKS AUTOGRAD) ---")
    x_inplace = x.clone()
    y_inplace = y.clone()
    
    # In-place compound assignment
    x_inplace += y_inplace  # This is equivalent to x_inplace.add_(y_inplace)
    loss_inplace = x_inplace.sum()
    print_tensor_info(x_inplace, "x_inplace after += y_inplace")
    
    try:
        loss_inplace.backward()
        print_tensor_info(x_inplace, "x_inplace after backward (in-place)")
        print_tensor_info(y_inplace, "y_inplace after backward (in-place)")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Compound operators break the computational graph!")

def demonstrate_common_inplace_operations():
    """Demonstrate various in-place operations that break autograd."""
    print("\n" + "=" * 80)
    print("COMMON IN-PLACE OPERATIONS THAT BREAK AUTOGRAD")
    print("=" * 80)
    
    # List of in-place operations to test
    inplace_ops = [
        ("add_", lambda x, y: x.add_(y)),
        ("sub_", lambda x, y: x.sub_(y)),
        ("mul_", lambda x, y: x.mul_(y)),
        ("div_", lambda x, y: x.div_(y)),
        ("pow_", lambda x, y: x.pow_(2)),
        ("clamp_", lambda x, y: x.clamp_(min=0, max=10)),
        ("relu_", lambda x, y: x.relu_()),
        ("sigmoid_", lambda x, y: x.sigmoid_()),
        ("tanh_", lambda x, y: x.tanh_()),
        ("fill_", lambda x, y: x.fill_(5.0)),
        ("zero_", lambda x, y: x.zero_()),
        ("uniform_", lambda x, y: x.uniform_()),
        ("normal_", lambda x, y: x.normal_()),
    ]
    
    for op_name, op_func in inplace_ops:
        print(f"\n--- Testing {op_name} ---")
        
        # Create fresh tensors
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        
        print_tensor_info(x, f"Original x before {op_name}")
        
        try:
            # Apply in-place operation
            result = op_func(x, y)
            loss = result.sum()
            print_tensor_info(result, f"Result after {op_name}")
            
            # Try backward pass
            loss.backward()
            print(f"✓ {op_name}: Backward successful")
            print_tensor_info(x, f"x after backward with {op_name}")
            
        except RuntimeError as e:
            print(f"✗ {op_name}: {e}")
        except Exception as e:
            print(f"✗ {op_name}: Unexpected error - {e}")

def demonstrate_safe_alternatives():
    """Demonstrate safe alternatives to in-place operations."""
    print("\n" + "=" * 80)
    print("SAFE ALTERNATIVES TO IN-PLACE OPERATIONS")
    print("=" * 80)
    
    # Example 1: Safe alternatives for common operations
    print("\nExample 1: Safe alternatives")
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    
    print_tensor_info(x, "Original x")
    print_tensor_info(y, "Original y")
    
    # Safe alternatives
    print("\n--- Safe alternatives (preserve autograd) ---")
    
    # Instead of x.add_(y)
    z1 = x + y  # or x.add(y)
    print_tensor_info(z1, "z1 = x + y (safe)")
    
    # Instead of x.mul_(2)
    z2 = x * 2  # or x.mul(2)
    print_tensor_info(z2, "z2 = x * 2 (safe)")
    
    # Instead of x.clamp_(min=0)
    z3 = torch.clamp(x, min=0)  # or x.clamp(min=0)
    print_tensor_info(z3, "z3 = torch.clamp(x, min=0) (safe)")
    
    # Instead of x.relu_()
    z4 = torch.relu(x)  # or x.relu()
    print_tensor_info(z4, "z4 = torch.relu(x) (safe)")
    
    # Test backward pass
    loss = z1.sum() + z2.sum() + z3.sum() + z4.sum()
    loss.backward()
    print_tensor_info(x, "x after backward (safe operations)")
    print_tensor_info(y, "y after backward (safe operations)")

def demonstrate_why_inplace_breaks_autograd():
    """Explain why in-place operations break autograd."""
    print("\n" + "=" * 80)
    print("WHY IN-PLACE OPERATIONS BREAK AUTOGRAD")
    print("=" * 80)
    
    print("""
    In-place operations break autograd because they modify tensors in-place,
    which destroys the computational graph needed for backpropagation.
    
    Here's what happens:
    
    1. When you create a tensor with requires_grad=True, PyTorch tracks it
    2. Operations create a computational graph connecting inputs to outputs
    3. In-place operations modify the original tensor, breaking the graph
    4. During backward pass, PyTorch can't trace back through the graph
    
    Example:
    """)
    
    # Create a simple computational graph
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    
    print_tensor_info(x, "x (requires_grad=True)")
    print_tensor_info(y, "y (requires_grad=True)")
    
    # Normal operation - creates graph
    z = x + y
    print_tensor_info(z, "z = x + y (creates computational graph)")
    
    # In-place operation - breaks graph
    x.add_(y)
    print_tensor_info(x, "x after x.add_(y) (breaks computational graph)")
    
    print("""
    After x.add_(y), the original tensor x is modified in-place.
    The computational graph that connected x to z is destroyed.
    PyTorch can no longer compute gradients with respect to x.
    """)

def demonstrate_practical_example():
    """Show a practical example of the problem."""
    print("\n" + "=" * 80)
    print("PRACTICAL EXAMPLE: NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    # Create a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model and data
    model = SimpleNet()
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)
    
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print_tensor_info(x, "Input x")
    print_tensor_info(target, "Target")
    
    # Correct way (out-of-place)
    print("\n--- Correct way (out-of-place) ---")
    model.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    print(f"Loss: {loss.item()}")
    print("Gradients computed successfully:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}.grad: {param.grad}")
    
    # Wrong way (in-place modification)
    print("\n--- Wrong way (in-place modification) ---")
    model.zero_grad()
    output = model(x)
    
    # This would break autograd if x had requires_grad=True
    # x.add_(torch.randn_like(x))  # This would cause an error
    
    # Instead, let's show what happens with model parameters
    print("In-place modification of model parameters would break autograd:")
    print("  model.linear.weight.add_(torch.randn_like(model.linear.weight))  # WRONG!")
    print("  model.linear.bias.add_(torch.randn_like(model.linear.bias))      # WRONG!")
    
    print("\nCorrect way to modify parameters:")
    print("  model.linear.weight = model.linear.weight + torch.randn_like(model.linear.weight)")
    print("  model.linear.bias = model.linear.bias + torch.randn_like(model.linear.bias)")

def demonstrate_exceptions_and_edge_cases():
    """Show exceptions and edge cases where in-place might work."""
    print("\n" + "=" * 80)
    print("EXCEPTIONS AND EDGE CASES")
    print("=" * 80)
    
    print("""
    There are some cases where in-place operations might work:
    
    1. Tensors without requires_grad=True
    2. Operations that don't affect the computational graph
    3. Detached tensors
    """)
    
    # Case 1: Tensor without requires_grad
    print("\nCase 1: Tensor without requires_grad=True")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    y = torch.tensor([0.5, 0.5, 0.5], requires_grad=False)
    
    print_tensor_info(x, "x (requires_grad=False)")
    x.add_(y)  # This works fine
    print_tensor_info(x, "x after add_(y) - works because no autograd needed")
    
    # Case 2: Detached tensor
    print("\nCase 2: Detached tensor")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    
    z = x + y
    x_detached = x.detach()  # Detach from computational graph
    print_tensor_info(x_detached, "x_detached (detached from graph)")
    
    x_detached.add_(y)  # This works because x_detached is not in the graph
    print_tensor_info(x_detached, "x_detached after add_(y) - works because detached")
    
    # Case 3: In-place operations on leaf nodes (still breaks autograd)
    print("\nCase 3: In-place operations on leaf nodes")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print_tensor_info(x, "x (leaf node with requires_grad=True)")
    
    try:
        x.add_(torch.tensor([0.1, 0.1, 0.1]))
        print("In-place operation on leaf node succeeded")
        print_tensor_info(x, "x after add_")
    except RuntimeError as e:
        print(f"In-place operation on leaf node failed: {e}")


def compare_inplace_vs_outofplace():
    print("=" * 60)
    print("Case 1: Without autograd")
    x = torch.ones(3)
    print("Initial: id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    x += 1  # in-place, x will point to the same storage but with changed value
    print("After x += 1: id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    x = x + 1  # out-of-place
    print("After x = x + 1: id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    x = x * 2 # out-of-place, x will point to a new storage
    print("After x = x * 2: id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    x = x / 2 # out-of-place, x will point to a new storage
    print("After x = x / 2: id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    print("=" * 60)
    print("Case 2: With autograd (requires_grad=True)")

    # In-place on leaf with grad will error
    x = torch.ones(3, requires_grad=True)
    print("Initial (leaf): id =", id(x), "data_ptr =", x.data_ptr(), "values =", x)

    try:
        x += 1
    except RuntimeError as e:
        print("x += 1 error:", e)

    # Out-of-place is safe
    x = torch.ones(3, requires_grad=True)
    y = x + 1
    print("x = x + 1 OK: id =", id(y), "data_ptr =", y.data_ptr(), "values =", y)

    print("=" * 60)

def inplace_assignment_vs_outofplace_assignment():
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.sigmoid(x)  # intermediate

    # ✅ Safe: clone before modifying
    y2 = y.clone()
    y2[:, 0] = 0
    loss = (y2 ** 2).sum()
    loss.backward()  # works

    # ❌ Unsafe: modify original
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.sigmoid(x) # also requires_grad=True

    # y2 is a reference to y, point to same storage
    # so, if you modify y2, you are also modifying y . But y involves gradient, so you break the computational graph.
    y2 = y 
    
    # check whether y2 and y are in the same storage
    print(f"y2 storage: {y2.data_ptr()}")
    print(f"y storage: {y.data_ptr()}")
    
    try:
        y2[:, 0] = 0 # this will modify the original tensor y, and y2 is a reference to y, so y will also be modified
        loss = (y2 ** 2).sum()
        loss.backward()
    except RuntimeError as e:
        print("Error without clone:", e)

if __name__ == "__main__":
    print("PYTORCH IN-PLACE OPERATIONS AND AUTOGRAD")
    print("Understanding why in-place operations break backward propagation")
    
    # demonstrate_inplace_operations_break_autograd()
    # demonstrate_compound_operators()
    # demonstrate_common_inplace_operations()
    # demonstrate_safe_alternatives()
    # demonstrate_why_inplace_breaks_autograd()
    # demonstrate_practical_example()
    # demonstrate_exceptions_and_edge_cases()
    # compare_inplace_vs_outofplace()
    inplace_assignment_vs_outofplace_assignment()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("1. In-place operations (add_(), +=, etc.) break autograd")
    print("2. They modify tensors in-place, destroying the computational graph")
    print("3. Use out-of-place operations (add(), +, etc.) instead")
    print("4. Exceptions: tensors without requires_grad=True or detached tensors")
    print("5. Always use safe alternatives in neural network training")
    print("=" * 80)