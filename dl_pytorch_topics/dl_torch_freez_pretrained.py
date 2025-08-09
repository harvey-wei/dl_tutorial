import timm
import torch.nn as nn


num_classes = 10
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, num_classes)  # Replace the classifier head


for name, param in model.named_parameters():
    if not name.startswith('head'):  # Freeze everything except the classification head
        param.requires_grad = False

'''
Because this is such a common pattern, requires_grad can also be set at the module level
with nn.Module.requires_grad_(). When applied to a module, .requires_grad_() takes effect on
all of the module’s parameters (which have requires_grad=True by default).
'''
model.requires_grad_(False)

trainable = [name for name, p in model.named_parameters() if p.requires_grad]
print("Trainable parameters:", trainable)

