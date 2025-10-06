import torch

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print("Name:", props.name)
print("Compute capability:", props.major, props.minor)
print("Is half precision supported:", torch.cuda.is_available() and props.major >= 5 and (props.major > 5 or props.minor >= 3))
print("Has Tensor Cores:", props.major >= 7)
print("Supports torch.cuda.is_bf16_supported():", torch.cuda.is_bf16_supported())