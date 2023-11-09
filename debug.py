import torch

# Create your initial tensor with shape [5, 512]
initial_tensor = torch.randn(5, 512)

# Desired shape after padding
desired_shape = (10, 512)

# Calculate the amount of padding needed for the first dimension (rows)
padding_rows = max(desired_shape[0] - initial_tensor.shape[0], 0)

# Use torch.nn.functional.pad to pad the tensor with zeros
padded_tensor = torch.nn.functional.pad(initial_tensor, (0, 0, 0, padding_rows))

print(padded_tensor)  # Should print [10, 512]
