# Import torch
import torch

# Create random tensor
tensor = torch.rand(7, 7)
tensor

# Create another random tensor
another_tensor = torch.rand(1, 7)

# Perform matrix multiplication 
x = torch.matmul(tensor, another_tensor.T)
x

# Set manual seed
RANDOM_SEED = 0

# Create two random tensors
torch.manual_seed(RANDOM_SEED)
tensor_A = torch.rand(7, 7)
tensor_B = torch.rand(1, 7)
# Matrix multiply tensors
x = torch.matmul(tensor_A, tensor_B.T)
x, x.shape

# Set random seed on the GPU
torch.cuda.manual_seed(1234)

# Set random seed
torch.manual_seed(1234)

# Check for access to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create two random tensors on GPU
tensor_C = torch.rand(2, 3).to(device)
tensor_D = torch.rand(2, 3).to(device)
print(f"Device: {device}")
print(tensor_C)
print(tensor_D)

# Perform matmul on tensor_A and tensor_B
z = torch.matmul(tensor_C, tensor_D.T)
z, z.shape

# Find max
print(torch.max(z))
#or
print(z.max())
# Find min
print(torch.min(z))
#or
print(z.min())

# Find arg max
print(torch.argmax(z))
#or
print(z.argmax())
# Find arg min
print(torch.argmin(z))
#or
print(z.argmin())

# Set seed
torch.manual_seed(7)

# Create random tensor
random_tensor = torch.rand(1, 1, 1, 10)

# Remove single dimensions
random_tensor_squeezed = random_tensor.squeeze()

# Print out tensors and their shapes
print(random_tensor, random_tensor.shape)
print(random_tensor_squeezed, random_tensor_squeezed.shape)