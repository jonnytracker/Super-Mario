import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    current_device = torch.cuda.current_device()
    
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(current_device)
    
    # Get GPU properties
    gpu_properties = torch.cuda.get_device_properties(current_device)

    print(f"Current GPU: {gpu_name}")
    print(f"GPU Properties: {gpu_properties}")
else:
    print("GPU is not available. Using CPU.")
