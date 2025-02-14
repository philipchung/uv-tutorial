# %%
# Check GPU devices visible to your code before adding any constraints
import torch

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print the name of each GPU
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")


# %%
