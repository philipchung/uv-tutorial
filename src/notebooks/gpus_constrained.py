# %%
# Add Constraints to Visible GPUs and then check visible GPU devices
import os

# NOTE: This must be done before you do any GPU code and is set for the entire script/notebook
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # <--- Only GPU0 & GPU1 visible by below script
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # <--- Only GPU1 visible by below script


import torch

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print the name of each GPU
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# %%
