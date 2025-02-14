# %%
from utils import load_environment

from main_project.run_r1 import main

load_environment()

# Load Model on GPU1 by constraining visibility to GPU1 only.
# Then run prediction.
output_text = main(
    input_text="Which word has more 'r' letters? 'strawberry' or 'raspberry'?",
    visible_gpus=1,
    device_index=0,
)
print(output_text)

# %%
