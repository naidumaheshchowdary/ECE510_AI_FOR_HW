import torch
from torchvision import models
from torchinfo import summary
import os

# Ensure output directory exists
#output_dir = "codefest/cf01/profiling"
#os.makedirs(output_dir, exist_ok=True)

# Load ResNet-18
model = models.resnet18(weights=None)

# Set to eval mode (important for inference profiling)
model.eval()

# Generate torchinfo summary
profile = summary(
    model,
    input_size=(1, 3, 224, 224),  # batch=1, 3x224x224
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    verbose=1
)

# Save full output to file
with open("profiling/resnet18_profile.txt", "w", encoding="utf-8") as f:
    f.write(str(profile))

print(f"Profile saved Successfully")