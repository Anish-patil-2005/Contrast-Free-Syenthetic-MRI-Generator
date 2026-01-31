import torch

# 1. Define your model architecture (replace 'YourModelClass' with your actual class)
# model = YourModelClass(...) 

# 2. Load the state dictionary from the .pth file
# Ensure you replace "model.pth" with the actual path to your file
model_state_dict = torch.load("./models/contrast_free_diffusion.pth")

# 3. Load the state dictionary into your model
# model.load_state_dict(model_state_dict) 

# You can also simply load the dictionary to inspect keys if needed
print(model_state_dict.keys())
