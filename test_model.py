import torch
import matplotlib
matplotlib.use('TkAgg')  # Force a window-compatible backend
import matplotlib.pyplot as plt
import os
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from train import prep_transforms, DEVICE # Import the prep steps

# 1. SETTINGS
MODEL_PATH = "./models/contrast_free_diffusion.pth"
# Use your TEST folder for the final exam
TEST_PATH = r"C:\Full Stack Development\Contrast_Free_Syenthetic_MRI\data\raw\test"

# 2. LOAD THE MODEL (CPU SAFE)
model = DiffusionModelUNet(
    spatial_dims=2, in_channels=4, out_channels=1,
    channels=(64, 128, 256, 512), attention_levels=(False, False, True, True),
    num_res_blocks=2, num_head_channels=32,
).to(DEVICE)

# Load the brain data even if it was trained on a GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 3. GET ONE IMAGE MANUALLY
patient_folder = [f for f in os.listdir(TEST_PATH) if f.startswith("Mets_")][0]
p_path = os.path.join(TEST_PATH, patient_folder)

# Prepare the data dictionary
data_dict = {
    "t1": os.path.join(p_path, "t1_pre.nii.gz"),
    "flair": os.path.join(p_path, "flair.nii.gz"),
    "bravo": os.path.join(p_path, "bravo.nii.gz"),
    "target": os.path.join(p_path, "t1_gd.nii.gz")
}

# Run the preprocessing (Resizing, Scaling, etc.)
processed_data = prep_transforms(data_dict)
inputs_2d = processed_data["image"][:, :, :, 70].unsqueeze(0).to(DEVICE)
real_target = processed_data["target"][:, :, :, 70].unsqueeze(0).to(DEVICE)

# 4. GENERATE (This takes ~1 minute on CPU)
print("🧠 AI is drawing... Please wait about 60 seconds.")
current_img = torch.randn_like(real_target).to(DEVICE)

# 4. GENERATE (Optimized for CPU speed)
print("🧠 AI is drawing... reducing steps to 20 for speed.")
current_img = torch.randn_like(real_target).to(DEVICE)

with torch.no_grad():
    # Only 20 steps makes it MUCH faster on CPU
    scheduler.set_timesteps(num_inference_steps=20) 
    for t in scheduler.timesteps:
        # Progress bar so you know it's not frozen
        print(f"Step {t.item()} / 20", end="\r")
        
        combined_input = torch.cat([inputs_2d, current_img], dim=1)
        model_output = model(x=combined_input, timesteps=torch.Tensor((t,)).to(DEVICE).long())
        current_img, _ = scheduler.step(model_output, t, current_img)
print("\n✅ Drawing complete!")

# 5. SAVE AND PLOT
save_path = "test_result.png"
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(inputs_2d[0, 0].cpu(), cmap="gray"); plt.title("Input (T1)")
plt.subplot(1, 3, 2); plt.imshow(current_img[0, 0].cpu(), cmap="gray"); plt.title("AI Synthetic T1-Gd")
plt.subplot(1, 3, 3); plt.imshow(real_target[0, 0].cpu(), cmap="gray"); plt.title("Real T1-Gd")

plt.savefig(save_path) # This creates a file in your folder
print(f"✅ Success! Image saved as: {os.path.abspath(save_path)}")
plt.show() # Try to show it anyway, but now we have a backup!