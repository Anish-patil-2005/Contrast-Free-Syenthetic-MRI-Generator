import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import plotting
import matplotlib.pyplot as plt
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from train import prep, DEVICE 

# ================= 1. CONFIGURATION =================
MODEL_PATH = "./models/contrast_free_diffusion.pth"
TEST_PATH = "./data/raw/test"
OUTPUT_NAME = "synthetic_3d_volume.nii.gz"
REPORT_NAME = "3d_ortho_view.png"
USE_FP16 = True  # Set to True to speed up generation on compatible GPUs

# ================= 2. LOAD MODEL =================
model = DiffusionModelUNet(
    spatial_dims=2, in_channels=4, out_channels=1,
    channels=(64, 128, 256, 512), attention_levels=(False, False, True, True),
    num_res_blocks=2, num_head_channels=32,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

if USE_FP16 and DEVICE == "cuda":
    model.half() # Convert weights to FP16 for speed

scheduler = DDPMScheduler(num_train_timesteps=250)

# Load Patient Data
patient = [f for f in os.listdir(TEST_PATH) if f.startswith("Mets_")][0]
p_path = os.path.join(TEST_PATH, patient)
data = {"t1": os.path.join(p_path, "t1_pre.nii.gz"), 
        "flair": os.path.join(p_path, "flair.nii.gz"),
        "bravo": os.path.join(p_path, "bravo.nii.gz"),
        "target": os.path.join(p_path, "t1_gd.nii.gz")}

processed = prep(data)
full_input_volume = processed["image"] # [Channels, H, W, D]
total_slices = full_input_volume.shape[-1]

# ================= 3. VOLUMETRIC INFERENCE =================
synthetic_volume = np.zeros((192, 192, total_slices))

print(f"🚀 Generating 3D Synthetic Volume: {patient}")

with torch.no_grad():
    # We process slices 25 to 85 (the core brain region) to save time
    for s in tqdm(range(25, 86), desc="Processing Slices"):
        slice_input = full_input_volume[:, :, :, s].unsqueeze(0).to(DEVICE)
        
        if USE_FP16 and DEVICE == "cuda":
            slice_input = slice_input.half()

        sample = torch.randn((1, 1, 192, 192)).to(DEVICE)
        if USE_FP16 and DEVICE == "cuda":
            sample = sample.half()

        for t in scheduler.timesteps:
            t_batch = torch.tensor([t], device=DEVICE).long()
            model_input = torch.cat([slice_input, sample], dim=1)
            noise_pred = model(model_input, t_batch)
            output = scheduler.step(noise_pred, t, sample)
            sample = output[0] if isinstance(output, tuple) else output.prev_sample
            
        synthetic_volume[:, :, s] = sample.cpu().float().numpy().squeeze()

# ================= 4. SAVE & VISUALIZE =================
# Load original header to maintain correct orientation
ref_img = nib.load(data["t1"])
final_nifti = nib.Nifti1Image(synthetic_volume, ref_img.affine)
nib.save(final_nifti, OUTPUT_NAME)
print(f"✅ Volume Saved: {OUTPUT_NAME}")

# Create the 3-plane view (Axial, Sagittal, Coronal)
print("📊 Generating Orthographic Report...")
display = plotting.plot_anat(
    final_nifti, 
    display_mode='ortho', 
    title="AI Synthetic T1-Gd (3D View)", 
    cut_coords=None # Automatically finds the center
)
display.savefig(REPORT_NAME)
plt.show()

print(f"✅ Report Saved: {REPORT_NAME}")