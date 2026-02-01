import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
# Ensure your training script is in the same folder to import 'prep'
from train import prep, DEVICE 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================= 1. CONFIGURATION =================
MODEL_PATH = "./models/contrast_free_diffusion.pth"
TEST_PATH = "./data/raw/test"
N_SAMPLES = 5  # Number of Monte Carlo passes (higher = more accurate)
SLICE_IDX = 70

# ================= 2. LOAD MODEL & DATA =================
model = DiffusionModelUNet(
    spatial_dims=2, in_channels=4, out_channels=1,
    channels=(64, 128, 256, 512), attention_levels=(False, False, True, True),
    num_res_blocks=2, num_head_channels=32,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=250)

# Load first patient from test set
patient = [f for f in os.listdir(TEST_PATH) if f.startswith("Mets_")][0]
p_path = os.path.join(TEST_PATH, patient)
data = {"t1": os.path.join(p_path, "t1_pre.nii.gz"), 
        "flair": os.path.join(p_path, "flair.nii.gz"),
        "bravo": os.path.join(p_path, "bravo.nii.gz"),
        "target": os.path.join(p_path, "t1_gd.nii.gz")}

processed = prep(data)
inputs = processed["image"][:, :, :, SLICE_IDX].unsqueeze(0).to(DEVICE)
real = processed["target"][:, :, :, SLICE_IDX].unsqueeze(0).to(DEVICE)

# ================= 3. STOCHASTIC SAMPLING =================
all_runs = []

print(f"🚀 Starting Uncertainty Mapping ({N_SAMPLES} passes)...")

for i in range(N_SAMPLES):
    # Each run starts with a NEW random noise seed (Stochasticity)
    sample = torch.randn_like(real).to(DEVICE)
    
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc=f"Run {i+1}/{N_SAMPLES}", leave=False):
            t_batch = torch.tensor([t], device=DEVICE).long()
            model_input = torch.cat([inputs, sample], dim=1)
            noise_pred = model(model_input, t_batch)
            
            output = scheduler.step(noise_pred, t, sample)
            sample = output[0] if isinstance(output, tuple) else output.prev_sample
            
    all_runs.append(sample.cpu().numpy().squeeze())

# ================= 4. MATHEMATICAL CALCULATION =================
all_runs = np.array(all_runs)  # Shape: [N_SAMPLES, H, W]

# Step 1: Mean (The "Best Guess" Image)
mean_synthetic = np.mean(all_runs, axis=0)

# Step 2: Standard Deviation (The Uncertainty Map)
uncertainty_map = np.std(all_runs, axis=0)

# # ================= 5. VISUALIZATION =================
# plt.figure(figsize=(16, 5))

# plt.subplot(1, 4, 1)
# plt.title("Input T1")
# plt.imshow(inputs[0, 0].cpu(), cmap="gray")
# plt.axis("off")

# plt.subplot(1, 4, 2)
# plt.title("Synthetic T1-Gd (Mean)")
# plt.imshow(mean_synthetic, cmap="gray")
# plt.axis("off")

# plt.subplot(1, 4, 3)
# plt.title("Uncertainty Map (Std Dev)")
# # We use 'hot' colormap to make high-variance areas look like a heatmap
# plt.imshow(uncertainty_map, cmap="hot")
# plt.colorbar(label="Uncertainty Level")
# plt.axis("off")

# plt.subplot(1, 4, 4)
# plt.title("Ground Truth")
# plt.imshow(real[0, 0].cpu(), cmap="gray")
# plt.axis("off")

# plt.tight_layout()
# plt.savefig("uncertainty_result.png")
# print("✅ Uncertainty map saved as uncertainty_result.png")
# plt.show()



# --- 1. Calculate Error Map ---
real_np = real[0, 0].cpu().numpy()
error_map = np.abs(mean_synthetic - real_np)

# --- 2. Calculate Metrics ---
# We normalize to [0,1] for standard metric calculation
data_range = real_np.max() - real_np.min()
val_ssim = ssim(real_np, mean_synthetic, data_range=data_range)
val_psnr = psnr(real_np, mean_synthetic, data_range=data_range)

print(f"\n📊 CLINICAL METRICS:")
print(f"   ➤ SSIM: {val_ssim:.4f} (Higher is better, max 1.0)")
print(f"   ➤ PSNR: {val_psnr:.2f} dB (Higher is better)")

# --- 3. Professional 5-Panel Display ---
plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.title("Input T1")
plt.imshow(inputs[0, 0].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.title("Synthetic (Mean)")
plt.imshow(mean_synthetic, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.title("Ground Truth")
plt.imshow(real_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.title("Uncertainty (AI Doubt)")
plt.imshow(uncertainty_map, cmap="hot")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis("off")

plt.subplot(1, 5, 5)
plt.title("Error Map (AI vs Real)")
plt.imshow(error_map, cmap="inferno") # Inferno clearly shows intensity misses
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis("off")

plt.tight_layout()
plt.show()