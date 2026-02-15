import os
import torch
import numpy as np
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, ToTensord, ConcatItemsd
)
from monai.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_PATH = "./data/raw/test"
MODEL_PATH = "./models/contrast_free_diffusion.pth"

# ================= TRANSFORMS =================
prep = Compose([
    LoadImaged(keys=["t1", "flair", "bravo", "target"]),
    EnsureChannelFirstd(keys=["t1", "flair", "bravo", "target"]),
    ScaleIntensityd(keys=["t1", "flair", "bravo", "target"]),
    Resized(keys=["t1", "flair", "bravo", "target"], spatial_size=(192,192,96)),
    ConcatItemsd(keys=["t1","flair","bravo"], name="image"),
    ToTensord(keys=["image","target"])
])

# ================= LOAD MODEL =================
model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=4,
    out_channels=1,
    channels=(64,128,256,512),
    attention_levels=(False,False,True,True),
    num_res_blocks=2,
    num_head_channels=32,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=250)
scheduler.set_timesteps(250)

# ================= METRIC STORAGE =================
mae_list = []
mse_list = []
psnr_list = []
hist_list = []

patients = [f for f in os.listdir(TEST_PATH) if f.startswith("Mets_")]
patients = patients[:10]  # First 10 patients

slice_idx = 70

print("🔎 Evaluating 10 Patients...\n")

for patient in patients:

    p = os.path.join(TEST_PATH, patient)

    data = [{
        "t1": os.path.join(p,"t1_pre.nii.gz"),
        "flair": os.path.join(p,"flair.nii.gz"),
        "bravo": os.path.join(p,"bravo.nii.gz"),
        "target": os.path.join(p,"t1_gd.nii.gz")
    }]

    ds = Dataset(data=data, transform=prep)
    loader = DataLoader(ds, batch_size=1)

    batch = next(iter(loader))

    inputs = batch["image"][:,:,:,:,slice_idx].to(DEVICE)
    real = batch["target"][:,:,:,:,slice_idx].to(DEVICE)

    # ================= SAMPLING =================
    with torch.no_grad():
        sample = torch.randn_like(real).to(DEVICE)

        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], device=DEVICE).long()
            model_input = torch.cat([inputs, sample], dim=1)
            noise_pred = model(model_input, t_tensor)
            output = scheduler.step(noise_pred, t, sample)
            sample = output[0] if isinstance(output, tuple) else output.prev_sample

    fake = sample

    # Clamp
    fake = torch.clamp(fake, 0.0, 1.0)

    # ================= METRICS =================
    mae = torch.mean(torch.abs(fake - real)).item()
    mse = torch.mean((fake - real)**2).item()

    fake_np = fake[0,0].cpu().numpy()
    real_np = real[0,0].cpu().numpy()

    psnr_val = psnr(real_np, fake_np, data_range=1.0)

    # Histogram Similarity (Bhattacharyya Coefficient)
    hist_fake, _ = np.histogram(fake_np.flatten(), bins=64, range=(0,1), density=True)
    hist_real, _ = np.histogram(real_np.flatten(), bins=64, range=(0,1), density=True)

    hist_fake = hist_fake / (hist_fake.sum() + 1e-8)
    hist_real = hist_real / (hist_real.sum() + 1e-8)

    hist_similarity = np.sum(np.sqrt(hist_fake * hist_real))

    mae_list.append(mae)
    mse_list.append(mse)
    psnr_list.append(psnr_val)
    hist_list.append(hist_similarity)

# ================= FINAL AVERAGE =================
print("===== FINAL AVERAGED RESULTS (10 Patients) =====")
print(f"MAE:  {np.mean(mae_list):.4f}")
print(f"MSE:  {np.mean(mse_list):.4f}")
print(f"PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Histogram Similarity: {np.mean(hist_list):.4f}")
