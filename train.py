import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, Resized, ToTensord, ConcatItemsd
)
from monai.data import Dataset, DataLoader

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Using absolute path to avoid any confusion
TRAIN_PATH = r"C:\Full Stack Development\Contrast_Free_Syenthetic_MRI\data\raw\train"
MODEL_SAVE_PATH = "./models/contrast_free_diffusion.pth"

os.makedirs("./models", exist_ok=True)

# --- 1. DATA PIPELINE ---
prep_transforms = Compose([
    LoadImaged(keys=["t1", "flair", "bravo", "target"]),
    EnsureChannelFirstd(keys=["t1", "flair", "bravo", "target"]),
    ScaleIntensityd(keys=["t1", "flair", "bravo", "target"]),
    # 256x256 is standard, 128 slices depth
    Resized(keys=["t1", "flair", "bravo", "target"], spatial_size=(256, 256, 128)), 
    ConcatItemsd(keys=["t1", "flair", "bravo"], name="image"),
    ToTensord(keys=["image", "target"])
])

def get_loader():
    # This looks at your Mets_005, Mets_010, etc. folders
    patient_folders = [f for f in os.listdir(TRAIN_PATH) if f.startswith("Mets_")]
    
    data_dicts = []
    for p in patient_folders:
        p_path = os.path.join(TRAIN_PATH, p)
        data_dicts.append({
            "t1": os.path.join(p_path, "t1_pre.nii.gz"),    
            "flair": os.path.join(p_path, "flair.nii.gz"),
            "bravo": os.path.join(p_path, "bravo.nii.gz"),
            "target": os.path.join(p_path, "t1_gd.nii.gz")
        })
    
    print(f"📂 Found {len(data_dicts)} patients for training.")
    ds = Dataset(data=data_dicts, transform=prep_transforms)
    # batch_size=1 is recommended for medical 3D images due to VRAM limits
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

# --- 2. MODEL SETUP ---
model = DiffusionModelUNet(
    spatial_dims=2, 
    in_channels=4,   # 3 (input images) + 1 (noisy target)
    out_channels=1,  # 1 (the predicted clean T1-Gd)
    channels=(64, 128, 256, 512),
    attention_levels=(False, False, True, True),
    num_res_blocks=2, 
    num_head_channels=32,
).to(DEVICE)

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# --- 3. TRAINING ENGINE ---
def train():
    print(f"🚀 Training starting on: {DEVICE}")
    loader = get_loader()

    for epoch in range(100):
        for i, batch in enumerate(loader):
            # We train on 2D slices to save memory
            # Slice 70 is usually the center of the brain volume
            slice_idx = 70 
            
            inputs_2d = batch["image"][:, :, :, :, slice_idx].to(DEVICE)
            target_2d = batch["target"][:, :, :, :, slice_idx].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Diffusion Logic
            noise = torch.randn_like(target_2d).to(DEVICE)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,), device=DEVICE).long()
            noisy_target = scheduler.add_noise(original_samples=target_2d, noise=noise, timesteps=timesteps)
            
            # Combine multi-modal inputs with noisy target
            combined_input = torch.cat([inputs_2d, noisy_target], dim=1)

            with autocast():
                noise_pred = model(x=combined_input, timesteps=timesteps)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Saved Model Checkpoint: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()