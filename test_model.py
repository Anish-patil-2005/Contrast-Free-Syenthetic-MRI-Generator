import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, ToTensord, ConcatItemsd
)
from monai.data import Dataset, DataLoader

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

# ================= LOAD ONE PATIENT =================
patient = [f for f in os.listdir(TEST_PATH) if f.startswith("Mets_")][0]
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

slice_idx = 70

inputs = batch["image"][:,:,:,:,slice_idx].to(DEVICE)
real = batch["target"][:,:,:,:,slice_idx].to(DEVICE)

# ================= MODEL =================
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

# ================= DIFFUSION SAMPLING =================
with torch.no_grad():
    sample = torch.randn_like(real)

    for t in scheduler.timesteps:
        t_tensor = torch.tensor([t], device=DEVICE)

        model_input = torch.cat([inputs, sample], dim=1)
        noise_pred = model(model_input, t_tensor)

        # MONAI older version returns tuple
        sample = scheduler.step(noise_pred, t, sample)[1]

fake = sample

# ================= NORMALIZATION =================
fake = (fake - fake.min()) / (fake.max() - fake.min())
real = (real - real.min()) / (real.max() - real.min())
inp = inputs[:,0:1]
inp = (inp - inp.min()) / (inp.max() - inp.min())

# ================= DISPLAY =================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input T1")
plt.imshow(inp[0,0].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("AI Synthetic T1-Gd")
plt.imshow(fake[0,0].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Real T1-Gd")
plt.imshow(real[0,0].cpu(), cmap="gray")
plt.axis("off")

plt.show()
