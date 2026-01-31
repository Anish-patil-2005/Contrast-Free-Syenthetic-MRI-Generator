import os
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, ToTensord, ConcatItemsd
)
from monai.data import Dataset, DataLoader

# ================= PERFORMANCE =================
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "./data/raw/train"
MODEL_SAVE_PATH = "./models/contrast_free_diffusion.pth"
os.makedirs("./models", exist_ok=True)

# ================= TRANSFORMS =================
prep = Compose([
    LoadImaged(keys=["t1", "flair", "bravo", "target"]),
    EnsureChannelFirstd(keys=["t1", "flair", "bravo", "target"]),
    ScaleIntensityd(keys=["t1", "flair", "bravo", "target"]),
    Resized(keys=["t1", "flair", "bravo", "target"], spatial_size=(192,192,96)),
    ConcatItemsd(keys=["t1","flair","bravo"], name="image"),
    ToTensord(keys=["image","target"])
])

def get_loader():
    patients = [f for f in os.listdir(TRAIN_PATH) if f.startswith("Mets_")]

    data = []
    for p in patients:
        path = os.path.join(TRAIN_PATH,p)
        data.append({
            "t1": os.path.join(path,"t1_pre.nii.gz"),
            "flair": os.path.join(path,"flair.nii.gz"),
            "bravo": os.path.join(path,"bravo.nii.gz"),
            "target": os.path.join(path,"t1_gd.nii.gz")
        })

    print(f"📂 Found {len(data)} patients")

    ds = Dataset(data=data, transform=prep)

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

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

scheduler = DDPMScheduler(num_train_timesteps=250)
scheduler.betas = scheduler.betas.to(DEVICE)
scheduler.alphas = scheduler.alphas.to(DEVICE)
scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)



optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scaler = GradScaler("cuda")

# resume
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH,map_location=DEVICE))
    print("✅ Resumed checkpoint")

# ================= TRAIN =================
def train():
    loader = get_loader()

    for epoch in range(20):

        for i,batch in enumerate(loader):

            slice_idx = torch.randint(20,80,(1,)).item()

            inputs = batch["image"][:,:,:,:,slice_idx].to(DEVICE)
            target = batch["target"][:,:,:,:,slice_idx].to(DEVICE)

            noise = torch.randn_like(target)
            t = torch.randint(0,scheduler.num_train_timesteps,(1,),device=DEVICE).long()

            noisy = scheduler.add_noise(target,noise,t)

            model_input = torch.cat([inputs,noisy],dim=1)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                pred = model(model_input,t)
                x0 = scheduler.step(pred, t, noisy)[0]
                loss = F.mse_loss(pred, noise) + 0.01 * F.l1_loss(x0, target)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i%10==0:
                print(f"Epoch {epoch} | Step {i} | Loss {loss.item():.4f}")

        torch.save(model.state_dict(),MODEL_SAVE_PATH)
        print(f"💾 Saved epoch {epoch}")

if __name__=="__main__":
    train()
