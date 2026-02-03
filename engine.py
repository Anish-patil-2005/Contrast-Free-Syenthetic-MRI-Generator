import os
import torch
import numpy as np
import uuid
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/contrast_free_diffusion.pth"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    model = DiffusionModelUNet(
        spatial_dims=2, in_channels=4, out_channels=1,
        channels=(64, 128, 256, 512), attention_levels=(False, False, True, True),
        num_res_blocks=2, num_head_channels=32,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def run_synthesis_logic(inputs, n_samples=5, prog_bar=None, status_txt=None):
    model = load_model()
    scheduler = DDPMScheduler(num_train_timesteps=250)
    session_id = str(uuid.uuid4())[:8]
    all_runs = []

    with torch.no_grad():
        for i in range(n_samples):
            sample = torch.randn((1, 1, 192, 192)).to(DEVICE)
            
            for step_idx, t in enumerate(scheduler.timesteps):
                t_tensor = torch.tensor([t], device=DEVICE).long()
                model_input = torch.cat([inputs, sample], dim=1)
                noise_pred = model(model_input, t_tensor)
                output = scheduler.step(noise_pred, t, sample)
                sample = output[0] if isinstance(output, tuple) else output.prev_sample
                
                # Update UI Progress
                if prog_bar and status_txt:
                    # Current step across all runs
                    total_steps_done = (i * 250) + (step_idx + 1)
                    total_expected = n_samples * 250
                    prog_bar.progress(total_steps_done / total_expected)
                    status_txt.markdown(f"**Run {i+1}/{n_samples}** | Denoising Step: `{step_idx+1}/250`")
            
            all_runs.append(sample.cpu().numpy().squeeze())

    all_runs = np.array(all_runs)
    standard_synthetic = all_runs[0] 
    uncertainty_map = np.std(all_runs, axis=0)
    standard_synthetic = (standard_synthetic - standard_synthetic.min()) / (standard_synthetic.max() - standard_synthetic.min() + 1e-8)

    np.save(os.path.join(OUTPUT_DIR, f"synth_{session_id}.npy"), standard_synthetic)
    np.save(os.path.join(OUTPUT_DIR, f"uncert_{session_id}.npy"), uncertainty_map)

    return standard_synthetic, uncertainty_map, session_id