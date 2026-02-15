import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(fake, real):
    """
    fake: torch tensor [1,1,H,W]
    real: torch tensor [1,1,H,W]
    """

    # Move to CPU
    fake = fake.detach().cpu()
    real = real.detach().cpu()

    fake_np = fake[0, 0].numpy()
    real_np = real[0, 0].numpy()

    # -----------------------------
    # MAE (Primary Contrast Metric)
    # -----------------------------
    mae = F.l1_loss(fake, real).item()

    # -----------------------------
    # MSE
    # -----------------------------
    mse = F.mse_loss(fake, real).item()

    # -----------------------------
    # PSNR
    # -----------------------------
    val_psnr = psnr(real_np, fake_np, data_range=1.0)

    # -----------------------------
    # Histogram Similarity (Contrast Distribution)
    # -----------------------------
    # Histogram Similarity (Bhattacharyya Coefficient)

    hist_fake, _ = np.histogram(fake_np.flatten(), bins=64, range=(0, 1), density=True)
    hist_real, _ = np.histogram(real_np.flatten(), bins=64, range=(0, 1), density=True)

    # Normalize histograms
    hist_fake = hist_fake / (hist_fake.sum() + 1e-8)
    hist_real = hist_real / (hist_real.sum() + 1e-8)

    # Bhattacharyya coefficient
    hist_similarity = np.sum(np.sqrt(hist_fake * hist_real))

    


    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "PSNR": float(val_psnr),
        "HistSimilarity": float(hist_similarity)

    }
