import torch
import numpy as np

# -------------------------------
# Load tensors
# -------------------------------
ground_truth_pt = "/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/ls2s2_preprocessed/landsat_3dates.pt"
model_output_pt = "/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/swin_output_full.pt"

stacked_tensor = torch.load(ground_truth_pt)
ground_truth = stacked_tensor[2].numpy()  # target image

model_output = torch.load(model_output_pt).numpy()

ground_truth = np.clip(ground_truth, 0, 1)
model_output = model_output / 10000.0
model_output = np.clip(model_output, 0, 1)

assert ground_truth.shape == model_output.shape, "Ground truth and model output must have same shape!"

# -------------------------------
# Helper functions
# -------------------------------
def rmse_np(gt, pred):
    return np.sqrt(np.mean((gt - pred)**2))

def r2_np(gt, pred):
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - np.mean(gt))**2)
    return 1 - ss_res / (ss_tot + 1e-100)

def psnr_np(gt, pred):
    return 20 * np.log10(1.0 / (rmse_np(gt, pred) + 1e-100))

def ergas_np(gt, pred):
    return np.sqrt(np.mean(((rmse_np(gt, pred) / (np.mean(gt) + 1e-100))**2))) * 100

def sam_np(gt, pred):
    gt_vec = gt.reshape(-1)
    pred_vec = pred.reshape(-1)
    dot = np.sum(gt_vec * pred_vec)
    norm_gt = np.sqrt(np.sum(gt_vec**2))
    norm_pred = np.sqrt(np.sum(pred_vec**2))
    angle = np.arccos(np.clip(dot / (norm_gt * norm_pred + 1e-100), -1, 1))
    return np.degrees(angle)

def ssim_np(img1, img2, K1=0.01, K2=0.03, L=1.0, window_size=11):
    """Simplified SSIM for large images using a single global mean/std/cov."""
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    C1 = (K1*L)**2
    C2 = (K2*L)**2

    ssim_val = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))
    return ssim_val

# -------------------------------
# Compute metrics per band
# -------------------------------
num_bands = ground_truth.shape[0]

rmse_list = []
r2_list = []
ssim_list = []
sam_list = []
psnr_list = []
ergas_list = []

for b in range(num_bands):
    gt = ground_truth[b]
    pred = model_output[b]

    rmse_val = rmse_np(gt, pred)
    rmse_list.append(np.clip(rmse_val, 0, 1))

    r2_val = r2_np(gt, pred)
    r2_list.append(np.clip(r2_val, 0, 1))

    ssim_val = ssim_np(gt, pred)
    ssim_list.append(np.clip(ssim_val, 0, 1))

    sam_val = sam_np(gt, pred) /90.0
    sam_list.append(np.clip(sam_val, 0, 1))

    psnr_val = psnr_np(gt, pred) / 50.0
    psnr_list.append(np.clip(psnr_val, 0, 1))

    ergas_val = ergas_np(gt, pred) / 100.0
    ergas_list.append(np.clip(ergas_val, 0, 1))

# -------------------------------
# Print results
# -------------------------------
print("\nPer-band metrics:")
for b in range(num_bands):
    print(f"Band {b+1}: RMSE={rmse_list[b]:.4f}, R²={r2_list[b]:.4f}, SSIM={ssim_list[b]:.4f}, "
          f"SAM={sam_list[b]:.4f}, PSNR={psnr_list[b]:.4f} dB, ERGAS={ergas_list[b]:.4f}")

print("\nAverage across all bands:")
print(f"RMSE={np.mean(rmse_list):.4f}, R²={np.mean(r2_list):.4f}, SSIM={np.mean(ssim_list):.4f}, "
      f"SAM={np.mean(sam_list):.4f}, PSNR={np.mean(psnr_list):.4f} dB, ERGAS={np.mean(ergas_list):.4f}")

