"""Shared contracts for the seven-channel conditioning path (no new dependencies)."""
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_depth(path):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 3:
        raise ValueError(f"Depth must be single-channel: {path}")
    if array.dtype == np.uint8:
        scale = 255.0
    elif array.dtype == np.uint16 or (array.dtype == np.int32 and array.min() >= 0 and array.max() <= 65535):
        scale = 65535.0  # Pillow may expose a 16-bit PNG as int32.
    elif np.issubdtype(array.dtype, np.floating):
        scale = 1.0
    else:
        raise ValueError(f"Unsupported depth encoding: {array.dtype}")
    array = array.astype(np.float32) / scale
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Conditioning depth must be finite and normalized to [0, 1].")
    return torch.from_numpy(array).unsqueeze(0)


def load_normal(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def clip_preprocess(rgb):
    # Fixed input contract: RGB in [-1, 1], including all-positive images.
    rgb = F.interpolate(rgb.float(), (224, 224), mode='bicubic', align_corners=True)
    rgb = (rgb + 1) / 2
    mean = rgb.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = rgb.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (rgb - mean) / std


def diffusion_target(scheduler, clean, noise, timesteps):
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return noise
    if kind == 'v_prediction':
        return scheduler.get_velocity(clean, noise, timesteps)
    if kind == 'sample':
        return clean
    raise ValueError(f"Unsupported prediction_type: {kind}")


def predict_clean(scheduler, noisy, prediction, timesteps):
    a = scheduler.alphas_cumprod.to(noisy.device)[timesteps].float().view(-1, 1, 1, 1)
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return (noisy.float() - (1-a).sqrt()*prediction.float()) / a.sqrt().clamp_min(1e-8)
    if kind == 'v_prediction':
        return a.sqrt()*noisy.float() - (1-a).sqrt()*prediction.float()
    if kind == 'sample':
        return prediction.float()
    raise ValueError(f"Unsupported prediction_type: {kind}")


def foreground_loss(prediction, target, mask):
    """Per-view normalization prevents a large silhouette dominating other views."""
    error = (prediction.float() - target.float()).square()
    mask = F.interpolate(mask.float(), error.shape[-2:], mode='area')
    numerator = (error * mask).sum((1, 2, 3))
    denominator = mask.sum((1, 2, 3)) * error.shape[1]
    valid = denominator > 0
    if not valid.any():
        return prediction.sum() * 0
    return (numerator[valid] / denominator[valid]).mean()


class ConditioningEncoder(torch.nn.Module):
    """Use forward(), so Accelerate/DDP can observe the trainable VAE path."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        return self.vae.encode(image.float()).latent_dist.mode() * self.vae.config.scaling_factor


def canonical_poses(poses, reference):
    """Map the first target/context camera to the inference camera, preserving relatives."""
    poses = poses.clone()
    poses[:, :3, 3] /= poses[:, :3, 3].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference = reference.clone()
    reference[:3, 3] /= reference[:3, 3].norm().clamp_min(1e-8)
    return reference.unsqueeze(0) @ torch.linalg.inv(poses[0]).unsqueeze(0) @ poses
