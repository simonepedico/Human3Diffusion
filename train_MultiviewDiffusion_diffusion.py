import argparse
from pathlib import Path
import os
import copy
from packaging import version
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import einops
import torch.nn.functional as F
import shutil
from typing import Union, List, Optional, Dict 

from huggingface_hub import list_repo_files, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from sklearn.model_selection import GroupShuffleSplit

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import logging
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
import diffusers
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer
from mvdream.mv_unet import MultiViewUNetModel
from core.dataset_human_imagedream import Imagedream_LGM_dataset
from mvdream.pipeline_imagedream import ImageDreamPipeline

logger = get_logger(__name__)

if is_wandb_available():
    os.environ["WANDB_MODE"] = "offline"
    import wandb

def get_path_size_str(path: Union[str, Path]) -> str:
    p = Path(path)
    if not p.exists():
        return "0 B"
    if p.is_file():
        total_bytes = p.stat().st_size
    else:
        total_bytes = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_bytes < 1024.0:
            return f"{total_bytes:.2f} {unit}"
        total_bytes /= 1024.0
    return f"{total_bytes:.2f} PB"

def CLIP_preprocess(x):
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Argparser for ImageDream fine-tuning script.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.15) 
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=0.5, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--set_grads_to_none", default=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="ashawkey/imagedream-ipmv-diffusers")
    parser.add_argument("--output_dir", type=str, default="mvd_pretrain")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()

def _encode_text_prompt(tokenizer, text_encoder, prompt, device, batch_size):
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    bs_embed, seq_len, _ = prompt_embeds.shape
    return prompt_embeds.repeat(1, batch_size, 1).view(bs_embed * batch_size, seq_len, -1)

def print_model_info(model):
    print("="*20)
    print("model name: ", type(model).__name__)
    print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
    print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
    print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

# =========================================================================
# CALCOLO DETERMINISTICO DELLA VALIDATION LOSS (Senza Moving Target)
# =========================================================================
def compute_val_loss(
    unet, vae, vae_gt, image_encoder, text_encoder, tokenizer, noise_scheduler, val_dataloader, accelerator, weight_dtype, args
):
    unet.eval()
    vae.eval()
    vae_gt.eval()

    val_loss_total = 0.0
    val_steps = 0
    
    eval_timesteps = [200, 500, 800]
    val_generator = torch.Generator(device=accelerator.device).manual_seed(42)

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            batch_size = batch['imagedream_images_gt'].shape[0]
            num_view = batch['imagedream_images_gt'].shape[1]
            actual_num_frames = num_view + 1

            gt_image = batch["imagedream_images_gt"].float() 
            input_image = batch["context_image"].squeeze(dim=1).float()
            context_depth = batch["context_depth"].squeeze(dim=1).float()
            context_normal = batch["context_normal"].squeeze(dim=1).float()
            gt_pose = batch["imagedream_cam_poses_gt"].float()
            text_prompt = ", 3d asset photorealistic human scan"

            fused_input = torch.cat([input_image, context_depth, context_normal], dim=1) 

            gt_image = einops.rearrange(gt_image, "b n c h w -> (b n) c h w")
            gt_pose = einops.rearrange(gt_pose, "b n x y -> (b n) x y")

            gt_latents = vae_gt.encode(gt_image.to(dtype=weight_dtype)).latent_dist.mode() * vae_gt.config.scaling_factor
            gt_latents = gt_latents.to(dtype=torch.float32)

            image_latents = vae.encode(fused_input.float()).latent_dist.mode() * vae.config.scaling_factor

            image_clip = CLIP_preprocess(fused_input[:, :3]).to(dtype=weight_dtype)
            image_clip_embedding = image_encoder(image_clip, output_hidden_states=True).hidden_states[-2]

            text_prompt_embeds = _encode_text_prompt(tokenizer, text_encoder, text_prompt, gt_latents.device, batch_size)

            clear_image_latents = einops.rearrange(image_latents, "b c h w -> b 1 c h w")

            camera_pose_ = gt_pose.view(batch_size, 4, 16)
            padding = [0] * (len(camera_pose_.shape) * 2)
            padding[-3] = 1
            camera = F.pad(camera_pose_, tuple(padding)).to(dtype=gt_latents.dtype, device=gt_latents.device)
            camera = einops.rearrange(camera, "b nv c -> (b nv) c")

            batch_step_loss = 0.0

            for t_val in eval_timesteps:
                noise = torch.randn(gt_latents.shape, generator=val_generator, device=gt_latents.device, dtype=gt_latents.dtype)
                timesteps = torch.full((batch_size * num_view,), t_val, device=gt_latents.device, dtype=torch.long)

                noisy_latents = noise_scheduler.add_noise(gt_latents.float(), noise.float(), timesteps).to(dtype=gt_latents.dtype)
                noisy_gt_latents = einops.rearrange(noisy_latents, "(b n) c h w -> b n c h w", n=num_view)

                noisy_latents_with_ip_latent = einops.rearrange(
                    torch.cat([noisy_gt_latents, clear_image_latents], dim=1), "b nv c h w -> (b nv) c h w"
                )

                timestep = einops.rearrange(timesteps, '(b n) -> b n', b=batch_size)
                timestep_with_ip = einops.rearrange(
                    torch.cat([timestep, timestep.new_zeros((batch_size, 1))], dim=1), 'b nv -> (b nv)', nv=actual_num_frames
                )

                unet_inputs = {
                    'x': noisy_latents_with_ip_latent,
                    'timesteps': timestep_with_ip,
                    'context': torch.cat([text_prompt_embeds] * actual_num_frames),
                    'num_frames': actual_num_frames,
                    'camera': camera,
                    'ip': torch.cat([image_clip_embedding] * actual_num_frames),
                    'ip_img': image_latents,
                }

                unet_noise_pred = unet.forward(**unet_inputs)

                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(gt_latents, noise, timesteps)

                unet_noise_pred = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
                unet_orthogonal_noise_pred = einops.rearrange(unet_noise_pred[:, :-1, :, :, :], "b nv c h w -> (b nv) c h w")

                loss_t = F.mse_loss(unet_orthogonal_noise_pred.float(), target.float(), reduction="mean")
                batch_step_loss += loss_t.item()

            val_loss_total += (batch_step_loss / len(eval_timesteps))
            val_steps += 1

    return val_loss_total / max(val_steps, 1)

def main(args):
    args.max_train_steps = None
    args.learning_rate = 1e-4
    args.tracker_project_name = "train_mvd_direct_7ch"
    args.num_gpu = 1
    args.gradient_accumulation_steps = 2
    args.use_ema = True
    args.enable_xformers_memory_efficient_attention = True
    args.resolution = 256

    args.output_dir = args.output_dir + "_bs_" + str(args.train_batch_size * args.num_gpu * args.gradient_accumulation_steps)
    args.tracker_project_name = args.tracker_project_name + "_bs_" + str(args.train_batch_size * args.num_gpu * args.gradient_accumulation_steps)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token, private=True
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    
    '''
    unet = MultiViewUNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    from safetensors.torch import load_file
    ckpt_mvd_2k2k = load_file('checkpoints/model.safetensors', device='cpu')
    state_dict = unet.state_dict()
    for k, v in ckpt_mvd_2k2k.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
    '''
    unet = MultiViewUNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    from safetensors.torch import load_file
    
    # 1. Carica entrambe le parti dei pesi
    try:
        ckpt_part1 = load_file('checkpoints/model.safetensors', device='cpu')
        ckpt_part2 = load_file('checkpoints/model_1.safetensors', device='cpu')
    except Exception as e:
        raise FileNotFoundError(f"Errore nel caricamento dei safetensors. Assicurati che model.safetensors e model_1.safetensors siano nella cartella 'checkpoints/'. Dettaglio: {e}")

    # 2. Unisci i due dizionari in un unico dizionario completo
    ckpt_mvd_2k2k = {**ckpt_part1, **ckpt_part2}
    
    # 3. Inietta i pesi completi nel modello
    state_dict = unet.state_dict()
    for k, v in ckpt_mvd_2k2k.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)

    # =========================================================================
    # VAE_GT: COPIA A 3 CANALI PER TARGET GROUND TRUTH (100% CONGELATO)
    # =========================================================================
    vae_gt = copy.deepcopy(vae)
    vae_gt.eval()
    vae_gt.requires_grad_(False)

    # =========================================================================
    # VAE CONDITIONING: ESPANSIONE PRIMO LAYER DA 3 A 7 CANALI
    # =========================================================================
    old_conv = vae.encoder.conv_in
    new_conv = nn.Conv2d(
        in_channels=7,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :].copy_(old_conv.weight)
        nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.02)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    vae.encoder.conv_in = new_conv

    # =========================================================================
    # CONGELAMENTO / SCONGELAMENTO PARAMETRI E LEARNING RATES
    # =========================================================================
    vae.eval()
    vae.requires_grad_(False)
    vae.encoder.conv_in.requires_grad_(True)

    image_encoder.eval()
    image_encoder.requires_grad_(False)

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)

    for param in unet.input_blocks[0][0].parameters():
        param.requires_grad = True

    for name, param in unet.named_parameters():
        if "to_k_ip" in name or "to_v_ip" in name or "attn2" in name:
            param.requires_grad = True

    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_class = torch.optim.AdamW
    
    vae_trainable_params = [p for p in vae.encoder.conv_in.parameters() if p.requires_grad]
    unet_trainable_params = [p for p in unet.parameters() if p.requires_grad]

    params_da_addestrare = [
        {"params": vae_trainable_params, "lr": args.learning_rate},
        {"params": unet_trainable_params, "lr": args.learning_rate * 0.3}
    ]

    optimizer = optimizer_class(
        params_da_addestrare,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.use_ema:
        ema_unet = EMAModel(unet_trainable_params, decay=0.9999)

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)

    class OptAdapter:
        def __init__(self, argparse_args):
            self.num_views = getattr(argparse_args, 'num_views', 4)
            self.num_input_views = getattr(argparse_args, 'num_input_views', 4)
            self.input_size = getattr(argparse_args, 'input_size', 256)
            self.output_size = getattr(argparse_args, 'output_size', 512)
            self.prob_grid_distortion = getattr(argparse_args, 'prob_grid_distortion', 0.3)
            self.prob_cam_jitter = getattr(argparse_args, 'prob_cam_jitter', 0.3)
            self.fovy = getattr(argparse_args, 'fovy', 30.0)
            self.zfar = getattr(argparse_args, 'zfar', 100.0)
            self.znear = getattr(argparse_args, 'znear', 0.01)
            self.cam_radius = getattr(argparse_args, 'cam_radius', 4.0)

    opt = OptAdapter(args)

    repo_id = "siiimo/tesiMagistrale"
    tutti_i_file = list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
    file_rgb = [f for f in tutti_i_file if "rgb_" in f]

    local_dataset_root = snapshot_download(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN, local_files_only=False)

    quadruple_locali = []
    for rgb_path in file_rgb:
        cartella_rel, nome_file = os.path.split(rgb_path)
        id_campione = nome_file.replace("rgb_", "").replace(".png", "")
        quadruple_locali.append({
            "id": id_campione,
            "rgb": os.path.join(local_dataset_root, rgb_path),
            "matrix": os.path.join(local_dataset_root, cartella_rel, f"{id_campione}_RT.txt"),
            "depth": os.path.join(local_dataset_root, cartella_rel, f"depth_{id_campione}.png"),
            "normal": os.path.join(local_dataset_root, cartella_rel, f"normal_{id_campione}.png"),
            "cartella_origine": os.path.join(local_dataset_root, cartella_rel)
        })

    subject_ids = [os.path.basename(q["cartella_origine"]).split('_')[0] for q in quadruple_locali]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, resto_idx = next(gss1.split(quadruple_locali, groups=subject_ids))

    train_paths = [quadruple_locali[i] for i in train_idx]
    resto_paths = [quadruple_locali[i] for i in resto_idx]

    resto_subject_ids = [os.path.basename(q["cartella_origine"]).split('_')[0] for q in resto_paths]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(resto_paths, groups=resto_subject_ids))

    val_paths = [resto_paths[i] for i in val_idx]
    test_paths = [resto_paths[i] for i in test_idx]

    logger.info(f"-> Distribuzione dataset:\n   - Train: {len(train_paths)} | - Val: {len(val_paths)} | - Test: {len(test_paths)}")

    train_dataset = Imagedream_LGM_dataset(lista_quadruple=train_paths, opt=opt, training=True, white_bg=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1)

    val_dataset = Imagedream_LGM_dataset(lista_quadruple=val_paths, opt=opt, training=False, white_bg=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_steps = args.max_train_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    unet, vae, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        unet, vae, optimizer, train_dataloader, val_dataloader, scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    vae_gt.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                batch_size = batch['imagedream_images_gt'].shape[0]
                num_view = batch['imagedream_images_gt'].shape[1]
                actual_num_frames = num_view + 1

                gt_image = batch["imagedream_images_gt"].float()
                input_image = batch["context_image"].squeeze(dim=1).float()
                context_depth = batch["context_depth"].squeeze(dim=1).float()
                context_normal = batch["context_normal"].squeeze(dim=1).float()
                gt_pose = batch["imagedream_cam_poses_gt"].float()
                text_prompt = ", 3d asset photorealistic human scan"

                fused_input = torch.cat([input_image, context_depth, context_normal], dim=1)

                gt_image = einops.rearrange(gt_image, "b n c h w -> (b n) c h w")
                gt_pose = einops.rearrange(gt_pose, "b n x y -> (b n) x y")

                with torch.no_grad():
                    gt_latents = vae_gt.encode(gt_image.to(dtype=weight_dtype)).latent_dist.sample() * vae_gt.config.scaling_factor
                    gt_latents = gt_latents.to(dtype=torch.float32)

                noise = torch.randn_like(gt_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=gt_latents.device).long()
                timesteps = einops.repeat(timesteps, 'b -> (b n)', n=num_view)

                noisy_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps).to(dtype=gt_latents.dtype)

                p_drop = args.conditioning_dropout_prob
                r = torch.rand(batch_size, device=gt_latents.device)

                drop_rgb = (r < p_drop).reshape(batch_size, 1, 1, 1)
                drop_maps = ((r >= p_drop) & (r < 2 * p_drop)).reshape(batch_size, 1, 1, 1)
                drop_all = ((r >= 2 * p_drop) & (r < 3 * p_drop)).reshape(batch_size, 1, 1, 1)

                blank_rgb = torch.ones(batch_size, 3, args.resolution, args.resolution, device=gt_latents.device)
                blank_maps = torch.zeros(batch_size, 4, args.resolution, args.resolution, device=gt_latents.device)

                curr_rgb = fused_input[:, :3]
                curr_maps = fused_input[:, 3:]

                curr_rgb = torch.where(drop_rgb | drop_all, blank_rgb, curr_rgb)
                curr_maps = torch.where(drop_maps | drop_all, blank_maps, curr_maps)

                new_input_after_dropout = torch.cat([curr_rgb, curr_maps], dim=1)

                image_clip = CLIP_preprocess(new_input_after_dropout[:, :3]).to(dtype=weight_dtype)
                image_clip_embedding = image_encoder(image_clip, output_hidden_states=True).hidden_states[-2]

                image_latents = vae.encode(new_input_after_dropout.float()).latent_dist.sample() * vae.config.scaling_factor

                text_prompt_embeds = _encode_text_prompt(tokenizer, text_encoder, text_prompt, gt_latents.device, batch_size)

                noisy_gt_latents = einops.rearrange(noisy_latents, "(b n) c h w -> b n c h w", n=num_view)
                clear_image_latents = einops.rearrange(image_latents, "b c h w -> b 1 c h w")
                noisy_latents_with_ip_latent = einops.rearrange(
                    torch.cat([noisy_gt_latents, clear_image_latents], dim=1), "b nv c h w -> (b nv) c h w"
                )
                
                timestep = einops.rearrange(timesteps, '(b n) -> b n', b=batch_size)
                timestep_with_ip = einops.rearrange(
                    torch.cat([timestep, timestep.new_zeros((batch_size, 1))], dim=1), 'b nv -> (b nv)', nv=actual_num_frames
                )

                camera_pose_ = gt_pose.view(batch_size, 4, 16)
                padding = [0] * (len(camera_pose_.shape) * 2)
                padding[-3] = 1
                camera = F.pad(camera_pose_, tuple(padding)).to(dtype=gt_latents.dtype, device=gt_latents.device)
                camera = einops.rearrange(camera, "b nv c -> (b nv) c")

                unet_inputs = {
                    'x': noisy_latents_with_ip_latent,
                    'timesteps': timestep_with_ip,
                    'context': torch.cat([text_prompt_embeds] * actual_num_frames),
                    'num_frames': actual_num_frames,
                    'camera': camera,
                    'ip': torch.cat([image_clip_embedding] * actual_num_frames),
                    'ip_img': image_latents,
                }

                unet_noise_pred = unet.forward(**unet_inputs)

                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                
                unet_noise_pred = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
                unet_orthogonal_noise_pred = einops.rearrange(unet_noise_pred[:, :-1, :, :, :], "b nv c h w -> (b nv) c h w")
                
                loss = F.mse_loss(unet_orthogonal_noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    vae_unwrapped = accelerator.unwrap_model(vae)
                    params_to_clip = list(vae_unwrapped.encoder.conv_in.parameters()) + [p for p in unet.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                
                if args.use_ema:
                    ema_unet.step([p for p in unet.parameters() if p.requires_grad])
                    
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 1:
                        # --- 1. ROTAZIONE DEI FILE VECCHI ---
                        if args.checkpoints_total_limit is not None:
                            # Cancella vecchi checkpoint (stati di accelerate)
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                                    
                            # Cancella vecchie pipeline intermedie
                            pipelines = os.listdir(args.output_dir)
                            pipelines = [d for d in pipelines if d.startswith("pipeline")]
                            pipelines = [d for d in pipelines if len(d.split("-")) > 1 and d.split("-")[1].isdigit()]
                            pipelines = sorted(pipelines, key=lambda x: int(x.split("-")[1]))
                            
                            if len(pipelines) >= args.checkpoints_total_limit:
                                num_to_remove = len(pipelines) - args.checkpoints_total_limit + 1
                                removing_pipelines = pipelines[0:num_to_remove]
                                for removing_pipeline in removing_pipelines:
                                    removing_pipeline = os.path.join(args.output_dir, removing_pipeline)
                                    shutil.rmtree(removing_pipeline)

                        # --- 2. SALVATAGGIO STATO ACCELERATE (PER RIPRENDERE IL TRAINING) ---
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"-> Checkpoint salvato in: {save_path} | Dimensione: {get_path_size_str(save_path)}")

                        # --- 3. SALVATAGGIO PIPELINE COME A FINE TRAINING (PER TEST INFERENZA) ---
                        unet_unwrapped_ckpt = accelerator.unwrap_model(unet)
                        
                        if args.use_ema:
                            trainable_params_save = [p for p in unet_unwrapped_ckpt.parameters() if p.requires_grad]
                            ema_unet.store(trainable_params_save)
                            ema_unet.copy_to(trainable_params_save)
                            
                        pipeline_ckpt = ImageDreamPipeline(
                            vae=accelerator.unwrap_model(vae),
                            unet=unet_unwrapped_ckpt,
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            tokenizer=tokenizer,
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            scheduler=noise_scheduler
                        )
                        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
                        pipeline_ckpt.save_pretrained(pipeline_save_path)
                        print(f"-> Pipeline salvata in: {pipeline_save_path} | Dimensione: {get_path_size_str(pipeline_save_path)}")

                        # RIPRISTINO PESI ORIGINALI DOPO IL SALVATAGGIO INTERMEDIO
                        if args.use_ema:
                            ema_unet.restore([p for p in unet_unwrapped_ckpt.parameters() if p.requires_grad])

                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    logger.info(f"[INFO] {step}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}")

            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'],
                    "loss_epoch": loss_epoch / num_train_elems, "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # =========================================================================
        # CICLO DI VALIDAZIONE DETERMINISTICO (CON SWAP EMA TEMPORANEO)
        # =========================================================================
        if args.use_ema:
            trainable_params_val = [p for p in unet.parameters() if p.requires_grad]
            ema_unet.store(trainable_params_val)
            ema_unet.copy_to(trainable_params_val)

        val_loss = compute_val_loss(
            unet, vae, vae_gt, image_encoder, text_encoder, tokenizer,
            noise_scheduler, val_dataloader, accelerator, weight_dtype, args
        )
        
        if args.use_ema:
            ema_unet.restore([p for p in unet.parameters() if p.requires_grad])

        train_loss_epoch_avg = loss_epoch / max(num_train_elems, 1)
        logger.info(f"--- [EPOCH {epoch}] Train Loss Epoch: {train_loss_epoch_avg:.6f} | Val Loss: {val_loss:.6f} ---")
        accelerator.log({"val_loss": val_loss, "epoch": epoch}, step=global_step)

    # =========================================================================
    # SALVATAGGIO PIPELINE E UPLOAD PIPELINE A FINE TRAINING
    # =========================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        
        # SWAP FINALE EMA PRIMA DI SALVARE
        if args.use_ema:
            trainable_params_save = [p for p in unet_unwrapped.parameters() if p.requires_grad]
            ema_unet.store(trainable_params_save)
            ema_unet.copy_to(trainable_params_save)
            
        pipeline = ImageDreamPipeline(
            vae=accelerator.unwrap_model(vae),
            unet=unet_unwrapped,
            image_encoder=accelerator.unwrap_model(image_encoder),
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(text_encoder),
            scheduler=noise_scheduler
        )
        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
        pipeline.save_pretrained(pipeline_save_path)
        print(f"-> File/Cartella salvato in: {pipeline_save_path} | Dimensione: {get_path_size_str(pipeline_save_path)}")

        # RIPRISTINO PESI ORIGINALI DOPO IL SALVATAGGIO
        if args.use_ema:
            ema_unet.restore([p for p in unet_unwrapped.parameters() if p.requires_grad])

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=pipeline_save_path,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
