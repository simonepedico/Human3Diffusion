import argparse
import json
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
import time

from huggingface_hub.constants import HF_HUB_CACHE
from sklearn.model_selection import GroupShuffleSplit

HF_TOKEN = os.environ.get("HF_TOKEN")

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
from core.conditioning_utils import (clip_preprocess, diffusion_target, predict_clean, foreground_loss, ConditioningEncoder)
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

CLIP_preprocess = clip_preprocess

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Argparser for ImageDream fine-tuning script.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.10) # MODIFICATO: da 0.15 a 0.10
    parser.add_argument("--checkpointing_steps", type=int, default=100) # Mantenuto per non rompere script esterni, ma non usato
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=0.2, type=float) # MODIFICATO: da 0.5 a 0.2
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
    parser.add_argument("--num_train_epochs", type=int, default=30) # MODIFICATO: da 100 a 30
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    parser.add_argument('--subject_ids_json', type=str, default=None, help='JSON: mesh folder name -> real subject ID; all poses and outfits of one person share an ID.')
    parser.add_argument('--foreground_loss_weight', type=float, default=1.0)
    parser.add_argument('--rgb_loss_weight', type=float, default=0.1)
    parser.add_argument('--rgb_loss_max_t', type=int, default=500)
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

            image_latents = vae(fused_input)

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
                    'context': text_prompt_embeds.repeat_interleave(actual_num_frames, dim=0),
                    'num_frames': actual_num_frames,
                    'camera': camera,
                    'ip': image_clip_embedding.repeat_interleave(actual_num_frames, dim=0),
                    'ip_img': image_latents,
                }

                unet_noise_pred = unet(**unet_inputs)

                target = diffusion_target(noise_scheduler, gt_latents, noise, timesteps)

                unet_noise_pred = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
                unet_orthogonal_noise_pred = einops.rearrange(unet_noise_pred[:, :-1, :, :, :], "b nv c h w -> (b nv) c h w")

                loss_t = F.mse_loss(unet_orthogonal_noise_pred.float(), target.float(), reduction="mean")
                batch_step_loss += loss_t.item()

            val_loss_total += (batch_step_loss / len(eval_timesteps)) * batch_size
            val_steps += batch_size

    stats = accelerator.reduce(torch.tensor([val_loss_total, val_steps], device=accelerator.device, dtype=torch.float64), reduction='sum')
    if stats[1] == 0:
        raise ValueError('Empty validation set')
    return (stats[0] / stats[1]).item()

def main(args):
    args.max_train_steps = None
    args.learning_rate = 5e-5 # MODIFICATO: da 1e-4 a 5e-5
    args.tracker_project_name = "train_mvd_direct_7ch"
    args.num_gpu = 1
    args.gradient_accumulation_steps = 4 # MODIFICATO: da 2 a 4

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
    loaded_keys = set()
    for k, v in ckpt_mvd_2k2k.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded_keys.add(k)

    missing = set(state_dict) - loaded_keys
    if missing:
        raise ValueError(f'Incomplete Human3Diffusion UNet checkpoint: {len(missing)} missing/mismatched keys: {sorted(missing)[:5]}')

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
        nn.init.zeros_(new_conv.weight[:, 3:, :, :])  # Initially preserve pretrained RGB behavior.
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    vae.encoder.conv_in = new_conv
    vae.register_to_config(in_channels=7)

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
        ema_vae = EMAModel(vae_trainable_params, decay=0.9999)

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

    # local_dataset_root = "/scratch_share/3dig/simone_pedico/dataset_hf/dataset_hf"
    local_dataset_root = "/scratch_local/s.pedico-thesis/dataset_hf"
    
    logger.info("-> Inizio scansione del dataset locale. Attendere...")
    start_time = time.time()

    quadruple_locali = []
    cartelle_processate = 0
    
    # Scansione della cartella locale per trovare tutti i render
    for root_dir, dirs, files in os.walk(local_dataset_root):
        if root_dir != local_dataset_root:
            cartelle_processate += 1
            if cartelle_processate % 5 == 0:
                logger.info(f"   ... scansionate {cartelle_processate} cartelle ...")

        for file in files:
            if file.startswith("rgb_") and file.endswith(".png"):
                id_campione = file.replace("rgb_", "").replace(".png", "")
                quadruple_locali.append({
                    "id": id_campione,
                    "rgb": os.path.join(root_dir, file),
                    "matrix": os.path.join(root_dir, f"{id_campione}_RT.txt"),
                    "depth": os.path.join(root_dir, f"depth_{id_campione}.png"),
                    "normal": os.path.join(root_dir, f"normal_{id_campione}.png"),
                    "cartella_origine": root_dir
                })

    end_time = time.time()
    logger.info(f"-> Scansione completata in {end_time - start_time:.2f} secondi. Trovate {len(quadruple_locali)} viste in totale.")

    subject_map = None
    if args.subject_ids_json:
        with open(args.subject_ids_json) as stream:
            subject_map = json.load(stream)
        for q in quadruple_locali:
            name = os.path.basename(q['cartella_origine'])
            if not isinstance(subject_map.get(name), str) or not subject_map[name].strip():
                raise ValueError(f'Missing subject ID for {name}')
    else:
        logger.warning('Filename-derived split is not verified identity-disjoint. Supply --subject_ids_json for generalization evaluation.')

    def subject_id(q, sizer=False):
        name = os.path.basename(q['cartella_origine'])
        return subject_map[name] if subject_map is not None else (name if sizer else name.split('_')[0])

    # --- LOGICA DI SPLITTING ---
    quadruple_normali = []
    quadruple_sizer = []
    
    # Divisione basata sul prefisso della cartella
    for q in quadruple_locali:
        folder_name = os.path.basename(q["cartella_origine"])
        if folder_name.startswith("sizer"):
            quadruple_sizer.append(q)
        else:
            quadruple_normali.append(q)

    train_paths = []
    val_paths = []
    test_paths = []

    # 1. Splitting Mesh Normali (Invariato: ~80% Train, ~10% Val, ~10% Test)
    if len(quadruple_normali) > 0:
        subject_ids_normali = [subject_id(q) for q in quadruple_normali]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx_norm, resto_idx_norm = next(gss1.split(quadruple_normali, groups=subject_ids_normali))
        
        train_paths.extend([quadruple_normali[i] for i in train_idx_norm])
        resto_paths_norm = [quadruple_normali[i] for i in resto_idx_norm]
        
        resto_subject_ids = [subject_id(q) for q in resto_paths_norm]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx_norm, test_idx_norm = next(gss2.split(resto_paths_norm, groups=resto_subject_ids))
        
        val_paths.extend([resto_paths_norm[i] for i in val_idx_norm])
        test_paths.extend([resto_paths_norm[i] for i in test_idx_norm])

    # 2. Splitting Mesh Sizer (Train 80%, Validation 20%, niente Test)
    if len(quadruple_sizer) > 0:
        subject_ids_sizer = [subject_id(q, sizer=True) for q in quadruple_sizer]
        gss_sizer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx_sizer, val_idx_sizer = next(gss_sizer.split(quadruple_sizer, groups=subject_ids_sizer))
        
        train_paths.extend([quadruple_sizer[i] for i in train_idx_sizer])
        val_paths.extend([quadruple_sizer[i] for i in val_idx_sizer])

    logger.info(f"-> Distribuzione dataset:\n   - Train: {len(train_paths)} | - Val: {len(val_paths)} | - Test: {len(test_paths)}")

    if subject_map is not None:
        groups = [{subject_id(q) for q in split} for split in (train_paths, val_paths, test_paths)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError('Subject leakage between splits, including across source datasets.')
    if not train_paths or not val_paths:
        raise ValueError('Training and validation must both be nonempty')
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'split_manifest.json'), 'w') as stream:
            json.dump({label: sorted({q['cartella_origine'] for q in split}) for label, split in
                       [('train', train_paths), ('val', val_paths), ('test', test_paths)]}, stream, indent=2)

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

    vae = ConditioningEncoder(vae)
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
        ema_vae.to(accelerator.device)
        accelerator.register_for_checkpointing(ema_unet, ema_vae)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    best_val_loss = float("inf")
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        state_path = os.path.join(args.resume_from_checkpoint, 'training_progress.json')
        if not os.path.isfile(state_path):
            raise ValueError('Only corrected checkpoints with training_progress.json can be resumed.')
        with open(state_path) as stream:
            progress = json.load(stream)
        accelerator.load_state(args.resume_from_checkpoint)
        first_epoch = int(progress['next_epoch'])
        global_step = int(progress['global_step'])
        best_val_loss = float(progress['best_val_loss'])

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
            with accelerator.accumulate(unet, vae):
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
                if not 0 <= p_drop <= 1:
                    raise ValueError('conditioning_dropout_prob must be in [0, 1]')
                drop_all = (torch.rand(batch_size, device=gt_latents.device) < p_drop)
                image_clip = CLIP_preprocess(input_image).to(dtype=weight_dtype)
                with torch.no_grad():
                    image_clip_embedding = image_encoder(image_clip, output_hidden_states=True).hidden_states[-2]
                    text_prompt_embeds = _encode_text_prompt(tokenizer, text_encoder, text_prompt, gt_latents.device, batch_size)
                    null_text = _encode_text_prompt(tokenizer, text_encoder, '', gt_latents.device, batch_size)
                image_latents = vae(fused_input)
                image_latents = torch.where(drop_all[:, None, None, None], torch.zeros_like(image_latents), image_latents)
                image_clip_embedding = torch.where(drop_all[:, None, None], torch.zeros_like(image_clip_embedding), image_clip_embedding)
                text_prompt_embeds = torch.where(drop_all[:, None, None], null_text, text_prompt_embeds)

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
                    'context': text_prompt_embeds.repeat_interleave(actual_num_frames, dim=0),
                    'num_frames': actual_num_frames,
                    'camera': camera,
                    'ip': image_clip_embedding.repeat_interleave(actual_num_frames, dim=0),
                    'ip_img': image_latents,
                }

                unet_noise_pred = unet(**unet_inputs)

                target = diffusion_target(noise_scheduler, gt_latents, noise, timesteps)
                
                unet_noise_pred = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
                unet_orthogonal_noise_pred = einops.rearrange(unet_noise_pred[:, :-1, :, :, :], "b nv c h w -> (b nv) c h w")
                
                loss = F.mse_loss(unet_orthogonal_noise_pred.float(), target.float(), reduction="mean")

                loss_diffusion = loss
                target_masks = einops.rearrange(batch['imagedream_masks_gt'], 'b n c h w -> (b n) c h w')
                loss_foreground = foreground_loss(unet_orthogonal_noise_pred, target, target_masks)
                loss_rgb = loss_diffusion * 0
                # Supervise decoded x0 only at moderate/low noise and with real conditioning.
                supervised = (timesteps <= args.rgb_loss_max_t) & ~drop_all.repeat_interleave(num_view)
                if args.rgb_loss_weight > 0 and supervised.any():
                    predicted_clean = predict_clean(noise_scheduler, noisy_latents, unet_orthogonal_noise_pred, timesteps)
                    decoded = vae_gt.decode((predicted_clean[supervised] / vae_gt.config.scaling_factor).to(weight_dtype)).sample
                    # Frozen decoder parameters still propagate gradients to the denoiser/VAE input.
                    loss_rgb = foreground_loss(decoded, gt_image[supervised], target_masks[supervised])
                loss = loss_diffusion + args.foreground_loss_weight * loss_foreground + args.rgb_loss_weight * loss_rgb
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    vae_unwrapped = accelerator.unwrap_model(vae)
                    params_to_clip = list(vae_unwrapped.vae.encoder.conv_in.parameters()) + [p for p in unet.parameters() if p.requires_grad]
                    conv_grad = vae_unwrapped.vae.encoder.conv_in.weight.grad
                    if conv_grad is None:
                        raise RuntimeError('Conditioning VAE is disconnected from the loss')
                    accelerator.log({'grad_vae_rgb': conv_grad[:, :3].float().norm().item(),
                                     'grad_vae_maps': conv_grad[:, 3:].float().norm().item()}, step=global_step)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                
                if args.use_ema:
                    ema_unet.step([p for p in unet.parameters() if p.requires_grad])
                    ema_vae.step(vae_trainable_params)
                    
                progress_bar.update(1)
                global_step += 1

                mem_free, mem_total = torch.cuda.mem_get_info()    
                logger.info(f"[INFO] {step}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}")

            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {"loss_diffusion": loss_diffusion.detach().item(), "loss_foreground": loss_foreground.detach().item(), "loss_rgb": loss_rgb.detach().item(), "loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'],
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
            ema_vae.store(vae_trainable_params)
            ema_unet.copy_to(trainable_params_val)
            ema_vae.copy_to(vae_trainable_params)

        val_loss = compute_val_loss(
            unet, vae, vae_gt, image_encoder, text_encoder, tokenizer,
            noise_scheduler, val_dataloader, accelerator, weight_dtype, args
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if accelerator.is_main_process:
                best_pipe = ImageDreamPipeline(
                    vae=accelerator.unwrap_model(vae).vae, unet=accelerator.unwrap_model(unet),
                    image_encoder=image_encoder, tokenizer=tokenizer, text_encoder=text_encoder,
                    scheduler=noise_scheduler)
                best_path = os.path.join(args.output_dir, 'pipeline-best')
                best_pipe.save_pretrained(best_path)
                vae_gt.save_pretrained(os.path.join(best_path, 'vae_rgb'))
                del best_pipe

        if args.use_ema:
            ema_unet.restore([p for p in unet.parameters() if p.requires_grad])
            ema_vae.restore(vae_trainable_params)

        train_loss_epoch_avg = loss_epoch / max(num_train_elems, 1)
        logger.info(f"--- [EPOCH {epoch}] Train Loss Epoch: {train_loss_epoch_avg:.6f} | Val Loss: {val_loss:.6f} ---")
        accelerator.log({"val_loss": val_loss, "epoch": epoch}, step=global_step)

        # =========================================================================
        # SALVATAGGIO PIPELINE E CHECKPOINT OGNI 5 EPOCHE
        # =========================================================================
        if (epoch + 1) % 5 == 0:
            accelerator.wait_for_everyone()
            save_path = os.path.join(args.output_dir, f'checkpoint-{epoch+1}')
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                with open(os.path.join(save_path, 'training_progress.json'), 'w') as stream:
                    json.dump({'next_epoch': epoch+1, 'global_step': global_step, 'best_val_loss': best_val_loss}, stream)
        if accelerator.is_main_process:
            if (epoch + 1) % 5 == 0:
                # --- 1. ROTAZIONE DEI FILE VECCHI ---
                if args.checkpoints_total_limit is not None:
                    # Cancella vecchi checkpoint (stati di accelerate)
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) > args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit
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
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
                print(f"-> Checkpoint salvato in: {save_path} | Dimensione: {get_path_size_str(save_path)}")

                # --- 3. SALVATAGGIO PIPELINE COME A FINE TRAINING (PER TEST INFERENZA) ---
                unet_unwrapped_ckpt = accelerator.unwrap_model(unet)
                
                if args.use_ema:
                    trainable_params_save = [p for p in unet_unwrapped_ckpt.parameters() if p.requires_grad]
                    ema_unet.store(trainable_params_save)
                    ema_vae.store(vae_trainable_params)
                    ema_unet.copy_to(trainable_params_save)
                    ema_vae.copy_to(vae_trainable_params)
                    
                pipeline_ckpt = ImageDreamPipeline(
                    vae=accelerator.unwrap_model(vae).vae,
                    unet=unet_unwrapped_ckpt,
                    image_encoder=accelerator.unwrap_model(image_encoder),
                    tokenizer=tokenizer,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    scheduler=noise_scheduler
                )
                pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{epoch+1}")
                pipeline_ckpt.save_pretrained(pipeline_save_path)
                vae_gt.save_pretrained(os.path.join(pipeline_save_path, 'vae_rgb'))
                print(f"-> Pipeline salvata in: {pipeline_save_path} | Dimensione: {get_path_size_str(pipeline_save_path)}")

                # RIPRISTINO PESI ORIGINALI DOPO IL SALVATAGGIO INTERMEDIO
                if args.use_ema:
                    ema_unet.restore([p for p in unet_unwrapped_ckpt.parameters() if p.requires_grad])
                    ema_vae.restore(vae_trainable_params)

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
            ema_vae.store(vae_trainable_params)
            ema_unet.copy_to(trainable_params_save)
            ema_vae.copy_to(vae_trainable_params)
            
        pipeline = ImageDreamPipeline(
            vae=accelerator.unwrap_model(vae).vae,
            unet=unet_unwrapped,
            image_encoder=accelerator.unwrap_model(image_encoder),
            tokenizer=tokenizer,
            text_encoder=accelerator.unwrap_model(text_encoder),
            scheduler=noise_scheduler
        )
        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-final")
        pipeline.save_pretrained(pipeline_save_path)
        vae_gt.save_pretrained(os.path.join(pipeline_save_path, 'vae_rgb'))
        print(f"-> File/Cartella salvato in: {pipeline_save_path} | Dimensione: {get_path_size_str(pipeline_save_path)}")

        # RIPRISTINO PESI ORIGINALI DOPO IL SALVATAGGIO
        if args.use_ema:
            ema_unet.restore([p for p in unet_unwrapped.parameters() if p.requires_grad])
            ema_vae.restore(vae_trainable_params)

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
