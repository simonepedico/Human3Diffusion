#### Author: Jiaxiang Tang
#### Org. Link: https://github.com/3DTopia/LGM/blob/main/mvdream/pipeline_mvdream.py

import torch
import torch.nn.functional as F
import inspect
import numpy as np
from typing import Callable, List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from mvdream.mv_unet import MultiViewUNetModel, get_camera

import PIL
import kornia
import einops

logger = logging.get_logger(__name__)


class ImageDreamPipeline(DiffusionPipeline):

    # PEDICO: aggiunto channel_adapter 
    _optional_components = ["feature_extractor", "image_encoder", "channel_adapter"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: MultiViewUNetModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDIMScheduler,
        image_encoder: CLIPVisionModel,
        channel_adapter: Optional[torch.nn.Module] = None, # Accetta l'adapter addestrato
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "  # type: ignore
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            channel_adapter=channel_adapter,
        )
        
        
        if self.channel_adapter is None:
            self.channel_adapter = torch.nn.Conv2d(7, 3, kernel_size=3, padding=1)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.channel_adapter]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.channel_adapter, self.unet, self.vae]:
            if cpu_offloaded_model is not None:
                _, hook = cpu_offload_with_hook(
                    cpu_offloaded_model, device, prev_module_hook=hook
                )

        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance: bool,
        negative_prompt=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` should be either a string or a list of strings, but got {type(prompt)}."
            )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy() 
        return image

    def check_inputs(self, image, height, width, callback_steps):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def CLIP_preprocess(self, x):
        dtype = x.dtype
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)
        x = (x + 1.) / 2. 
        x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                     torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return x
    
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)
            
            assert image.max() > 1.0, "Image should be in [0, 255] range"
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0 
        
        image = self.CLIP_preprocess(image)
        image = image.to(device=device, dtype=dtype)
        
        image_embeds = self.image_encoder(image, output_hidden_states=True).hidden_states[-2] 
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.zeros_like(image_embeds), image_embeds

    def encode_image_latents(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)
            
            assert image.max() > 1.0, "Image should be in [0, 255] range"
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0 
            image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

        image = image.to(device=device, dtype=dtype)

        posterior = self.vae.encode(image).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor 
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0) 

        return torch.zeros_like(latents), latents

    @torch.no_grad() 
    def __call__(
        self,
        prompt: str = "",
        image: Union[np.ndarray, torch.FloatTensor] = None, 
        depth_map: Union[np.ndarray, torch.FloatTensor] = None,  
        normal_map: Union[np.ndarray, torch.FloatTensor] = None, 
        height: int = 256,
        width: int = 256,
        elevation: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "numpy", 
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        num_frames: int = 4,
        camera_pose=None,
    ):  
        device = self._execution_device

        self.unet = self.unet.to(device=device)
        self.vae = self.vae.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)
        if self.channel_adapter is not None:
            self.channel_adapter = self.channel_adapter.to(device=device) 

        def to_tensor(x, expected_channels):
            if isinstance(x, np.ndarray):
                if x.ndim == 3 and expected_channels == 1: 
                    x = x[:, :, 0:1]
                if x.ndim == 2:
                    x = x[:, :, None]
                x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
                if x.max() > 1.0: x = x / 127.5 - 1.0
            elif isinstance(x, torch.Tensor):
                if x.ndim == 3: x = x.unsqueeze(0)
            return x.to(device)

        if image is not None and depth_map is not None and normal_map is not None:
            img_t = to_tensor(image, 3)
            depth_t = to_tensor(depth_map, 1)
            normal_t = to_tensor(normal_map, 3)

            if depth_t.shape[-2:] != img_t.shape[-2:]:
                depth_t = F.interpolate(depth_t, size=img_t.shape[-2:], mode='bilinear', align_corners=False)
            if normal_t.shape[-2:] != img_t.shape[-2:]:
                normal_t = F.interpolate(normal_t, size=img_t.shape[-2:], mode='bilinear', align_corners=False)

            input_7ch = torch.cat([img_t, depth_t, normal_t], dim=1)
            
            if self.channel_adapter is not None:
                # Allineamento dinamico del dtype (fp16 / fp32) basato sui parametri correnti del layer
                adapter_dtype = next(self.channel_adapter.parameters()).dtype
                input_7ch = input_7ch.to(dtype=adapter_dtype)
                
                image = self.channel_adapter(input_7ch)
                image = torch.clamp(image, -1.0, 1.0)

        self.check_inputs(image, height, width, callback_steps)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device) 
        timesteps = self.scheduler.timesteps                             

        if image is not None:
            self.image_encoder = self.image_encoder.to(device=device)
            image_embeds_neg, image_embeds_pos = self.encode_image(image, device, num_images_per_prompt)
            image_latents_neg, image_latents_pos = self.encode_image_latents(image, device, num_images_per_prompt) 
        else:
            assert False, "Image is required for the model"

        _prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt * batch_size, 
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )  
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)

        actual_num_frames = num_frames if image is None else num_frames + 1
        latents: torch.Tensor = self.prepare_latents(
            actual_num_frames * num_images_per_prompt * batch_size,
            4,
            height,
            width,
            prompt_embeds_pos.dtype,
            device,
            generator,
            None,
        )

        if image is not None:
            if camera_pose is None and elevation is None:
                assert False, "Camera pose or elevation is required for the model"
            if camera_pose is None:
                camera = get_camera(num_frames, elevation=elevation, extra_view=True).to(dtype=latents.dtype, device=device)
            else:
                camera_pose_ = camera_pose.view(batch_size, 4, 16) 
                padding = [0] * (len(camera_pose_.shape) * 2)  
                padding[-3] = 1
                padding_tuple = tuple(padding)
                camera = F.pad(camera_pose_, padding_tuple).to(dtype=latents.dtype, device=device) 
        else:
            assert False, "Image is required for the model"
        camera = camera.repeat_interleave(num_images_per_prompt, dim=0) 
        camera = einops.rearrange(camera, 'b nv c -> (b nv) c')
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                multiplier = 2 if do_classifier_free_guidance else 1
                latent_model_input = torch.cat([latents] * multiplier)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                unet_inputs = {
                    'x': latent_model_input, 
                    'timesteps': torch.tensor([t] * actual_num_frames * batch_size * multiplier, dtype=latent_model_input.dtype, device=device),
                    'context': torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames),
                    'num_frames': actual_num_frames,
                    'camera': torch.cat([camera] * multiplier),
                }

                if image is not None:
                    unet_inputs['ip'] = torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
                    unet_inputs['ip_img'] = torch.cat([image_latents_neg] + [image_latents_pos]) 
                else:
                    assert False, "Image is required for the model"
                
                noise_pred = self.unet.forward(**unet_inputs)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents: torch.Tensor = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents) 
        
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else: 
            image = self.decode_latents(latents)

        image = einops.rearrange(image, '(b nv) h w c -> b nv h w c', b=batch_size)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image
