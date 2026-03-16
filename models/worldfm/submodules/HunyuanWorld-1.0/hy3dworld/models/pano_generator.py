# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].

import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL

from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from diffusers.utils.torch_utils import randn_tensor

from .pipelines import FluxPipeline, FluxFillPipeline

class Text2PanoramaPipelines(FluxPipeline):
    @torch.no_grad()
    def __call__(self, prompt, **kwargs):
        """Main inpainting call."""
        return self._call_shared(prompt=prompt, is_inpainting=False, early_steps=3, **kwargs)


class Image2PanoramaPipelines(FluxFillPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        # Initilization from FluxFillPipeline
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )

        # change some part of initilization
        self.latent_channels = self.vae.config.latent_channels if getattr(
            self, "vae", None) else 16
        
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps *
                            strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_inpainting_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        is_strength_max=True,
        timestep=None,
    ):
        r"""
        Prepares the latents for the Image2PanoramaPipelines.
        """
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        # Return the latents if they are already provided
        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        # If no latents are provided, we need to encode the image
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(
                image=image, generator=generator)
        else:
            image_latents = image
        
        # Ensure image_latents has the correct shape
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat(
                [image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
        # Add noise to the latents
        noise = randn_tensor(shape, generator=generator,
                             device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        
        # prepare blended latents
        latents = torch.cat(
            [latents, latents[:, :, :, :self.blend_extend]], dim=-1)
        width_new_blended = latents.shape[-1]
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width_new_blended)
        # prepare latent image ids
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width_new_blended // 2, device, dtype)

        return latents, latent_image_ids, width_new_blended

    def prepare_blending_latent(
        self, latents, height, width, batch_size, num_channels_latents, width_new_blended=None
    ):
        return latents, width_new_blended
    
    def _apply_blending(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        num_channels_latents: int,
        batch_size: int,
        **karwgs,
    ) -> torch.Tensor:
        r"""Apply horizontal blending to latents."""
        # Unpack latents for processing
        latents_unpack = self._unpack_latents(
            latents, height, width_new_blended*self.vae_scale_factor, self.vae_scale_factor
        )
        # Apply blending
        latents_unpack = self.blend_h(latents_unpack, latents_unpack, self.blend_extend)
        
        latent_height = 2 * \
            (int(height) // (self.vae_scale_factor * 2))

        shifting_extend = karwgs.get("shifting_extend", None)
        if shifting_extend is None:
            shifting_extend = latents_unpack.size()[-1]//4
        
        latents_unpack = torch.roll(
            latents_unpack, shifting_extend, -1)
        
        # Repack latents after blending
        latents = self._pack_latents(
            latents_unpack, batch_size, num_channels_latents, latent_height, width_new_blended)
        return latents

    def _apply_blending_mask(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        num_channels_latents: int,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        r"""Apply horizontal blending to mask latents."""
        return self._apply_blending(
            latents, height, width_new_blended, 80, batch_size, **kwargs
        )

    def _final_process_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        width: int
    ) -> torch.Tensor:
        """Final processing of latents before decoding."""
        # Unpack and crop to target width
        latents_unpack = self._unpack_latents(
            latents, height, width_new_blended * self.vae_scale_factor, self.vae_scale_factor
        )
        latents_unpack = self.blend_h(
            latents_unpack, latents_unpack, self.blend_extend
        )
        latents_unpack = latents_unpack[:, :, :, :width // self.vae_scale_factor]
        
        # Repack for final output
        return self._pack_latents(
            latents_unpack,
            latents.shape[0],  # batch size
            latents.shape[2] // 4,  # num_channels_latents
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )

    @torch.no_grad()
    def __call__(self, **kwargs):
        """Main inpainting call."""
        return self._call_shared(is_inpainting=True, early_steps=3, blend_extra_chanel=True, **kwargs)
