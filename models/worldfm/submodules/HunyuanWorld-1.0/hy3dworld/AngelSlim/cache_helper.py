import types
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
)

class DeepCacheHelper(object):
    """A helper class for optimizing model inference through selective feature caching.
    
    This class implements a caching mechanism that can skip recomputation of certain
    layers/blocks during model inference, significantly improving performance for
    sequential operations like diffusion models.
    """
    def __init__(
        self,
        pipe_model: Optional[object] = None,
        timesteps: Optional[int] = None,
        no_cache_steps: Optional[list[int]] = None,
        no_cache_block_id: Optional[list[int]] = None,
        no_cache_layer_id: Optional[list[int]] = None
        ):
        """Initialize the DeepCache helper with model and caching configuration.
        
        Args:
            pipe_model (object, optional): The target model to be optimized.
            timesteps (int, optional): Total timesteps for sequential processing.
            no_cache_steps (list/int, optional): Steps where caching should be disabled.
            no_cache_block_id (list/int, optional): Block IDs to exclude from caching.
            no_cache_layer_id (list/int, optional): Layer IDs to exclude from caching.
        """
        if pipe_model is not None: self.pipe_model = pipe_model
        if timesteps is not None: self.timesteps = timesteps
        if no_cache_steps is not None: self.no_cache_steps = no_cache_steps
        if no_cache_block_id is not None: self.no_cache_block_id = no_cache_block_id
        if no_cache_layer_id is not None: self.no_cache_layer_id = no_cache_layer_id
        self.set_default_blocktypes()
        self.set_model_type()

    def set_default_blocktypes(self, default_blocktypes = None):
        """Configure default block types for caching.
        
        Args:
            default_blocktypes (list, optional): List of block types to cache.
                                                 Defaults to ['single'] if None.
        """
        self.default_blocktypes = ['single']

    def set_model_type(self, model_type = "flux"):
        """Set the model architecture type.
        
        Args:
            model_type (str): The type of model architecture ('flux' by default).
        """
        self.model_type = model_type
        
    def enable(self):
        """Activate the caching mechanism for the model."""
        assert self.pipe_model is not None
        self.reset_states()
        self.wrap_modules()

    def disable(self):
        """Deactivate the caching mechanism and restore original model functions."""
        self.unwrap_modules()
        self.reset_states()


    def is_skip_step(self, block_i, layer_i, blocktype):
        """Determine if current step should skip caching.
        
        Args:
            block_i (int): Current block index
            layer_i (int): Current layer index
            blocktype (str): Type of the current block
            
        Returns:
            bool: True if caching should be skipped, False otherwise
        """
        # For some pipeline that the first timestep != 0
        self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep

        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False
        if blocktype in self.default_blocktypes:
            if block_i in self.no_cache_block_id[blocktype]:
                return False
            else:
                return True
        return True

    def wrap_model_forward(self):
        """Wrap the model's forward function to enable caching control."""
        self.function_dict['model_forward'] = self.pipe_model.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapper function that maintains the original forward signature."""
            result = self.function_dict['model_forward'](*args, **kwargs)
            return result

        self.pipe_model.forward = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype):
        """Wrap a specific block's forward function with caching logic.
        
        Args:
            block (nn.Module): The block module to wrap
            block_name (str): Name identifier for the block
            block_i (int): Block index
            layer_i (int): Layer index
            blocktype (str): Type of the block
        """
        # Store original forward function
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Cached version of block forward function."""
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            # Use cached result if available and skipping is enabled
            result = self.cached_output[(blocktype, block_name, block_i, layer_i)] if skip else self.function_dict[
                (blocktype, block_name, block_i, layer_i)](*args, **kwargs)
            # Update cache if not skipping
            if not skip: self.cached_output[(blocktype, block_name, block_i, layer_i)] = result
            return result

        block.forward = wrapped_forward

    def wrap_modules(self):
        """Wrap all relevant modules in the model with caching functionality."""
        # 1. wrap flux forward
        self.wrap_model_forward()
        # 2. wrap double forward
        for block_i, block in enumerate(self.pipe_model.transformer_blocks):
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="double")

        # 3. wrap single forward
        block_num = len(self.pipe_model.single_transformer_blocks)
        for block_i, block in enumerate(self.pipe_model.single_transformer_blocks):
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="single")

    def unwrap_modules(self):
        """Restore original forward functions for all wrapped modules."""
        # 1. model forward
        self.pipe_model.forward = self.function_dict['model_forward']
        # 2. block forward
        for block_i, block in enumerate(self.pipe_model.transformer_blocks):
            block.forward = self.function_dict[("double", "block", block_i, 0)]

        # 3. single block forward
        block_num = len(self.pipe_model.single_transformer_blocks)
        for block_i, block in enumerate(self.pipe_model.single_transformer_blocks):
            block.forward = self.function_dict[("single", "block", block_num - block_i - 1, 0)]

    def reset_states(self):
        """Reset all caching-related states and clear cached outputs."""
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None


def flux_deepcache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    is_cache: bool = False,
    is_neg=False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape
            `(batch size, channel, height, width)`): Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape
            `(batch size, sequence_len, embed_dims)`): Conditional embeddings
            (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`)
            Embeddings projected from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`): A list of tensors
            that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, optional): A kwargs dictionary that if specified
            is passed along to the AttentionProcessor as defined under self.processor in
            https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py. # noqa: E501
        return_dict (`bool`, optional, defaults to `True`): Whether or not to return a
            [~models.transformer_2d.Transformer2DModelOutput] instead of a plain tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`]
        is returned, otherwise a `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    if not is_cache:
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                #hidden_states.shape torch.Size([1, 7380, 3072]) ,encoder_hidden_states.shape [1, 512, 3072]
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if is_cache:
            if index_block < len(self.single_transformer_blocks) - 1:
                continue
            if is_neg:
                hidden_states = self.neg_cache_feature
            else:
                hidden_states = self.cache_feature
        else:
            if index_block == len(self.single_transformer_blocks) - 1:
                if is_neg:
                    self.neg_cache_feature = hidden_states
                else:
                    self.cache_feature = hidden_states

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)




