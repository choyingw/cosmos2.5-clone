# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import field
from typing import Any, Callable, Dict, Optional, Tuple, cast

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed import get_process_group_ranks

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.context_parallel import broadcast, broadcast_split_tensor
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import IS_PREPROCESSED_KEY
from cosmos_transfer2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_transfer2._src.predict2.utils.dtensor_helper import broadcast_dtensor_model_states
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocationList,
    MultiViewCondition,
)
from cosmos_transfer2._src.predict2_multiview.models.view_sampling import sample_n_views_from_data_batch

TRAIN_SAMPLE_N_VIEWS_KEY = "train_sample_n_views"
TRAIN_SAMPLING_APPLIED_KEY = "train_sampling_applied"
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


@attrs.define(slots=False)
class MultiviewVid2VidModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames_per_view: int = 1
    max_num_conditional_frames_per_view: int = 2
    train_sample_views_range: Tuple[int, int] | None = None
    condition_locations: ConditionLocationList = field(default_factory=lambda: ConditionLocationList([]))
    state_t: int = 0
    view_condition_dropout_max: int = 0
    online_text_embeddings_as_dict: bool = True  # For backward compatibility with old experiments
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames
    # Scheduled self-forcing for train-time conditioning replacement.
    self_forcing_enabled: bool = False
    self_forcing_prob: float = 0.0
    self_forcing_warmup_iter: int = 0
    self_forcing_ramp_iters: int = 0
    self_forcing_use_ema_teacher: bool = False
    # Self-forcing autoregressive rollout options.
    self_forcing_autoregressive: bool = False
    self_forcing_chunk_overlap: int = 0  # latent frames per view
    self_forcing_detach_rollout: bool = True
    self_forcing_max_rollout_chunks: int = 0  # 0 means use all chunks


class MultiviewVid2VidModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def __init__(self, config: MultiviewVid2VidModelRectifiedFlowConfig):
        super().__init__(config)
        self.state_t = config.state_t
        self.empty_string_text_embeddings = None
        self.neg_text_embeddings = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            compute_empty_and_negative_text_embeddings(self)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        # let n_views 2 also cp-encoded
        if n_views > 1 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.state_t
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        # let n_views 2 also cp-decoded
        if n_views > 1 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.state_t)
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = latent.shape[2] // self.state_t
        assert n_views <= cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return training_step_multiview(self, data_batch, iteration)

    def inplace_compute_text_embeddings_online(self, data_batch: dict[str, torch.Tensor]) -> None:
        inplace_compute_text_embeddings_online_multiview(self, data_batch)

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: MultiViewCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        n_views = x0_B_C_T_H_W.shape[2] // self.state_t
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        reshape_sigma_B_T = False
        if sigma_B_T is not None:
            assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
            if sigma_B_T.shape[-1] != 1:
                assert (
                    sigma_B_T.shape[-1] % n_views == 0
                ), f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                reshape_sigma_B_T = True
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if reshape_sigma_B_T:
            sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)

        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch_with_latent_view_indices = data_batch.copy()
        data_batch_with_latent_view_indices["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch_with_latent_view_indices

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_data_key if input_key is None else input_key
        is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True

        num_video_frames_per_view = (
            self.tokenizer.get_pixel_num_frames(self.state_t)
            if is_preprocessed
            else data_batch["num_video_frames_per_view"]
        )
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
        n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
        if input_key in data_batch:
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, MultiViewCondition]:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        raw_state, latent_state, condition = super(Video2WorldModelRectifiedFlow, self).get_data_and_condition(
            data_batch_with_latent_view_indices
        )
        condition = cast(MultiViewCondition, condition)
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=self.config.view_condition_dropout_max,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
            log.debug(f"Using {num_conditional_frames=} from data batch")
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch_with_latent_view_indices)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
        if parallel_state.is_initialized():
            pass
        else:
            assert (
                not self.net.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = uncond_v + guidance * (cond_v - uncond_v)  # align with pred2
            return velocity_pred

        return velocity_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        samples_B_C_T_H_W = super().generate_samples_from_batch(
            data_batch_with_latent_view_indices,
            guidance,
            seed,
            state_shape,
            n_sample,
            is_negative_prompt,
            num_steps,
            shift,
            **kwargs,
        )
        if cp_size > 1:
            samples_B_C_T_H_W = rearrange(
                samples_B_C_T_H_W, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.state_t // cp_size
            )
        return samples_B_C_T_H_W

    def set_up_model(self):
        """Override set_up_model to initialize cross-view attention from base model."""
        # Call parent's set_up_model first
        super().set_up_model()

        # Load base model and initialize cross-view attention if enabled
        if (
            hasattr(self.net, "enable_cross_view_attn")
            and self.net.enable_cross_view_attn
            and self.net.init_cross_view_attn_weight_from is not None
        ):
            log.info("Loading base model for cross-view attention initialization")
            self.net.init_cross_view_attn_with_self_attn_weights(is_ema=False)
            if self.config.ema.enabled:
                self.net_ema.init_cross_view_attn_with_self_attn_weights(is_ema=True)

            # Broadcast to ensure all ranks have consistent weights
            if self.fsdp_device_mesh is not None:
                log.info("Broadcasting cross-view attention weights for consistency")
                broadcast_dtensor_model_states(self.net, self.fsdp_device_mesh)
                if self.config.ema.enabled:
                    broadcast_dtensor_model_states(self.net_ema, self.fsdp_device_mesh)


def compute_text_embeddings_online_multiview_single_caption(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute prompt embeddings
    if len(data_batch["ai_caption"]) != 1:
        raise NotImplementedError(f"Expected batch size of 1, got {len(data_batch['ai_caption'])}")

    if len(data_batch["ai_caption"][0]) != 1:
        raise ValueError(f"Expected a single caption, got {len(data_batch['ai_caption'][0])}")

    caption = data_batch["ai_caption"][0][0]
    assert isinstance(caption, str)
    view0_text_embeddings_B_L_D = model.text_encoder.compute_text_embeddings_online(
        data_batch={model.input_caption_key: [caption]},
        input_caption_key=model.input_caption_key,
    )
    assert view0_text_embeddings_B_L_D.shape[0] == 1
    assert (
        view0_text_embeddings_B_L_D.shape[1] == 512
    ), f"view0_text_embeddings should be of shape (B, 512, D), got {view0_text_embeddings_B_L_D.shape}"
    output_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_neg_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_text_embeddings = rearrange(output_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    # Assign prompt embeddings to the front camera view
    for i_b in range(B):
        front_cam_view_idx_sample_position = data_batch["front_cam_view_idx_sample_position"][i_b]
        output_text_embeddings[front_cam_view_idx_sample_position, i_b] = view0_text_embeddings_B_L_D[i_b]
        output_neg_text_embeddings[front_cam_view_idx_sample_position, i_b] = model.neg_text_embeddings[0]
    output_text_embeddings = rearrange(output_text_embeddings, "V B L D -> B (V L) D")
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "V B L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)

    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview_multiple_captions(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute each view's caption separately
    if not len(data_batch["ai_caption"]) == 1:
        raise NotImplementedError(f"Expected batch size of 1, got {len(data_batch['ai_caption'])}")

    captions = data_batch["ai_caption"][0]
    if len(captions) != n_views:
        raise ValueError(f"Expected {n_views} captions, got {len(captions)}: {captions}")
    view_text_embeddings = []
    for caption in captions:
        data_batch_per_view = {model.input_caption_key: [caption]}
        view_text_embedding = model.text_encoder.compute_text_embeddings_online(
            data_batch_per_view, model.input_caption_key
        )
        view_text_embeddings.append(view_text_embedding)

    view_text_embeddings_B_V_L_D = torch.stack(view_text_embeddings, dim=1)
    assert view_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), f"view_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {view_text_embeddings_B_V_L_D.shape}"
    output_text_embeddings = rearrange(view_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    # repeat negative embedding for each per view
    output_neg_text_embeddings_L_D = model.neg_text_embeddings[0].clone()
    output_neg_text_embeddings_B_V_L_D = rearrange(output_neg_text_embeddings_L_D, "(B V L) D -> B V L D", B=1, V=1)
    output_neg_text_embeddings_B_V_L_D = output_neg_text_embeddings_B_V_L_D.repeat(B, n_views, 1, 1)
    assert output_neg_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), f"output_neg_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {output_neg_text_embeddings_B_V_L_D.shape}"
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captions = data_batch["ai_caption"][0]
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    assert len(captions) == 1 or len(captions) == n_views, f"Expected 1 or {n_views} captions, got {len(captions)}"
    if len(captions) == 1:
        return compute_text_embeddings_online_multiview_single_caption(model, data_batch)
    else:
        return compute_text_embeddings_online_multiview_multiple_captions(model, data_batch)


def inplace_compute_text_embeddings_online_multiview(model, data_batch: dict[str, torch.Tensor]) -> None:
    output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings = (
        compute_text_embeddings_online_multiview(model, data_batch)
    )
    t5_text_embeddings = {
        "text_embeddings": output_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    neg_t5_text_embeddings = {
        "text_embeddings": output_neg_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    data_batch["t5_text_embeddings"] = (
        t5_text_embeddings if model.config.online_text_embeddings_as_dict else t5_text_embeddings["text_embeddings"]
    )
    data_batch["neg_t5_text_embeddings"] = (
        neg_t5_text_embeddings
        if model.config.online_text_embeddings_as_dict
        else neg_t5_text_embeddings["text_embeddings"]
    )
    data_batch["t5_text_mask"] = torch.ones(
        output_text_embeddings.shape[0], output_text_embeddings.shape[1], device="cuda"
    )


def compute_empty_and_negative_text_embeddings(model):
    # Compute empty string embeddings for text embedding dropout
    if model.empty_string_text_embeddings is None:
        empty_string_data_batch = {
            model.input_caption_key: [" "],
        }
        model.empty_string_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            empty_string_data_batch, model.input_caption_key
        )

    # compute negative prompt embeddings for sampling
    if model.neg_text_embeddings is None:
        neg_promt_data_batch = {
            model.input_caption_key: [_DEFAULT_NEGATIVE_PROMPT],
        }
        model.neg_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            neg_promt_data_batch, model.input_caption_key
        )


def preprocess_databatch(
    data_batch: dict[str, Any],
    train_sample_views_range: Optional[tuple[int, int]],
) -> dict[str, Any]:
    """Preprocess data batch with dynamic view sampling."""

    if TRAIN_SAMPLING_APPLIED_KEY in data_batch and data_batch[TRAIN_SAMPLING_APPLIED_KEY] is True:
        return data_batch
    if train_sample_views_range is not None:
        min_views, max_views = train_sample_views_range
        log.debug(f"Randomly sampling {min_views} to {max_views} views")
        if SMOKE:
            train_sample_n_views = 1
        elif TRAIN_SAMPLE_N_VIEWS_KEY in data_batch:
            train_sample_n_views = data_batch[TRAIN_SAMPLE_N_VIEWS_KEY]
            log.debug(f"Using {TRAIN_SAMPLE_N_VIEWS_KEY} from data batch: {train_sample_n_views}")
            if train_sample_n_views < 1:  # No sampling is applied
                return data_batch
        else:
            # Sample n_views to keep
            train_sample_n_views = random.randint(min_views, max_views)
            if dist.is_initialized():
                # Have all ranks globally sample the same number of views such that tensors are same shape for efficiency
                train_sample_n_views_tensor = torch.tensor(
                    [train_sample_n_views], device=data_batch["sample_n_views"].device, dtype=torch.int64
                )
                dist.broadcast(train_sample_n_views_tensor, src=0)
                train_sample_n_views = train_sample_n_views_tensor[0].cpu().item()

        n_views = data_batch["sample_n_views"].cpu().item()
        log.debug(f"Sampling {train_sample_n_views} views out of {n_views}")
        available_view_indices = list(range(n_views))
        if data_batch["front_cam_view_idx_sample_position"][0] is not None and train_sample_n_views < n_views:
            # In cases where we only have a single caption, we shouldn't drop the front camera view
            front_cam_view_idx_sample_position = data_batch["front_cam_view_idx_sample_position"][0].cpu().item()
            available_view_indices.pop(front_cam_view_idx_sample_position)
        keep_view_indices = sorted(random.sample(available_view_indices, train_sample_n_views))
        if parallel_state.get_context_parallel_group() is not None:
            # Broadcast the exact views to keep for each context parallel group
            keep_view_indices = broadcast(keep_view_indices, parallel_state.get_context_parallel_group())
        log.debug(f"Sampled and broadcasted keep_view_indices={keep_view_indices}")
        data_batch = sample_n_views_from_data_batch(data_batch, keep_view_indices)
        data_batch[TRAIN_SAMPLING_APPLIED_KEY] = True
    return data_batch


def training_step_multiview(
    model, data_batch: dict[str, torch.Tensor], iteration: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Performs a single training step for the diffusion model.

    This method is responsible for executing one iteration of the model's training. It involves:
    1. Adding noise to the input data using the SDE process.
    2. Passing the noisy data through the network to generate predictions.
    3. Computing the loss based on the difference between the predictions and the original data, \
        considering any configured loss weighting.

    Args:
        data_batch (dict): raw data batch draw from the training data loader.
        iteration (int): Current iteration number.

    Returns:
        tuple: A tuple containing two elements:
            - dict: additional data that used to debug / logging / callbacks
            - Tensor: The computed loss for the training step as a PyTorch Tensor.

    Raises:
        AssertionError: If the class is conditional, \
            but no number of classes is specified in the network configuration.

    Notes:
        - The method handles different types of conditioning
        - The method also supports Kendall's loss
    """
    model._update_train_stats(data_batch)

    # only happens in training
    data_batch = preprocess_databatch(data_batch, model.config.train_sample_views_range)

    if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
        model.inplace_compute_text_embeddings_online(data_batch)

    # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
    _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)

    if _should_log_frame_debug(iteration):
        num_video_frames_per_view = data_batch.get("num_video_frames_per_view")
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.flatten()[0].cpu().item())
        elif num_video_frames_per_view is not None:
            num_video_frames_per_view = int(num_video_frames_per_view)
        n_views = _get_num_views(data_batch, condition, int(getattr(model, "state_t", 0)))
        latent_t_total = int(x0_B_C_T_H_W.shape[2])
        latent_t_per_view = latent_t_total // max(n_views, 1)
        log.info(
            "[self-forcing-train] iteration=%s pixel_frames_per_view=%s latent_frames_per_view=%s "
            "n_views=%s x0_shape=%s %s",
            iteration,
            num_video_frames_per_view,
            latent_t_per_view,
            n_views,
            tuple(x0_B_C_T_H_W.shape),
            _get_cuda_memory_debug_string(x0_B_C_T_H_W.device),
        )

    if _should_use_autoregressive_self_forcing(model, data_batch, condition, x0_B_C_T_H_W):
        return _training_step_multiview_autoregressive(model, x0_B_C_T_H_W, condition, iteration, data_batch)

    # Sample pertubation noise levels and N(0, 1) noises
    epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)
    batch_size = x0_B_C_T_H_W.size()[0]
    t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
    t_B = rearrange(t_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
    x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = model.broadcast_split_for_model_parallelsim(
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
    )
    timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)
    sigmas = model.rectified_flow.get_sigmas(
        timesteps,
        model.tensor_kwargs_fp32,
    )
    timesteps = rearrange(timesteps, "b -> b 1")
    sigmas = rearrange(sigmas, "b -> b 1")
    xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

    self_forcing_applied = False
    self_forcing_prob = _get_self_forcing_probability(model, iteration)
    if _should_apply_self_forcing(model, self_forcing_prob, condition):
        condition = _apply_self_forcing(
            model=model,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W,
            timesteps_B_T=timesteps,
        )
        self_forcing_applied = True

    vt_pred_B_C_T_H_W = model.denoise(
        noise=epsilon_B_C_T_H_W,
        xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
        timesteps_B_T=timesteps,
        condition=condition,
    )

    time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)
    per_instance_loss = torch.mean((vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim())))

    loss = torch.mean(time_weights_B * per_instance_loss)

    output_batch = {
        "x0": x0_B_C_T_H_W,
        "xt": xt_B_C_T_H_W,
        "sigma": sigmas,
        "condition": condition,
        "model_pred": vt_pred_B_C_T_H_W,
        "edm_loss": loss,
        "self_forcing_applied": torch.tensor(float(self_forcing_applied), device=loss.device),
        "self_forcing_prob": torch.tensor(float(self_forcing_prob), device=loss.device),
    }

    return output_batch, loss


def _condition_replace(condition: Any, **updates: Any) -> Any:
    kwargs = condition.to_dict(skip_underscore=False)
    kwargs.update(updates)
    return type(condition)(**kwargs)


def _should_log_frame_debug(iteration: int) -> bool:
    if dist.is_initialized() and dist.get_rank() != 0:
        return False
    return iteration < 5 or iteration % 100 == 0


def _get_cuda_memory_debug_string(device: torch.device | None = None) -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if device is None or device.type != "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    allocated_gb = torch.cuda.memory_allocated(device=device) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device=device) / (1024**3)
    max_allocated_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
    max_reserved_gb = torch.cuda.max_memory_reserved(device=device) / (1024**3)
    return (
        f"cuda_allocated_gb={allocated_gb:.2f} "
        f"cuda_reserved_gb={reserved_gb:.2f} "
        f"cuda_max_allocated_gb={max_allocated_gb:.2f} "
        f"cuda_max_reserved_gb={max_reserved_gb:.2f}"
    )


def _get_self_forcing_probability(model: Any, iteration: int) -> float:
    enabled = bool(getattr(model.config, "self_forcing_enabled", False))
    if not enabled:
        return 0.0
    max_prob = float(getattr(model.config, "self_forcing_prob", 0.0))
    if max_prob <= 0.0:
        return 0.0
    warmup_iter = int(getattr(model.config, "self_forcing_warmup_iter", 0))
    if iteration < warmup_iter:
        return 0.0
    ramp_iters = int(getattr(model.config, "self_forcing_ramp_iters", 0))
    if ramp_iters <= 0:
        return max_prob
    progress = min(1.0, float(iteration - warmup_iter + 1) / float(ramp_iters))
    return max_prob * progress


def _should_apply_self_forcing(model: Any, self_forcing_prob: float, condition: Any) -> bool:
    if self_forcing_prob <= 0.0:
        return False
    if not hasattr(condition, "condition_video_input_mask_B_C_T_H_W"):
        return False
    condition_mask = condition.condition_video_input_mask_B_C_T_H_W
    if condition_mask is None or condition_mask.numel() == 0:
        return False
    return _sample_synced_self_forcing_decision(self_forcing_prob, condition_mask.device)


def _apply_self_forcing(
    model: Any,
    condition: Any,
    epsilon_B_C_T_H_W: torch.Tensor,
    xt_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
):
    """
    Self-forcing for train-time conditioning:
    1) Run a no-grad teacher pass without GT-frame injection mask.
    2) Convert velocity prediction to x0.
    3) Replace only conditional-frame GT slots with predicted x0.
    """
    condition_mask = condition.condition_video_input_mask_B_C_T_H_W
    if condition_mask is None:
        return condition

    teacher_condition = _condition_replace(
        condition,
        condition_video_input_mask_B_C_T_H_W=torch.zeros_like(condition_mask),
    )

    prev_net = None
    use_ema_teacher = bool(getattr(model.config, "self_forcing_use_ema_teacher", False))
    if use_ema_teacher and hasattr(model, "net_ema") and model.net_ema is not None:
        # Guard against EMA on CPU while training tensors are on CUDA.
        ema_param = next(model.net_ema.parameters(), None)
        train_device = xt_B_C_T_H_W.device
        if ema_param is not None and ema_param.device == train_device:
            prev_net = model.net
            model.net = model.net_ema

    has_denoise_replace = hasattr(model.config, "denoise_replace_gt_frames")
    prev_denoise_replace = getattr(model.config, "denoise_replace_gt_frames", None)
    if has_denoise_replace:
        model.config.denoise_replace_gt_frames = False

    try:
        with torch.no_grad():
            teacher_v_B_C_T_H_W = model.denoise(
                noise=epsilon_B_C_T_H_W,
                xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
                timesteps_B_T=timesteps_B_T,
                condition=teacher_condition,
            ).float()
            teacher_x0_B_C_T_H_W = epsilon_B_C_T_H_W - teacher_v_B_C_T_H_W
            teacher_x0_B_C_T_H_W = teacher_x0_B_C_T_H_W.to(dtype=condition.gt_frames.dtype)
            cond_mask = condition_mask.to(dtype=condition.gt_frames.dtype)
            replaced_gt = cond_mask * teacher_x0_B_C_T_H_W + (1.0 - cond_mask) * condition.gt_frames
            condition = _condition_replace(condition, gt_frames=replaced_gt.detach())
    finally:
        if has_denoise_replace:
            model.config.denoise_replace_gt_frames = prev_denoise_replace
        if prev_net is not None:
            model.net = prev_net

    return condition


def _should_use_autoregressive_self_forcing(
    model: Any,
    data_batch: dict[str, torch.Tensor],
    condition: Any,
    x0_B_C_T_H_W: torch.Tensor,
) -> bool:
    if not bool(getattr(model.config, "self_forcing_enabled", False)):
        return False
    if not bool(getattr(model.config, "self_forcing_autoregressive", False)):
        return False

    state_t = int(getattr(model, "state_t", 0))
    if state_t <= 0:
        return False

    total_t = int(x0_B_C_T_H_W.shape[2])
    n_views = _get_num_views(data_batch, condition, state_t)
    if n_views <= 0 or total_t % n_views != 0:
        return False
    total_t_per_view = total_t // n_views
    if total_t_per_view <= state_t:
        return False

    overlap = _get_self_forcing_overlap(model, state_t)
    if overlap <= 0:
        return False

    return True


def _get_context_parallel_group_safe():
    if hasattr(parallel_state, "is_initialized") and parallel_state.is_initialized():
        return parallel_state.get_context_parallel_group()
    return None


def _training_step_multiview_autoregressive(
    model: Any,
    x0_B_C_T_H_W: torch.Tensor,
    condition: Any,
    iteration: int,
    data_batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state_t = int(model.state_t)
    n_views = _get_num_views(data_batch, condition, state_t)
    total_t_per_view = x0_B_C_T_H_W.shape[2] // n_views
    overlap = _get_self_forcing_overlap(model, state_t)
    step = max(1, state_t - overlap)
    num_chunks = max(1, (total_t_per_view - overlap + step - 1) // step)

    max_chunks = int(getattr(model.config, "self_forcing_max_rollout_chunks", 0))
    if max_chunks > 0:
        num_chunks = min(num_chunks, max_chunks)

    losses = []
    self_forcing_prob = _get_self_forcing_probability(model, iteration)
    self_forcing_applied = False
    prev_generated_x0 = None
    last_output_batch = None

    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * step
        end_frame = min(start_frame + state_t, total_t_per_view)
        if start_frame >= total_t_per_view:
            break

        x0_chunk = _slice_multiview_window(x0_B_C_T_H_W, n_views=n_views, start_frame=start_frame, end_frame=end_frame)
        condition_chunk = _slice_condition_window(
            condition=condition,
            n_views=n_views,
            start_frame=start_frame,
            end_frame=end_frame,
            total_t_per_view=total_t_per_view,
        )

        if _should_log_frame_debug(iteration):
            log.info(
                "[self-forcing-ar] iteration=%s chunk=%s/%s latent_frames_per_view=%s "
                "window=[%s,%s) overlap=%s n_views=%s x0_chunk_shape=%s %s",
                iteration,
                chunk_idx + 1,
                num_chunks,
                end_frame - start_frame,
                start_frame,
                end_frame,
                overlap,
                n_views,
                tuple(x0_chunk.shape),
                _get_cuda_memory_debug_string(x0_chunk.device),
            )

        if chunk_idx > 0 and prev_generated_x0 is not None:
            apply_rollout_self_forcing = _sample_synced_self_forcing_decision(self_forcing_prob, x0_chunk.device)
            if apply_rollout_self_forcing:
                detach_rollout = bool(getattr(model.config, "self_forcing_detach_rollout", True))
                rollout_source = prev_generated_x0.detach() if detach_rollout else prev_generated_x0
                condition_chunk, replaced = _inject_rollout_prediction_into_condition(
                    condition=condition_chunk,
                    generated_prev_chunk_x0=rollout_source,
                    n_views=n_views,
                    overlap=overlap,
                )
                self_forcing_applied = self_forcing_applied or replaced

        output_batch_chunk, loss_chunk = _run_single_chunk_training_step(model, x0_chunk, condition_chunk)
        losses.append(loss_chunk)
        last_output_batch = output_batch_chunk

        process_group = _get_context_parallel_group_safe()
        cp_size = 1 if process_group is None else len(get_process_group_ranks(process_group))
        if cp_size > 1:
            # In CP mode model_pred is shard-local. Build a full-chunk rollout prediction for next chunk conditioning.
            prev_generated_x0 = _predict_rollout_x0_full(
                model=model,
                condition_chunk=condition_chunk,
                epsilon_full_B_C_T_H_W=output_batch_chunk["epsilon_full"],
                xt_full_B_C_T_H_W=output_batch_chunk["xt_full"],
                timesteps_full_B_T=output_batch_chunk["timesteps_full"],
            )
        else:
            prev_generated_x0 = (output_batch_chunk["epsilon"] - output_batch_chunk["model_pred"]).to(
                dtype=condition_chunk.gt_frames.dtype
            )

    if not losses:
        raise RuntimeError("Autoregressive self-forcing was enabled but no training chunks were processed.")

    loss = torch.stack(losses).mean()
    assert last_output_batch is not None
    output_batch = {
        "x0": last_output_batch["x0"],
        "xt": last_output_batch["xt"],
        "sigma": last_output_batch["sigma"],
        "condition": last_output_batch["condition"],
        "model_pred": last_output_batch["model_pred"],
        "edm_loss": loss,
        "self_forcing_applied": torch.tensor(float(self_forcing_applied), device=loss.device),
        "self_forcing_prob": torch.tensor(float(self_forcing_prob), device=loss.device),
        "self_forcing_num_chunks": torch.tensor(float(len(losses)), device=loss.device),
    }
    return output_batch, loss


def _run_single_chunk_training_step(
    model: Any,
    x0_chunk_B_C_T_H_W: torch.Tensor,
    condition_chunk: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    epsilon_full_B_C_T_H_W = torch.randn(x0_chunk_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)
    batch_size = x0_chunk_B_C_T_H_W.size()[0]
    t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
    t_B_full_B_T = rearrange(t_B, "b -> b 1")
    timesteps_full_B_T = model.rectified_flow.get_discrete_timestamp(t_B_full_B_T, model.tensor_kwargs_fp32)
    sigmas_full_B_T = model.rectified_flow.get_sigmas(
        timesteps_full_B_T,
        model.tensor_kwargs_fp32,
    )
    timesteps_full_B_T = rearrange(timesteps_full_B_T, "b -> b 1")
    sigmas_full_B_T = rearrange(sigmas_full_B_T, "b -> b 1")
    xt_full_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(
        epsilon_full_B_C_T_H_W, x0_chunk_B_C_T_H_W, sigmas_full_B_T
    )

    x0_B_C_T_H_W, condition_chunk, epsilon_B_C_T_H_W, t_B = model.broadcast_split_for_model_parallelsim(
        x0_chunk_B_C_T_H_W, condition_chunk, epsilon_full_B_C_T_H_W, t_B_full_B_T
    )
    timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)
    sigmas = model.rectified_flow.get_sigmas(
        timesteps,
        model.tensor_kwargs_fp32,
    )
    timesteps = rearrange(timesteps, "b -> b 1")
    sigmas = rearrange(sigmas, "b -> b 1")
    xt_B_C_T_H_W, vt_local_B_C_T_H_W = model.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

    vt_pred_B_C_T_H_W = model.denoise(
        noise=epsilon_B_C_T_H_W,
        xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
        timesteps_B_T=timesteps,
        condition=condition_chunk,
    )
    time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)
    per_instance_loss = torch.mean(
        (vt_pred_B_C_T_H_W - vt_local_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
    )
    loss = torch.mean(time_weights_B * per_instance_loss)
    output_batch = {
        "x0": x0_B_C_T_H_W,
        "xt": xt_B_C_T_H_W,
        "sigma": sigmas,
        "condition": condition_chunk,
        "model_pred": vt_pred_B_C_T_H_W,
        "epsilon": epsilon_B_C_T_H_W,
        "epsilon_full": epsilon_full_B_C_T_H_W,
        "xt_full": xt_full_B_C_T_H_W,
        "timesteps_full": timesteps_full_B_T,
    }
    return output_batch, loss


def _get_num_views(data_batch: dict[str, torch.Tensor], condition: Any, state_t: int) -> int:
    view_indices_B_T = getattr(condition, "view_indices_B_T", None)
    if view_indices_B_T is not None and view_indices_B_T.ndim == 2 and state_t > 0:
        if view_indices_B_T.shape[1] % state_t == 0:
            n_views = int(view_indices_B_T.shape[1] // state_t)
            if n_views > 0:
                return n_views
    sample_n_views = data_batch.get("sample_n_views")
    if isinstance(sample_n_views, torch.Tensor) and sample_n_views.numel() > 0:
        return int(sample_n_views.flatten()[0].item())
    if isinstance(sample_n_views, int):
        return sample_n_views
    return 1


def _get_self_forcing_overlap(model: Any, state_t: int) -> int:
    overlap = int(getattr(model.config, "self_forcing_chunk_overlap", 0))
    if overlap <= 0:
        overlap = int(getattr(model.config, "max_num_conditional_frames_per_view", 0))
    if overlap <= 0:
        overlap = int(getattr(model.config, "max_num_conditional_frames", 0))
    return min(max(overlap, 0), max(state_t - 1, 0))


def _slice_multiview_window(
    tensor_B_C_VT_H_W: torch.Tensor,
    n_views: int,
    start_frame: int,
    end_frame: int,
) -> torch.Tensor:
    tensor_B_C_V_T_H_W = rearrange(tensor_B_C_VT_H_W, "B C (V T) H W -> B C V T H W", V=n_views)
    tensor_B_C_V_T_H_W = tensor_B_C_V_T_H_W[:, :, :, start_frame:end_frame, :, :]
    return rearrange(tensor_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")


def _slice_condition_window(
    condition: Any,
    n_views: int,
    start_frame: int,
    end_frame: int,
    total_t_per_view: int,
) -> Any:
    updates = {}
    expected_total_t = n_views * total_t_per_view

    for key in ["gt_frames", "condition_video_input_mask_B_C_T_H_W", "latent_control_input"]:
        value = getattr(condition, key, None)
        if value is None or not isinstance(value, torch.Tensor) or value.ndim != 5:
            continue
        if value.shape[2] != expected_total_t:
            continue
        updates[key] = _slice_multiview_window(value, n_views=n_views, start_frame=start_frame, end_frame=end_frame)

    view_indices_B_T = getattr(condition, "view_indices_B_T", None)
    if isinstance(view_indices_B_T, torch.Tensor) and view_indices_B_T.ndim == 2:
        if view_indices_B_T.shape[1] == expected_total_t:
            view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_views)
            updates["view_indices_B_T"] = rearrange(view_indices_B_V_T[:, :, start_frame:end_frame], "B V T -> B (V T)")
        elif view_indices_B_T.shape[1] % n_views == 0:
            # Fallback for batches where only one chunk worth of view indices is provided.
            base_t = view_indices_B_T.shape[1] // n_views
            view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_views)
            repeats = max(1, (total_t_per_view + base_t - 1) // base_t)
            view_indices_tiled = view_indices_B_V_T.repeat(1, 1, repeats)[:, :, :total_t_per_view]
            updates["view_indices_B_T"] = rearrange(view_indices_tiled[:, :, start_frame:end_frame], "B V T -> B (V T)")

    if not updates:
        return condition
    return _condition_replace(condition, **updates)


def _inject_rollout_prediction_into_condition(
    condition: Any,
    generated_prev_chunk_x0: torch.Tensor,
    n_views: int,
    overlap: int,
) -> tuple[Any, bool]:
    if overlap <= 0:
        return condition, False
    condition_gt = getattr(condition, "gt_frames", None)
    if condition_gt is None:
        return condition, False

    cur_chunk_t = int(condition_gt.shape[2] // n_views)
    prev_chunk_t = int(generated_prev_chunk_x0.shape[2] // n_views)
    overlap = min(overlap, cur_chunk_t, prev_chunk_t)
    if overlap <= 0:
        return condition, False

    prev_tail = _slice_multiview_window(
        generated_prev_chunk_x0, n_views=n_views, start_frame=prev_chunk_t - overlap, end_frame=prev_chunk_t
    ).to(dtype=condition_gt.dtype, device=condition_gt.device)
    prev_tail_B_C_V_T_H_W = rearrange(prev_tail, "B C (V T) H W -> B C V T H W", V=n_views)
    gt_B_C_V_T_H_W = rearrange(condition_gt, "B C (V T) H W -> B C V T H W", V=n_views).clone()
    gt_B_C_V_T_H_W[:, :, :, :overlap, :, :] = prev_tail_B_C_V_T_H_W

    updates = {"gt_frames": rearrange(gt_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")}
    condition_mask = getattr(condition, "condition_video_input_mask_B_C_T_H_W", None)
    if condition_mask is not None:
        mask_B_C_V_T_H_W = rearrange(condition_mask, "B C (V T) H W -> B C V T H W", V=n_views).clone()
        mask_B_C_V_T_H_W[:, :, :, :overlap, :, :] = 1.0
        updates["condition_video_input_mask_B_C_T_H_W"] = rearrange(mask_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")

    return _condition_replace(condition, **updates), True


def _sample_synced_self_forcing_decision(probability: float, device: torch.device) -> bool:
    if probability <= 0.0:
        return False
    local = torch.tensor(float(torch.rand((), device=device).item() < probability), device=device)
    cp_group = _get_context_parallel_group_safe()
    if cp_group is not None:
        local = broadcast(local, cp_group)
    return bool(local.item() > 0.5)


def _predict_rollout_x0_full(
    model: Any,
    condition_chunk: Any,
    epsilon_full_B_C_T_H_W: torch.Tensor,
    xt_full_B_C_T_H_W: torch.Tensor,
    timesteps_full_B_T: torch.Tensor,
) -> torch.Tensor:
    cp_group = _get_context_parallel_group_safe()
    was_cp_enabled = bool(getattr(model.net, "is_context_parallel_enabled", False))
    if was_cp_enabled and hasattr(model.net, "disable_context_parallel"):
        model.net.disable_context_parallel()

    has_denoise_replace = hasattr(model.config, "denoise_replace_gt_frames")
    prev_denoise_replace = getattr(model.config, "denoise_replace_gt_frames", None)
    if has_denoise_replace:
        model.config.denoise_replace_gt_frames = False

    try:
        with torch.no_grad():
            vt_pred_full = model.denoise(
                noise=epsilon_full_B_C_T_H_W,
                xt_B_C_T_H_W=xt_full_B_C_T_H_W.to(**model.tensor_kwargs),
                timesteps_B_T=timesteps_full_B_T,
                condition=condition_chunk,
            ).float()
            x0_pred_full = epsilon_full_B_C_T_H_W - vt_pred_full
            x0_pred_full = x0_pred_full.to(dtype=condition_chunk.gt_frames.dtype)
    finally:
        if has_denoise_replace:
            model.config.denoise_replace_gt_frames = prev_denoise_replace
        if was_cp_enabled and cp_group is not None and hasattr(model.net, "enable_context_parallel"):
            model.net.enable_context_parallel(cp_group)

    return x0_pred_full
