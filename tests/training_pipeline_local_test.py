from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from cosmos_transfer2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_transfer2._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    _training_step_multiview_autoregressive,
)
from cosmos_transfer2._src.transfer2_multiview.models.multiview_vid2vid_model_control_vace_rectified_flow import (
    MultiviewControlVideo2WorldModelRectifiedFlow,
)
from cosmos_transfer2._src.transfer2_multiview.networks import multiview_dit_control as _mv_dit_control
from cosmos_transfer2._src.transfer2_multiview.networks.multiview_dit_control import MultiViewControlDiT

try:
    import megatron.core.parallel_state as _parallel_state
except ImportError:
    _parallel_state = None

_MODEL_DTYPE = torch.bfloat16


@dataclass
class _Condition:
    gt_frames: torch.Tensor
    condition_video_input_mask_B_C_T_H_W: torch.Tensor
    view_indices_B_T: torch.Tensor
    crossattn_emb: torch.Tensor
    fps: torch.Tensor
    latent_control_input: torch.Tensor | None = None
    is_video: bool = True
    use_video_condition: bool = True

    def to_dict(self, skip_underscore: bool = False) -> dict:
        del skip_underscore
        return {
            "gt_frames": self.gt_frames,
            "condition_video_input_mask_B_C_T_H_W": self.condition_video_input_mask_B_C_T_H_W,
            "view_indices_B_T": self.view_indices_B_T,
            "crossattn_emb": self.crossattn_emb,
            "fps": self.fps,
            "latent_control_input": self.latent_control_input,
            "is_video": self.is_video,
            "use_video_condition": self.use_video_condition,
        }


def _build_real_net(device: torch.device) -> MultiViewControlDiT:
    net = MultiViewControlDiT(
        max_img_h=8,
        max_img_w=8,
        max_frames=16,
        state_t=4,
        in_channels=1,
        out_channels=1,
        patch_spatial=1,
        patch_temporal=1,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        concat_padding_mask=False,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=False,
        adaln_lora_dim=16,
        atten_backend="minimal_a2a",
        extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
        n_cameras=2,
        n_cameras_emb=2,
        view_condition_dim=1,
        concat_view_embedding=False,
        use_input_hint_block=False,
        condition_strategy="first_n",
        vace_block_every_n=1,
        num_max_modalities=1,
        use_wan_fp32_strategy=True,
    )
    net.to(device=device, dtype=_MODEL_DTYPE)
    net.init_weights()
    return net


def _patch_single_process_context_parallel() -> None:
    orig_get_process_group_ranks = _mv_dit_control.get_process_group_ranks

    def _safe_get_process_group_ranks(pg):
        if pg is None:
            return [0]
        return orig_get_process_group_ranks(pg)

    _mv_dit_control.get_process_group_ranks = _safe_get_process_group_ranks

    if _parallel_state is not None and hasattr(_parallel_state, "is_initialized"):
        if not _parallel_state.is_initialized():
            _parallel_state.get_context_parallel_group = lambda: None


def _build_real_model(
    *,
    enable_autoregressive: bool,
    self_forcing_prob: float = 1.0,
) -> MultiviewControlVideo2WorldModelRectifiedFlow:
    device = torch.device("cuda")
    model = object.__new__(MultiviewControlVideo2WorldModelRectifiedFlow)
    nn.Module.__init__(model)
    model.state_t = 4
    model.tensor_kwargs = {"device": device, "dtype": _MODEL_DTYPE}
    model.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
    model.config = SimpleNamespace(
        train_sample_views_range=None,
        text_encoder_config=None,
        self_forcing_enabled=True,
        self_forcing_prob=self_forcing_prob,
        self_forcing_warmup_iter=0,
        self_forcing_ramp_iters=0,
        self_forcing_use_ema_teacher=False,
        self_forcing_autoregressive=enable_autoregressive,
        self_forcing_chunk_overlap=2,
        self_forcing_detach_rollout=True,
        self_forcing_max_rollout_chunks=0,
        max_num_conditional_frames_per_view=2,
        denoise_replace_gt_frames=True,
        conditional_frame_timestep=-1.0,
    )
    model.net = _build_real_net(device)
    model.net_ema = _build_real_net(device)
    model.rectified_flow = RectifiedFlow(
        velocity_field=model.net,
        train_time_distribution="uniform",
        train_time_weight_method="uniform",
        use_dynamic_shift=False,
        shift=3,
        device=device,
        dtype=torch.float32,
    )
    return model


def _make_dummy_training_inputs(
    device: torch.device, *, n_views: int = 2, total_t_per_view: int = 6, state_t: int = 4
) -> tuple[torch.Tensor, _Condition, dict[str, torch.Tensor]]:
    total_t = n_views * total_t_per_view
    x0 = torch.randn((1, 1, total_t, 1, 1), device=device, dtype=_MODEL_DTYPE)
    cond_mask = torch.zeros_like(x0)
    cond_mask[:, :, : (n_views * 2)] = 1.0  # mark first 2 frames per view as conditional frames

    view_indices = torch.arange(n_views, device=device, dtype=torch.long).repeat_interleave(state_t).unsqueeze(0)
    crossattn_emb = torch.zeros((1, n_views * 512, 1024), device=device, dtype=_MODEL_DTYPE)
    fps = torch.tensor([10.0], device=device, dtype=torch.float32)
    latent_control_input = torch.zeros_like(x0)

    condition = _Condition(
        gt_frames=x0.clone(),
        condition_video_input_mask_B_C_T_H_W=cond_mask,
        view_indices_B_T=view_indices,
        crossattn_emb=crossattn_emb,
        fps=fps,
        latent_control_input=latent_control_input,
    )
    data_batch = {"sample_n_views": torch.tensor([n_views], device=device, dtype=torch.long)}
    return x0, condition, data_batch


def _patch_training_inputs(
    model: MultiviewControlVideo2WorldModelRectifiedFlow,
    x0: torch.Tensor,
    condition: _Condition,
) -> None:
    def _get_data_and_condition(_data_batch):
        return x0.clone(), x0.clone(), condition

    def _broadcast_split_for_model_parallelsim(x0_in, condition_in, epsilon_in, sigma_in):
        return x0_in, condition_in, epsilon_in, sigma_in

    model._update_train_stats = lambda _data_batch: None
    model.get_data_and_condition = _get_data_and_condition
    model.broadcast_split_for_model_parallelsim = _broadcast_split_for_model_parallelsim


def _assert_has_grad(model: nn.Module) -> None:
    grad_found = False
    for p in model.parameters():
        if p.grad is not None:
            if torch.isfinite(p.grad).all():
                grad_found = True
                break
    assert grad_found, "Expected at least one finite gradient after backward pass."


def test_training_pipeline_standard() -> None:
    model = _build_real_model(enable_autoregressive=False, self_forcing_prob=1.0)
    x0, condition, data_batch = _make_dummy_training_inputs(device=torch.device("cuda"), total_t_per_view=4)
    _patch_training_inputs(model, x0, condition)

    model.train()
    model.net.zero_grad(set_to_none=True)
    output_batch, loss = model.training_step(data_batch, iteration=10)
    loss.backward()

    assert torch.isfinite(loss)
    assert output_batch["model_pred"].shape == x0.shape
    assert "self_forcing_applied" in output_batch
    _assert_has_grad(model.net)


def test_training_pipeline_autoregressive() -> None:
    model = _build_real_model(enable_autoregressive=True, self_forcing_prob=1.0)
    x0, condition, data_batch = _make_dummy_training_inputs(device=torch.device("cuda"), total_t_per_view=6)

    model.train()
    model.net.zero_grad(set_to_none=True)
    output_batch, loss = _training_step_multiview_autoregressive(
        model=model,
        x0_B_C_T_H_W=x0,
        condition=condition,
        iteration=10,
        data_batch=data_batch,
    )
    loss.backward()

    assert torch.isfinite(loss)
    expected_t = int(data_batch["sample_n_views"].item()) * model.state_t
    assert output_batch["model_pred"].shape[2] == expected_t
    assert "self_forcing_num_chunks" in output_batch
    assert float(output_batch["self_forcing_num_chunks"].item()) >= 2.0
    _assert_has_grad(model.net)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This local script expects CUDA. Please run in your GPU environment.")

    _patch_single_process_context_parallel()

    test_training_pipeline_standard()
    test_training_pipeline_autoregressive()

    print("training_pipeline_local_test.py: all checks passed")


if __name__ == "__main__":
    main()
