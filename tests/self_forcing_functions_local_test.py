from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from cosmos_transfer2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_transfer2._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    _apply_self_forcing,
    _get_num_views,
    _get_self_forcing_overlap,
    _get_self_forcing_probability,
    _inject_rollout_prediction_into_condition,
    _predict_rollout_x0_full,
    _run_single_chunk_training_step,
    _sample_synced_self_forcing_decision,
    _should_apply_self_forcing,
    _should_use_autoregressive_self_forcing,
    _slice_condition_window,
    _slice_multiview_window,
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
    """
    Force single-process behavior for local tests when Megatron model-parallel
    state has not been initialized.
    """
    orig_get_process_group_ranks = _mv_dit_control.get_process_group_ranks

    def _safe_get_process_group_ranks(pg):
        if pg is None:
            return [0]
        return orig_get_process_group_ranks(pg)

    _mv_dit_control.get_process_group_ranks = _safe_get_process_group_ranks

    if _parallel_state is not None and hasattr(_parallel_state, "is_initialized"):
        if not _parallel_state.is_initialized():
            _parallel_state.get_context_parallel_group = lambda: None


def _build_real_model() -> MultiviewControlVideo2WorldModelRectifiedFlow:
    device = torch.device("cuda")
    model = object.__new__(MultiviewControlVideo2WorldModelRectifiedFlow)
    nn.Module.__init__(model)
    model.state_t = 4
    model.tensor_kwargs = {"device": device, "dtype": _MODEL_DTYPE}
    model.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
    model.config = SimpleNamespace(
        self_forcing_enabled=True,
        self_forcing_prob=1.0,
        self_forcing_warmup_iter=2,
        self_forcing_ramp_iters=3,
        self_forcing_use_ema_teacher=False,
        self_forcing_autoregressive=True,
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


def _make_test_tensors(device: torch.device):
    n_views = 2
    total_t_per_view = 6
    total_t = n_views * total_t_per_view
    gt = torch.arange(total_t, device=device, dtype=_MODEL_DTYPE).view(1, 1, total_t, 1, 1)
    mask = torch.zeros_like(gt)
    # Keep latent view indices aligned with state_t=4 per view (B, V*state_t = 8),
    # which matches how _get_num_views infers number of views during training.
    view_indices = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device, dtype=torch.long)
    # MultiViewCrossAttention infers number of cameras from context length: n_cameras = context_len // 512.
    # Provide 512 tokens per view so n_cameras matches n_views in this test.
    crossattn_emb = torch.zeros((1, n_views * 512, 1024), device=device, dtype=_MODEL_DTYPE)
    fps = torch.tensor([10.0], device=device, dtype=torch.float32)
    latent_control_input = torch.zeros_like(gt)
    cond = _Condition(
        gt_frames=gt.clone(),
        condition_video_input_mask_B_C_T_H_W=mask.clone(),
        view_indices_B_T=view_indices.clone(),
        crossattn_emb=crossattn_emb,
        fps=fps,
        latent_control_input=latent_control_input,
    )
    data_batch = {"sample_n_views": torch.tensor([n_views], device=device, dtype=torch.long)}
    return gt, cond, data_batch


def test_probability_schedule(model: MultiviewControlVideo2WorldModelRectifiedFlow):
    assert _get_self_forcing_probability(model, 0) == 0.0
    assert _get_self_forcing_probability(model, 1) == 0.0
    p2 = _get_self_forcing_probability(model, 2)
    p3 = _get_self_forcing_probability(model, 3)
    p10 = _get_self_forcing_probability(model, 10)
    assert 0.0 < p2 <= p3 <= 1.0
    assert p10 == 1.0


def test_synced_decision_and_apply_gate(model: MultiviewControlVideo2WorldModelRectifiedFlow, cond: _Condition):
    assert _sample_synced_self_forcing_decision(0.0, cond.gt_frames.device) is False
    assert _sample_synced_self_forcing_decision(1.0, cond.gt_frames.device) is True
    assert _should_apply_self_forcing(model, 1.0, cond) is True
    cond_zero_mask = _Condition(
        gt_frames=cond.gt_frames,
        condition_video_input_mask_B_C_T_H_W=torch.empty(0, device=cond.gt_frames.device),
        view_indices_B_T=cond.view_indices_B_T,
        crossattn_emb=cond.crossattn_emb,
        fps=cond.fps,
        latent_control_input=cond.latent_control_input,
    )
    assert _should_apply_self_forcing(model, 1.0, cond_zero_mask) is False


def test_slice_helpers(cond: _Condition):
    n_views = 2
    sliced = _slice_multiview_window(cond.gt_frames, n_views=n_views, start_frame=1, end_frame=4)
    assert sliced.shape == (1, 1, n_views * 3, 1, 1)

    sliced_cond = _slice_condition_window(cond, n_views=n_views, start_frame=1, end_frame=4, total_t_per_view=6)
    assert sliced_cond.gt_frames.shape == (1, 1, n_views * 3, 1, 1)
    assert sliced_cond.view_indices_B_T.shape[-1] == n_views * 3


def test_apply_self_forcing(model: MultiviewControlVideo2WorldModelRectifiedFlow, cond: _Condition):
    n_views = 2
    cond_chunk = _slice_condition_window(cond, n_views=n_views, start_frame=0, end_frame=4, total_t_per_view=6)
    eps = torch.full_like(cond_chunk.gt_frames, 7.0)
    xt = cond_chunk.gt_frames.clone()
    timesteps = torch.zeros((1, 1), device=cond_chunk.gt_frames.device, dtype=_MODEL_DTYPE)

    mask_v = torch.zeros((1, 1, n_views, 4, 1, 1), device=cond_chunk.gt_frames.device, dtype=_MODEL_DTYPE)
    mask_v[:, :, :, :2] = 1.0
    cond_masked = _Condition(
        gt_frames=cond_chunk.gt_frames.clone(),
        condition_video_input_mask_B_C_T_H_W=mask_v.view(1, 1, n_views * 4, 1, 1).clone(),
        view_indices_B_T=cond_chunk.view_indices_B_T,
        crossattn_emb=cond_chunk.crossattn_emb,
        fps=cond_chunk.fps,
        latent_control_input=cond_chunk.latent_control_input,
    )

    out = _apply_self_forcing(model, cond_masked, eps, xt, timesteps)

    teacher_cond = _Condition(
        gt_frames=cond_masked.gt_frames,
        condition_video_input_mask_B_C_T_H_W=torch.zeros_like(cond_masked.condition_video_input_mask_B_C_T_H_W),
        view_indices_B_T=cond_masked.view_indices_B_T,
        crossattn_emb=cond_masked.crossattn_emb,
        fps=cond_masked.fps,
        latent_control_input=cond_masked.latent_control_input,
    )
    prev_replace = model.config.denoise_replace_gt_frames
    model.config.denoise_replace_gt_frames = False
    try:
        teacher_v = model.denoise(eps, xt, timesteps, teacher_cond)
    finally:
        model.config.denoise_replace_gt_frames = prev_replace
    teacher_x0 = eps - teacher_v
    expected = cond_masked.condition_video_input_mask_B_C_T_H_W * teacher_x0 + (
        1.0 - cond_masked.condition_video_input_mask_B_C_T_H_W
    ) * cond_masked.gt_frames
    expected = expected.to(out.gt_frames.dtype)
    assert torch.allclose(out.gt_frames, expected, atol=2e-2, rtol=2e-2)


def test_rollout_injection(cond: _Condition):
    n_views = 2
    prev_pred = torch.full((1, 1, n_views * 4, 1, 1), 99.0, device=cond.gt_frames.device, dtype=_MODEL_DTYPE)
    cur_chunk = _slice_condition_window(cond, n_views=n_views, start_frame=2, end_frame=6, total_t_per_view=6)
    out, replaced = _inject_rollout_prediction_into_condition(cur_chunk, prev_pred, n_views=n_views, overlap=2)
    assert replaced is True
    out_v = out.gt_frames.view(1, 1, n_views, 4, 1, 1)
    assert torch.allclose(out_v[:, :, :, :2], torch.full_like(out_v[:, :, :, :2], 99.0))


def test_ar_path(
    model: MultiviewControlVideo2WorldModelRectifiedFlow,
    cond: _Condition,
    data_batch: dict,
    gt: torch.Tensor,
):
    assert _get_num_views(data_batch, cond, model.state_t) == 2
    assert _get_self_forcing_overlap(model, model.state_t) == 2
    assert _should_use_autoregressive_self_forcing(model, data_batch, cond, gt) is True

    chunk_x0 = _slice_multiview_window(gt, n_views=2, start_frame=0, end_frame=4)
    out_chunk, loss_chunk = _run_single_chunk_training_step(model, chunk_x0, _slice_condition_window(cond, 2, 0, 4, 6))
    assert torch.isfinite(loss_chunk)
    assert out_chunk["model_pred"].shape == chunk_x0.shape

    out_batch, loss = _training_step_multiview_autoregressive(model, gt, cond, iteration=10, data_batch=data_batch)
    assert torch.isfinite(loss)
    assert float(out_batch["self_forcing_applied"].item()) == 1.0
    assert float(out_batch["self_forcing_num_chunks"].item()) == 2.0

    x0_rollout = _predict_rollout_x0_full(
        model=model,
        condition_chunk=_slice_condition_window(cond, 2, 0, 4, 6),
        epsilon_full_B_C_T_H_W=torch.ones((1, 1, 8, 1, 1), device=gt.device, dtype=_MODEL_DTYPE),
        xt_full_B_C_T_H_W=torch.zeros((1, 1, 8, 1, 1), device=gt.device, dtype=_MODEL_DTYPE),
        timesteps_full_B_T=torch.zeros((1, 1), device=gt.device, dtype=_MODEL_DTYPE),
    )
    assert x0_rollout.shape == (1, 1, 8, 1, 1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This local script expects CUDA. Please run in your GPU environment.")

    _patch_single_process_context_parallel()
    model = _build_real_model()
    gt, cond, data_batch = _make_test_tensors(device=torch.device("cuda"))

    test_probability_schedule(model)
    test_synced_decision_and_apply_gate(model, cond)
    test_slice_helpers(cond)
    test_apply_self_forcing(model, cond)
    test_rollout_injection(cond)
    test_ar_path(model, cond, data_batch, gt)

    print("self_forcing_functions_local_test.py: all checks passed")


if __name__ == "__main__":
    main()
