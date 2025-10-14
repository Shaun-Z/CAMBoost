"""
Quantization and CAM math helpers for INT8 GradCAM.
"""

from __future__ import annotations

import torch


class Int8CamUtils:
    """Helper utilities for INT8 quantization and GradCAM math."""

    @staticmethod
    def quantize_to_int8(x: torch.Tensor, symmetric: bool = True):
        """Quantize a tensor to INT8."""
        x_min, x_max = x.min(), x.max()
        if symmetric:
            max_abs = max(abs(x_min.item()), abs(x_max.item()))
            scale = max_abs / 127.0 if max_abs > 0 else 1e-8
            zp = 0
            q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        else:
            scale = (x_max - x_min) / 255.0 if (x_max - x_min) > 0 else 1e-8
            zp = torch.round(-x_min / scale).clamp(0, 255)
            q = torch.clamp(torch.round(x / scale + zp), 0, 255).to(torch.uint8)
            q = (q.to(torch.int16) - 128).to(torch.int8)
            zp = zp - 128
        return q, scale, zp

    @staticmethod
    def quantize_gradient(grad: torch.Tensor, clip_ratio: float = 0.02):
        """Robust gradient quantization with clipping of extreme values."""
        g = grad.detach().float()
        abs_vals = g.abs().flatten()
        topk = int(len(abs_vals) * (1 - clip_ratio))
        if topk <= 0:
            max_val = abs_vals.max()
        else:
            max_val = abs_vals.kthvalue(topk).values
        max_val = max(max_val, torch.tensor(1e-6, device=g.device))
        scale = max_val / 127.0
        q = torch.clamp(torch.round(g / scale), -128, 127).to(torch.int8)
        return q, scale, 0

    @staticmethod
    def dequantize_from_int8(q: torch.Tensor, scale: float, zp: float):
        """Dequantize INT8 back to FP32."""
        return (q.float() - zp) * scale

    @staticmethod
    def global_avg_pool_int8(x_int8: torch.Tensor, scale: float, zp: float):
        """INT8 global average pooling implemented with INT32 accumulation."""
        height, width = x_int8.shape[2], x_int8.shape[3]
        x_int32 = (x_int8.to(torch.int32) - int(zp))
        sum_int32 = x_int32.sum(dim=(2, 3), keepdim=True)
        mean_scale = scale / (height * width)
        return sum_int32, mean_scale

    @staticmethod
    def weighted_sum_int8(
        weights_int32: torch.Tensor,
        w_scale: float,
        act_int8: torch.Tensor,
        a_scale: float,
        zp_a: float,
    ):
        """INT8 weighted sum for CAM computation."""
        a_int32 = (act_int8.to(torch.int32) - int(zp_a))
        cam_int32 = (weights_int32 * a_int32).sum(dim=1, keepdim=True)
        cam_scale = w_scale * a_scale
        return cam_int32, cam_scale

    @staticmethod
    def relu_int32(x: torch.Tensor):
        """ReLU in INT32 domain."""
        return torch.clamp(x, min=0)

    @staticmethod
    def normalize_and_upsample(cam_int32: torch.Tensor, cam_scale: float, target_size):
        """Normalize CAM to [0, 1] and upsample to target spatial size."""
        cam_fp32 = cam_int32.float() * cam_scale
        cam_fp32 = cam_fp32 - cam_fp32.amin(dim=(2, 3), keepdim=True)
        cam_fp32 = cam_fp32 / (cam_fp32.amax(dim=(2, 3), keepdim=True) + 1e-8)
        cam_fp32 = torch.nn.functional.interpolate(
            cam_fp32, size=target_size, mode="bilinear", align_corners=False
        )
        return cam_fp32

    @staticmethod
    def quantize_input_image(input_tensor: torch.Tensor):
        """Quantize an FP32 input image."""
        return Int8CamUtils.quantize_to_int8(input_tensor)
