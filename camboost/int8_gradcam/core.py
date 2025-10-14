"""
Core INT8 GradCAM classes and quantization helpers.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn

from .utils import Int8CamUtils


def quantize_model_int8_static(model_fp32: nn.Module, calibration_data):
    """
    将 FP32 模型量化为 INT8（静态量化）

    使用 PyTorch 的静态量化，支持真正的 INT8 输入
    """
    print("\n   [量化模型] 使用静态 INT8 量化...")

    torch.backends.quantized.engine = "fbgemm"

    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    print("   - 校准中...")
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)

    model_int8 = torch.quantization.convert(model_prepared, inplace=False)

    print("   ✓ 模型量化完成 (静态量化 → 真正的 INT8 输入/输出)")
    return model_int8


class Int8GradCAM:
    """
    完整的 INT8 GradCAM 实现

    支持两种模式：
    1. FP32 模型 + INT8 CAM 计算
    2. INT8 模型 + INT8 CAM 计算（完整 INT8 流程）
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        model_fp32: Optional[nn.Module] = None,
        target_layer_fp32: Optional[nn.Module] = None,
        use_int8_conv: bool = False,
    ):
        """
        Args:
            model: 推理模型（可以是 FP32 或 INT8）
            target_layer: 目标层
            model_fp32: FP32 模型（用于梯度计算，INT8 模型不支持 autograd）
            target_layer_fp32: FP32 目标层
            use_int8_conv: 是否使用 INT8 卷积
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.use_int8_conv = use_int8_conv

        self.model_fp32 = model_fp32 if model_fp32 is not None else model
        self.target_layer_fp32 = target_layer_fp32 if target_layer_fp32 is not None else target_layer
        self.activations_fp32 = None

        target_layer.register_forward_hook(self.forward_hook)
        if model_fp32 is not None:
            target_layer_fp32.register_forward_hook(self.forward_hook_fp32)

    def forward_hook(self, module, input_data, output):
        self.activations = output

    def forward_hook_fp32(self, module, input_data, output):
        self.activations_fp32 = output

    def __call__(self, input_tensor: torch.Tensor, target_class: int, verbose: bool = False):
        self.model.eval()
        batch_size = input_tensor.shape[0]

        if self.use_int8_conv and verbose:
            print("\n   [Step 1] 准备 INT8 输入...")
            print("      注意: 静态量化模型会自动在第一层量化输入")
            print(f"      输入范围: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")

        input_for_model = input_tensor

        if verbose:
            conv_type = "INT8" if self.use_int8_conv else "FP32"
            print(f"\n   [Step 2] 前向传播 ({conv_type} 卷积)...")

        t0 = time.time()
        output = self.model(input_for_model)
        forward_time = (time.time() - t0) * 1000

        if verbose:
            print(f"      前向传播耗时: {forward_time:.2f} ms")

        scores = output[range(batch_size), target_class]

        if verbose:
            print("\n   [Step 3] 计算梯度（使用 FP32 模型）...")

        t0 = time.time()

        if self.model_fp32 != self.model:
            if verbose:
                print("      (INT8 模型，从反量化激活继续 FP32 传播)")

            act_int8 = self.activations
            if act_int8.dtype in [torch.quint8, torch.qint8, torch.qint32]:
                if verbose:
                    print("      反量化 INT8 激活...")
                act_fp32 = act_int8.dequantize()
            else:
                act_fp32 = act_int8.float()

            act_fp32 = act_fp32.detach().requires_grad_(True)
            device = next(self.model_fp32.parameters()).device
            act_fp32 = act_fp32.to(device)

            with torch.enable_grad():
                x = self.model_fp32.avgpool(act_fp32)
                x = torch.flatten(x, 1)
                output_fp32 = self.model_fp32.fc(x)

                scores_fp32 = output_fp32[range(batch_size), target_class]

                gradients = torch.autograd.grad(
                    outputs=scores_fp32.sum(),
                    inputs=act_fp32,
                    retain_graph=False,
                    create_graph=False,
                )[0].detach()

        else:
            gradients = torch.autograd.grad(
                outputs=scores.sum(),
                inputs=self.activations,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

        grad_time = (time.time() - t0) * 1000

        if verbose:
            print(f"      梯度计算耗时: {grad_time:.2f} ms")
            print(f"      梯度范围: [{gradients.min():.6f}, {gradients.max():.6f}]")

        if verbose:
            print("\n   [Step 4] INT8 CAM 计算...")

        t0 = time.time()

        if self.activations.device != gradients.device:
            gradients = gradients.to(self.activations.device)

        activations = self.activations
        if activations.dtype in [torch.quint8, torch.qint8, torch.qint32]:
            if verbose:
                print("      检测到量化激活，反量化中...")
            activations = activations.dequantize()

        act_int8, act_scale, act_zp = Int8CamUtils.quantize_to_int8(activations)
        grad_int8, grad_scale, grad_zp = Int8CamUtils.quantize_gradient(gradients)
        weights_int32, w_scale = Int8CamUtils.global_avg_pool_int8(grad_int8, grad_scale, grad_zp)
        cam_int32, cam_scale = Int8CamUtils.weighted_sum_int8(weights_int32, w_scale, act_int8, act_scale, act_zp)
        cam_int32 = Int8CamUtils.relu_int32(cam_int32)
        cam_fp32 = Int8CamUtils.normalize_and_upsample(cam_int32, cam_scale, target_size=input_tensor.shape[-2:])

        cam_time = (time.time() - t0) * 1000

        if verbose:
            print(f"      CAM 计算耗时: {cam_time:.2f} ms")
            print(f"      激活量化: scale={act_scale:.6f}, zp={act_zp:.1f}")
            print(f"      梯度量化: scale={grad_scale:.6f}, zp={grad_zp:.1f}")

        return cam_fp32.squeeze(1)
