#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 INT8 GradCAM 实现
包括 INT8 输入数据 + INT8 卷积 + INT8 CAM 计算
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pytorch_grad_cam.utils.image import preprocess_image
import requests
from PIL import Image
import matplotlib.pyplot as plt
import time


# ==============================
# INT8 量化工具函数
# ==============================
# def quantize_to_int8(x):
#     """量化 tensor 到 INT8"""
#     x_min, x_max = x.min(), x.max()
#     scale = (x_max - x_min) / 255.0 if (x_max - x_min) > 0 else 1e-8
#     zp = -x_min / scale
#     q = torch.clamp(torch.round(x / scale + zp), 0, 255).to(torch.uint8)
#     q = (q.to(torch.int16) - 128).to(torch.int8)
#     return q, scale, zp - 128


# def dequantize_from_int8(q, scale, zp):
#     """从 INT8 反量化"""
#     return (q.float() - zp) * scale
def quantize_to_int8(x, symmetric=True):
    """
    将 tensor 量化为 INT8
    Args:
        x: FP32 tensor
        symmetric: 是否使用对称量化（默认 True）
    Returns:
        q: INT8 tensor
        scale: scale
        zp: zero point (在 INT8 表达域下)
    """
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

def quantize_gradient(grad, clip_ratio=0.02):
    """
    对梯度进行稳健量化：去除极值并使用对称量化
    """
    g = grad.detach().float()
    abs_vals = g.abs().flatten()
    # 去掉顶部 clip_ratio 的极值
    topk = int(len(abs_vals) * (1 - clip_ratio))
    max_val = abs_vals.kthvalue(topk).values
    max_val = max(max_val, torch.tensor(1e-6, device=g.device))
    scale = max_val / 127.0
    q = torch.clamp(torch.round(g / scale), -128, 127).to(torch.int8)
    return q, scale, 0


def dequantize_from_int8(q, scale, zp):
    """反量化"""
    return (q.float() - zp) * scale



# def global_avg_pool_int8(x_int8, scale, zp):
#     """INT8 全局平均池化"""
#     x_int32 = (x_int8.to(torch.int32) - int(zp))
#     # PyTorch mean() 不支持 INT32，需要转换为 float 计算
#     mean_float = x_int32.float().mean(dim=(2, 3), keepdim=True)
#     # 转回 INT32 (保留精度)
#     mean_int32 = mean_float.to(torch.int32)
#     mean_scale = scale
#     return mean_int32, mean_scale
def global_avg_pool_int8(x_int8, scale, zp):
    H, W = x_int8.shape[2], x_int8.shape[3]
    x_int32 = (x_int8.to(torch.int32) - int(zp))
    sum_int32 = x_int32.sum(dim=(2, 3), keepdim=True)
    mean_scale = scale / (H * W)
    return sum_int32, mean_scale



def weighted_sum_int8(weights_int32, w_scale, act_int8, a_scale, zp_a):
    """INT8 加权求和"""
    a_int32 = (act_int8.to(torch.int32) - int(zp_a))
    cam_int32 = (weights_int32 * a_int32).sum(dim=1, keepdim=True)
    cam_scale = w_scale * a_scale
    return cam_int32, cam_scale
# def weighted_sum_int8(weights_int32, w_scale, act_int8, a_scale, zp_a):
#     """
#     INT8 加权求和 (使用 INT64 累加避免溢出)
#     """
#     a_int32 = (act_int8.to(torch.int32) - int(zp_a))
#     cam_int64 = (weights_int32.to(torch.int64) * a_int32.to(torch.int64)).sum(dim=1, keepdim=True)
#     cam_int32 = cam_int64.clamp(-2**31, 2**31 - 1).to(torch.int32)
#     cam_scale = w_scale * a_scale
#     return cam_int32, cam_scale



def relu_int32(x):
    """INT32 域的 ReLU"""
    return torch.clamp(x, min=0)


def normalize_and_upsample(cam_int32, cam_scale, target_size):
    """归一化并上采样"""
    cam_fp32 = cam_int32.float() * cam_scale
    cam_fp32 = cam_fp32 - cam_fp32.amin(dim=(2, 3), keepdim=True)
    cam_fp32 = cam_fp32 / (cam_fp32.amax(dim=(2, 3), keepdim=True) + 1e-8)
    cam_fp32 = torch.nn.functional.interpolate(
        cam_fp32, size=target_size, mode="bilinear", align_corners=False
    )
    return cam_fp32


# ==============================
# 量化输入图像
# ==============================
def quantize_input_image(input_tensor):
    """
    将 FP32 输入图像量化为 INT8

    Args:
        input_tensor: FP32 tensor [N, C, H, W]

    Returns:
        input_int8: INT8 tensor
        input_scale: 量化scale
        input_zp: 量化zero point
    """
    input_int8, input_scale, input_zp = quantize_to_int8(input_tensor)
    return input_int8, input_scale, input_zp


# ==============================
# 量化模型（静态量化）
# ==============================
def quantize_model_int8_static(model_fp32, calibration_data):
    """
    将 FP32 模型量化为 INT8（静态量化）

    使用 PyTorch 的静态量化，支持真正的 INT8 输入

    Args:
        model_fp32: FP32 模型
        calibration_data: 校准数据

    Returns:
        model_int8: INT8 量化模型
    """
    print("\n   [量化模型] 使用静态 INT8 量化...")

    # 设置量化后端（CPU）
    torch.backends.quantized.engine = 'fbgemm'

    # 1. 准备模型（插入量化/反量化节点）
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    # 2. 校准（收集激活值统计信息）
    print("   - 校准中...")
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)

    # 3. 转换为量化模型
    model_int8 = torch.quantization.convert(model_prepared, inplace=False)

    print("   ✓ 模型量化完成 (静态量化 → 真正的 INT8 输入/输出)")
    return model_int8


# ==============================
# Int8GradCAM 主类
# ==============================
class Int8GradCAM:
    """
    完整的 INT8 GradCAM 实现

    支持两种模式：
    1. FP32 模型 + INT8 CAM 计算
    2. INT8 模型 + INT8 CAM 计算（完整 INT8 流程）
    """

    def __init__(self, model, target_layer, model_fp32=None, target_layer_fp32=None, use_int8_conv=False):
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

        # INT8 模型需要 FP32 模型来计算梯度
        self.model_fp32 = model_fp32 if model_fp32 is not None else model
        self.target_layer_fp32 = target_layer_fp32 if target_layer_fp32 is not None else target_layer
        self.activations_fp32 = None

        # 注册 hook
        target_layer.register_forward_hook(self.forward_hook)
        if model_fp32 is not None:
            target_layer_fp32.register_forward_hook(self.forward_hook_fp32)

    def forward_hook(self, module, input_data, output):
        """保存激活值"""
        self.activations = output

    def forward_hook_fp32(self, module, input_data, output):
        """保存 FP32 激活值（用于梯度计算）"""
        self.activations_fp32 = output

    def __call__(self, input_tensor, target_class, verbose=False):
        """
        计算 GradCAM

        Args:
            input_tensor: 输入 tensor (FP32)
            target_class: 目标类别
            verbose: 是否输出详细信息

        Returns:
            cam: GradCAM 热力图
        """
        self.model.eval()
        batch_size = input_tensor.shape[0]

        # ========== Step 1: INT8 输入（使用 INT8 卷积时）==========
        if self.use_int8_conv and verbose:
            print("\n   [Step 1] 准备 INT8 输入...")
            print(f"      注意: 静态量化模型会自动在第一层量化输入")
            print(f"      输入范围: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")

        # 直接使用 FP32 输入，静态量化模型会在内部自动量化
        input_for_model = input_tensor

        # ========== Step 2: 前向传播（INT8 或 FP32 卷积）==========
        if verbose:
            conv_type = "INT8" if self.use_int8_conv else "FP32"
            print(f"\n   [Step 2] 前向传播 ({conv_type} 卷积)...")

        t0 = time.time()
        output = self.model(input_for_model)
        forward_time = (time.time() - t0) * 1000

        if verbose:
            print(f"      前向传播耗时: {forward_time:.2f} ms")

        scores = output[range(batch_size), target_class]

        # ========== Step 3: 梯度计算（使用 FP32 模型）==========
        if verbose:
            print(f"\n   [Step 3] 计算梯度（使用 FP32 模型）...")

        t0 = time.time()

        # ==================================================
        # 使用 INT8→FP32 机制计算梯度
        # ==================================================
        if self.model_fp32 != self.model:
            # INT8 模型：反量化 INT8 激活，通过 FP32 后续层，对激活求梯度
            if verbose:
                print(f"      (INT8 模型，从反量化激活继续 FP32 传播)")

            # 1. 反量化 INT8 激活
            act_int8 = self.activations
            if act_int8.dtype in [torch.quint8, torch.qint8, torch.qint32]:
                if verbose:
                    print(f"      反量化 INT8 激活...")
                act_fp32 = act_int8.dequantize()
            else:
                act_fp32 = act_int8.float()

            # 设置需要梯度
            act_fp32 = act_fp32.detach().requires_grad_(True)

            # 2. 从激活值继续通过后续的 FP32 层
            # 获取目标层之后的层
            device = next(self.model_fp32.parameters()).device
            act_fp32 = act_fp32.to(device)

            with torch.enable_grad():
                # 通过 avgpool 和 fc
                x = self.model_fp32.avgpool(act_fp32)
                x = torch.flatten(x, 1)
                output_fp32 = self.model_fp32.fc(x)

                scores_fp32 = output_fp32[range(batch_size), target_class]

                # 对激活值求梯度
                gradients = torch.autograd.grad(
                    outputs=scores_fp32.sum(),
                    inputs=act_fp32,
                    retain_graph=False,
                    create_graph=False
                )[0].detach()

        else:
            # FP32 模型情况：直接计算梯度
            gradients = torch.autograd.grad(
                outputs=scores.sum(),
                inputs=self.activations,
                retain_graph=False,
                create_graph=False
            )[0].detach()

        grad_time = (time.time() - t0) * 1000

        # 打印调试信息
        if verbose:
            print(f"      梯度计算耗时: {grad_time:.2f} ms")
            print(f"      梯度范围: [{gradients.min():.6f}, {gradients.max():.6f}]")

        # ========== Step 4: INT8 CAM 计算 ==========
        if verbose:
            print(f"\n   [Step 4] INT8 CAM 计算...")

        t0 = time.time()

        # 确保激活和梯度在同一设备上
        if self.activations.device != gradients.device:
            gradients = gradients.to(self.activations.device)

        # 如果激活值是量化 tensor，先反量化
        activations = self.activations
        if activations.dtype in [torch.quint8, torch.qint8, torch.qint32]:
            if verbose:
                print(f"      检测到量化激活，反量化中...")
            activations = activations.dequantize()


        # 量化激活
        act_int8, act_scale, act_zp = quantize_to_int8(activations)

        # 量化梯度
        # grad_int8, grad_scale, grad_zp = quantize_to_int8(gradients)
        grad_int8, grad_scale, grad_zp = quantize_gradient(gradients)


        # Global Average Pooling (INT8→INT32)
        weights_int32, w_scale = global_avg_pool_int8(grad_int8, grad_scale, grad_zp)

        # 加权求和
        cam_int32, cam_scale = weighted_sum_int8(weights_int32, w_scale, act_int8, act_scale, act_zp)

        # ReLU + Normalize + Upsample
        cam_int32 = relu_int32(cam_int32)
        cam_fp32 = normalize_and_upsample(cam_int32, cam_scale, target_size=input_tensor.shape[-2:])

        cam_time = (time.time() - t0) * 1000

        if verbose:
            print(f"      CAM 计算耗时: {cam_time:.2f} ms")
            print(f"      激活量化: scale={act_scale:.6f}, zp={act_zp:.1f}")
            print(f"      梯度量化: scale={grad_scale:.6f}, zp={grad_zp:.1f}")

        return cam_fp32.squeeze(1)


# ==============================
# 测试函数
# ==============================
def test_int8_gradcam():
    print("\n" + "="*70)
    print("INT8 vs FP32 GradCAM 对比测试 (均在 CPU 上运行)")
    print("="*70)

    # 1. 加载图像
    print("\n1. 加载测试图像...")
    image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
    image = np.array(Image.open(requests.get(image_url, stream=True).raw))
    original_image = image.copy()
    image = np.float32(image) / 255.0
    input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    print(f"   图像大小: {original_image.shape}")

    # 2. 加载 FP32 模型
    print("\n2. 加载 ResNet50 模型 (FP32)...")
    model_fp32 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    print("   ✓ FP32 模型加载完成")

    # 3. 加载量化版 ResNet50
    print("\n3. 加载 INT8 量化模型...")
    print("   使用 torchvision 预训练的量化 ResNet50")

    # 加载 torchvision 提供的量化 ResNet50
    from torchvision.models.quantization import resnet50 as resnet50_quantized
    from torchvision.models.quantization import ResNet50_QuantizedWeights

    model_int8 = resnet50_quantized(
        weights=ResNet50_QuantizedWeights.DEFAULT,
        quantize=True
    ).cpu().eval()

    target_layer_int8 = model_int8.layer4[-1]
    target_layer_fp32 = model_fp32.layer4[-1]

    print("   ✓ INT8 量化模型加载完成")

    # FP32 和 INT8 都在 CPU 上运行（公平对比）
    print(f"   ✓ 使用设备: CPU (公平对比)")
    model_fp32 = model_fp32.cpu()
    input_tensor_cpu = input_tensor.cpu()

    # 4. 检查两个模型的预测结果
    print("\n" + "="*70)
    print("4. 检查模型预测（诊断差异原因）")
    print("="*70)

    with torch.no_grad():
        # FP32 预测
        output_fp32 = model_fp32(input_tensor_cpu)
        probs_fp32 = torch.softmax(output_fp32, dim=1)
        top5_fp32 = torch.topk(probs_fp32, 5)

        print("\n   FP32 模型 Top-5 预测:")
        for i in range(5):
            print(f"      {i+1}. Class {top5_fp32.indices[0][i].item()}: {top5_fp32.values[0][i].item():.4f}")

        # INT8 预测
        output_int8 = model_int8(input_tensor_cpu)
        probs_int8 = torch.softmax(output_int8, dim=1)
        top5_int8 = torch.topk(probs_int8, 5)

        print("\n   INT8 模型 Top-5 预测:")
        for i in range(5):
            print(f"      {i+1}. Class {top5_int8.indices[0][i].item()}: {top5_int8.values[0][i].item():.4f}")

        # 检查目标类别 246 的置信度
        print(f"\n   目标类别 246 的置信度:")
        print(f"      FP32: {probs_fp32[0, 246].item():.6f}")
        print(f"      INT8: {probs_int8[0, 246].item():.6f}")

        # 使用 FP32 模型的 Top-1 预测作为目标类别
        target_class = top5_fp32.indices[0][0].item()
        print(f"\n   ✓ 使用 FP32 Top-1 预测 (Class {target_class}) 作为目标类别")

    # 5. 测试 FP32 GradCAM (基准 - CPU)
    print("\n" + "="*70)
    print(f"5. 测试 FP32 GradCAM (Class {target_class} - CPU)")
    print("="*70)

    cam_fp32 = Int8GradCAM(
        model_fp32,
        model_fp32.layer4[-1],
        use_int8_conv=False
    )

    with torch.enable_grad():
        heatmap_fp32 = cam_fp32(input_tensor_cpu, target_class=target_class, verbose=True)

    heatmap_fp32_np = heatmap_fp32.detach().cpu().numpy()[0]
    print(f"\n   结果: 范围=[{heatmap_fp32_np.min():.3f}, {heatmap_fp32_np.max():.3f}], "
          f"均值={heatmap_fp32_np.mean():.3f}")

    # 6. 测试 INT8 卷积 + INT8 CAM (完整 INT8 - CPU)
    print("\n" + "="*70)
    print(f"6. 测试 INT8 卷积 + INT8 CAM (Class {target_class} - CPU)")
    print("="*70)

    cam_int8_conv = Int8GradCAM(
        model_int8,
        target_layer_int8,
        model_fp32=model_fp32,
        target_layer_fp32=target_layer_fp32,
        use_int8_conv=True
    )

    with torch.enable_grad():
        heatmap_int8 = cam_int8_conv(input_tensor_cpu, target_class=target_class, verbose=True)

    heatmap_int8_np = heatmap_int8.detach().cpu().numpy()[0]
    print(f"\n   结果: 范围=[{heatmap_int8_np.min():.3f}, {heatmap_int8_np.max():.3f}], "
          f"均值={heatmap_int8_np.mean():.3f}")

    # 7. 质量对比
    print("\n" + "="*70)
    print("7. 质量对比 (INT8 vs FP32)")
    print("="*70)

    diff = np.abs(heatmap_int8_np - heatmap_fp32_np)
    corr = np.corrcoef(heatmap_int8_np.flatten(), heatmap_fp32_np.flatten())[0, 1]
    mse = ((heatmap_int8_np - heatmap_fp32_np) ** 2).mean()
    rmse = np.sqrt(mse)

    print(f"   平均绝对误差 (MAE): {diff.mean():.6f}")
    print(f"   最大误差 (Max):     {diff.max():.6f}")
    print(f"   均方根误差 (RMSE):  {rmse:.6f}")
    print(f"   相关系数 (Corr):    {corr:.6f}")

    # 8. 可视化对比
    print("\n8. 生成可视化对比...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 第一行：热力图对比
    # FP32 热力图
    im1 = axes[0, 0].imshow(heatmap_fp32_np, cmap='jet', vmin=0, vmax=1)
    axes[0, 0].set_title('FP32 GradCAM', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # INT8 热力图
    im2 = axes[0, 1].imshow(heatmap_int8_np, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('INT8 GradCAM', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 差异图
    im3 = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
    axes[0, 2].set_title(f'Difference\n(MAE={diff.mean():.4f})', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 第二行：叠加图对比
    # FP32 叠加
    axes[1, 0].imshow(original_image)
    axes[1, 0].imshow(heatmap_fp32_np, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title('FP32 Overlay', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # INT8 叠加
    axes[1, 1].imshow(original_image)
    axes[1, 1].imshow(heatmap_int8_np, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title('INT8 Overlay', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

    # 原图
    axes[1, 2].imshow(original_image)
    axes[1, 2].set_title('Original Image', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = "./int8_vs_fp32_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 可视化对比已保存: {save_path}")
    plt.close()

    # 9. 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print("✅ INT8 vs FP32 GradCAM 对比完成！")
    print(f"\n   质量指标:")
    print(f"   - 相关系数: {corr:.6f} (越接近1越好)")
    print(f"   - 平均误差: {diff.mean():.6f} (越小越好)")
    print(f"   - RMSE:     {rmse:.6f} (越小越好)")
    print(f"\n   性能对比 (均在 CPU 上运行):")
    print(f"   - FP32: 标准 FP32 计算")
    print(f"   - INT8: 量化 INT8 计算 (fbgemm 后端)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    test_int8_gradcam()
