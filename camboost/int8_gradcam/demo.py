"""
Demo workflows for comparing FP32 and INT8 GradCAM heatmaps.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image

from .core import Int8GradCAM


DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"


@dataclass
class DemoResult:
    name: str
    target_class: int
    top5_fp32: List[Tuple[int, float]]
    top5_int8: List[Tuple[int, float]]
    heatmap_fp32: np.ndarray
    heatmap_int8: np.ndarray
    metrics: Dict[str, float]
    save_path: str


def load_demo_image(image_source: str = DEFAULT_IMAGE_URL) -> Tuple[np.ndarray, torch.Tensor]:
    """Load an RGB image and preprocess it for classification models."""
    if image_source.startswith("http"):
        response = requests.get(image_source, stream=True, timeout=10)
        response.raise_for_status()
        pil_image = Image.open(response.raw).convert("RGB")
    else:
        pil_image = Image.open(image_source).convert("RGB")

    image = np.array(pil_image)
    original_image = image.copy()
    image = np.float32(image) / 255.0
    input_tensor = preprocess_image(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return original_image, input_tensor


def _format_topk(topk: torch.return_types.topk) -> List[Tuple[int, float]]:
    indices = topk.indices[0].tolist()
    values = topk.values[0].tolist()
    return list(zip(indices, values))


def render_comparison(
    name: str,
    heatmap_fp32: np.ndarray,
    heatmap_int8: np.ndarray,
    original_image: np.ndarray,
    diff: np.ndarray,
    save_path: str,
) -> None:
    """Render and persist visual comparison artifacts."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    im1 = axes[0, 0].imshow(heatmap_fp32, cmap="jet", vmin=0, vmax=1)
    axes[0, 0].set_title("FP32 GradCAM", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(heatmap_int8, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title("INT8 GradCAM", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(0.1, diff.max()))
    axes[0, 2].set_title(f"Difference\n(MAE={diff.mean():.4f})", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(original_image)
    axes[1, 0].imshow(heatmap_fp32, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title("FP32 Overlay", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(original_image)
    axes[1, 1].imshow(heatmap_int8, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("INT8 Overlay", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(original_image)
    axes[1, 2].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle(f"{name} INT8 vs FP32 GradCAM", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_demo_summary(result: DemoResult) -> None:
    """Pretty-print demo results."""
    print("\n" + "=" * 70)
    print(f"{result.name} GradCAM Demo Summary")
    print("=" * 70)

    print(f"\nTarget class: {result.target_class}")

    print("\nFP32 Top-5 predictions:")
    for idx, (cls_id, score) in enumerate(result.top5_fp32, start=1):
        print(f"   {idx}. Class {cls_id}: {score:.4f}")

    print("\nINT8 Top-5 predictions:")
    for idx, (cls_id, score) in enumerate(result.top5_int8, start=1):
        print(f"   {idx}. Class {cls_id}: {score:.4f}")

    print("\nQuality metrics (lower is better unless stated otherwise):")
    print(f"   - MAE:      {result.metrics['mae']:.6f}")
    print(f"   - Max Err:  {result.metrics['max_error']:.6f}")
    print(f"   - RMSE:     {result.metrics['rmse']:.6f}")
    print(f"   - Corr:     {result.metrics['corr']:.6f}")

    print(f"\nVisualization saved to: {result.save_path}")
    print("\n" + "=" * 70 + "\n")


def run_int8_cam_demo(
    demo_name: str,
    model_fp32: nn.Module,
    model_int8: nn.Module,
    target_layer_fp32: nn.Module,
    target_layer_int8: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    device: str = "cpu",
    target_class: Optional[int] = None,
    save_dir: str = ".",
    verbose: bool = True,
) -> DemoResult:
    """Shared implementation for INT8 vs FP32 GradCAM comparisons."""
    device_obj = torch.device(device)
    if device_obj.type != "cpu":
        raise ValueError("Quantized INT8 models currently require CPU execution.")

    model_fp32 = model_fp32.to(device_obj).eval()
    model_int8 = model_int8.to(device_obj).eval()
    input_tensor = input_tensor.clone().to(device_obj)

    with torch.no_grad():
        output_fp32 = model_fp32(input_tensor)
        probs_fp32 = torch.softmax(output_fp32, dim=1)
        top5_fp32 = torch.topk(probs_fp32, k=5)

        output_int8 = model_int8(input_tensor)
        probs_int8 = torch.softmax(output_int8, dim=1)
        top5_int8 = torch.topk(probs_int8, k=5)

    if target_class is None:
        target_class = top5_fp32.indices[0][0].item()

    cam_fp32 = Int8GradCAM(model_fp32, target_layer_fp32, use_int8_conv=False)
    cam_int8 = Int8GradCAM(
        model_int8,
        target_layer_int8,
        model_fp32=model_fp32,
        target_layer_fp32=target_layer_fp32,
        use_int8_conv=True,
    )

    with torch.enable_grad():
        heatmap_fp32 = cam_fp32(input_tensor, target_class=target_class, verbose=verbose)

    with torch.enable_grad():
        heatmap_int8 = cam_int8(input_tensor, target_class=target_class, verbose=verbose)

    heatmap_fp32_np = heatmap_fp32.detach().cpu().numpy()[0]
    heatmap_int8_np = heatmap_int8.detach().cpu().numpy()[0]
    diff = np.abs(heatmap_int8_np - heatmap_fp32_np)

    mse = np.mean((heatmap_int8_np - heatmap_fp32_np) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(diff.mean())
    max_error = float(diff.max())
    if heatmap_fp32_np.std() < 1e-8 or heatmap_int8_np.std() < 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(heatmap_int8_np.flatten(), heatmap_fp32_np.flatten())[0, 1])

    metrics = {
        "mae": mae,
        "max_error": max_error,
        "rmse": rmse,
        "corr": corr,
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{demo_name}_int8_vs_fp32.png")
    render_comparison(demo_name, heatmap_fp32_np, heatmap_int8_np, original_image, diff, save_path)

    result = DemoResult(
        name=demo_name,
        target_class=target_class,
        top5_fp32=_format_topk(top5_fp32),
        top5_int8=_format_topk(top5_int8),
        heatmap_fp32=heatmap_fp32_np,
        heatmap_int8=heatmap_int8_np,
        metrics=metrics,
        save_path=save_path,
    )

    print_demo_summary(result)
    return result


def test_int8_gradcam(
    image_url: str = DEFAULT_IMAGE_URL,
    device: str = "cpu",
    save_dir: str = ".",
    target_class: Optional[int] = None,
    verbose: bool = True,
    image_assets: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
) -> DemoResult:
    """ResNet50 INT8 vs FP32 GradCAM demo."""
    torch.backends.quantized.engine = "fbgemm"
    if image_assets is None:
        original_image, input_tensor = load_demo_image(image_url)
    else:
        original_image, input_tensor = image_assets

    model_fp32 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()

    from torchvision.models.quantization import (
        ResNet50_QuantizedWeights,
        resnet50 as resnet50_quantized,
    )

    model_int8 = resnet50_quantized(
        weights=ResNet50_QuantizedWeights.DEFAULT,
        quantize=True,
    ).eval()

    target_layer_fp32 = model_fp32.layer4[-1]
    target_layer_int8 = model_int8.layer4[-1]

    return run_int8_cam_demo(
        demo_name="resnet50",
        model_fp32=model_fp32,
        model_int8=model_int8,
        target_layer_fp32=target_layer_fp32,
        target_layer_int8=target_layer_int8,
        input_tensor=input_tensor,
        original_image=original_image,
        device=device,
        target_class=target_class,
        save_dir=save_dir,
        verbose=verbose,
    )


def demo_mobilenet_int8_gradcam(
    image_url: str = DEFAULT_IMAGE_URL,
    device: str = "cpu",
    save_dir: str = ".",
    target_class: Optional[int] = None,
    verbose: bool = False,
    image_assets: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
) -> DemoResult:
    """MobileNetV2 INT8 vs FP32 GradCAM demo."""
    torch.backends.quantized.engine = "fbgemm"
    if image_assets is None:
        original_image, input_tensor = load_demo_image(image_url)
    else:
        original_image, input_tensor = image_assets

    model_fp32 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

    from torchvision.models.quantization import (
        MobileNet_V2_QuantizedWeights,
        mobilenet_v2 as mobilenet_v2_quantized,
    )

    model_int8 = mobilenet_v2_quantized(
        weights=MobileNet_V2_QuantizedWeights.DEFAULT,
        quantize=True,
    ).eval()

    target_layer_fp32 = model_fp32.features[-1]
    target_layer_int8 = model_int8.features[-1]

    return run_int8_cam_demo(
        demo_name="mobilenet_v2",
        model_fp32=model_fp32,
        model_int8=model_int8,
        target_layer_fp32=target_layer_fp32,
        target_layer_int8=target_layer_int8,
        input_tensor=input_tensor,
        original_image=original_image,
        device=device,
        target_class=target_class,
        save_dir=save_dir,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="INT8 GradCAM demos")
    parser.add_argument(
        "--demo",
        choices=["resnet", "mobilenet", "all"],
        default="resnet",
        help="Choose which demo to run.",
    )
    parser.add_argument(
        "--image-url",
        default=DEFAULT_IMAGE_URL,
        help="HTTP(S) URL or local path for the demo image.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device (INT8 quantized models currently support CPU only).",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory to store visualization outputs.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Manually specify a target class id instead of using the FP32 top-1 prediction.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose profiling information during GradCAM computation.",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"CUDA device: {device_name}")

    image_assets = load_demo_image(args.image_url)

    if args.demo in {"resnet", "all"}:
        test_int8_gradcam(
            image_url=args.image_url,
            device=args.device,
            save_dir=args.save_dir,
            target_class=args.target_class,
            verbose=args.verbose,
            image_assets=image_assets,
        )

    if args.demo in {"mobilenet", "all"}:
        demo_mobilenet_int8_gradcam(
            image_url=args.image_url,
            device=args.device,
            save_dir=args.save_dir,
            target_class=args.target_class,
            verbose=args.verbose,
            image_assets=image_assets,
        )


if __name__ == "__main__":
    main()
