#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare FP32 and INT8 models on Tiny-ImageNet and export GradCAM visuals.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
)
from camboost.int8_gradcam import Int8GradCAM


warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-ImageNet FP32 vs INT8 comparison")
    parser.add_argument(
        "--data-root",
        default="data/tiny-imagenet",
        help="Path to the Tiny-ImageNet root directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to store evaluation report and GradCAM images.",
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=1000,
        help="Number of sample images to export GradCAM overlays for.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Limit evaluation to the first N images (per run) for quicker experiments.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle evaluation dataset ordering.",
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "resnet101"],
        default="resnet50",
        help="Backbone architecture to evaluate (supported: resnet18/resnet50/resnet101).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for subset selection and shuffling.",
    )
    return parser.parse_args()


def load_wnids(root: Path) -> List[str]:
    return (root / "wnids.txt").read_text().strip().splitlines()


def build_imagenet_mapping(root: Path, wnids: List[str]) -> Tuple[List[int], Dict[str, int]]:
    idx2wnid: Dict[int, str] = {}
    with open(root / "imagenet_class_index.json", "r") as f:
        class_index = json.load(f)
        for k, (wnid, _) in class_index.items():
            idx2wnid[int(k)] = wnid

    wnid_to_imagenet = {}
    for w in wnids:
        matches = [idx for idx, mapped in idx2wnid.items() if mapped == w]
        if matches:
            wnid_to_imagenet[w] = matches[0]
    missing = [w for w in wnids if w not in wnid_to_imagenet]
    if missing:
        raise RuntimeError(f"Missing ImageNet mapping for wnids: {missing}")

    imagenet_indices = [wnid_to_imagenet[w] for w in wnids]
    wnid_order = {wnid: i for i, wnid in enumerate(wnids)}
    return imagenet_indices, wnid_order


def build_dataloader(
    root: Path,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    shuffle: bool,
    generator: Optional[torch.Generator],
):
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    data_dir = root / "train"
    base_dataset = dsets.ImageFolder(str(data_dir), transform=transform)
    classes = base_dataset.classes

    if max_images is not None:
        max_images = max(1, min(max_images, len(base_dataset)))
        if shuffle:
            if generator is None:
                indices = torch.randperm(len(base_dataset))[:max_images].tolist()
            else:
                indices = torch.randperm(len(base_dataset), generator=generator)[:max_images].tolist()
        else:
            indices = list(range(max_images))
        dataset = Subset(base_dataset, indices)
    else:
        dataset = base_dataset

    if shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataset, loader, classes


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    imagenet_indices: List[int],
    folder_to_tiny: Dict[int, int],
) -> Dict[str, float]:
    model.eval()
    param = next(model.parameters(), None)
    if param is not None:
        device = param.device
    else:
        buffer = next(model.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")
    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            remapped = torch.tensor([folder_to_tiny[int(t)] for t in targets], device=device)
            images = images.to(device)
            logits = model(images)
            logits_200 = logits[:, imagenet_indices]

            pred1 = logits_200.argmax(dim=1)
            top1 += (pred1 == remapped).sum().item()

            _, pred5 = logits_200.topk(5, dim=1)
            top5 += (pred5 == remapped.unsqueeze(1)).any(dim=1).sum().item()

            total += images.size(0)

    return {
        "top1": top1 / total * 100.0,
        "top5": top5 / total * 100.0,
        "samples": float(total),
    }


def load_models(model_name: str):
    torch.backends.quantized.engine = "fbgemm"

    if model_name == "resnet50":
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
        display_name = "ResNet50"

    elif model_name == "resnet18":
        model_fp32 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

        from torchvision.models.quantization import (
            ResNet18_QuantizedWeights,
            resnet18 as resnet18_quantized,
        )

        model_int8 = resnet18_quantized(
            weights=ResNet18_QuantizedWeights.DEFAULT,
            quantize=True,
        ).eval()

        target_layer_fp32 = model_fp32.layer4[-1]
        target_layer_int8 = model_int8.layer4[-1]
        display_name = "ResNet18"

    elif model_name == "resnet101":
        model_fp32 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).eval()

        from torchvision.models.quantization import (
            ResNeXt101_32X8D_QuantizedWeights,
            resnext101_32x8d as resnet101_quantized,
        )

        model_int8 = resnet101_quantized(
            weights=ResNeXt101_32X8D_QuantizedWeights.DEFAULT,
            quantize=True,
        ).eval()

        target_layer_fp32 = model_fp32.layer4[-1]
        target_layer_int8 = model_int8.layer4[-1]
        display_name = "ResNet101"

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model_fp32 = model_fp32.to("cpu")
    model_int8 = model_int8.to("cpu")

    return model_fp32, model_int8, target_layer_fp32, target_layer_int8, display_name


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]
    overlay = (0.6 * image + 0.4 * colored).clip(0, 1)
    return overlay


def save_gradcam_figure(
    original: np.ndarray,
    heatmap_fp32: np.ndarray,
    heatmap_int8: np.ndarray,
    fp32_title: str,
    int8_title: str,
    save_path: Path,
) -> None:
    overlay_fp32 = overlay_heatmap(original, heatmap_fp32)
    overlay_int8 = overlay_heatmap(original, heatmap_int8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_fp32)
    axes[1].set_title(fp32_title)
    axes[1].axis("off")

    axes[2].imshow(overlay_int8)
    axes[2].set_title(int8_title)
    axes[2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_samples(
    dataset: dsets.ImageFolder,
    imagenet_indices: List[int],
    folder_to_tiny: Dict[int, int],
    wnids: List[str],
    model_fp32: torch.nn.Module,
    model_int8: torch.nn.Module,
    target_layer_fp32: torch.nn.Module,
    target_layer_int8: torch.nn.Module,
    results_dir: Path,
    num_samples: int,
    model_label: str,
) -> None:
    model_fp32.eval()
    model_int8.eval()

    cam_fp32 = Int8GradCAM(model_fp32, target_layer_fp32, use_int8_conv=False)
    cam_int8 = Int8GradCAM(
        model_int8,
        target_layer_int8,
        model_fp32=model_fp32,
        target_layer_fp32=target_layer_fp32,
        use_int8_conv=True,
    )

    model_slug = model_label.lower().replace(" ", "_")

    for idx in range(min(num_samples, len(dataset))):
        image_tensor, folder_idx = dataset[idx]
        folder_idx = int(folder_idx)
        tiny_idx = folder_to_tiny[folder_idx]
        imagenet_target = imagenet_indices[tiny_idx]
        wnid = wnids[tiny_idx]

        input_tensor = image_tensor.unsqueeze(0)
        original_np = denormalize(image_tensor)

        with torch.enable_grad():
            heatmap_fp32 = cam_fp32(input_tensor, target_class=imagenet_target, verbose=False)
        with torch.enable_grad():
            heatmap_int8 = cam_int8(input_tensor, target_class=imagenet_target, verbose=False)

        heatmap_fp32_np = heatmap_fp32.detach().cpu().numpy()[0]
        heatmap_int8_np = heatmap_int8.detach().cpu().numpy()[0]

        save_path = results_dir / f"{model_slug}_gradcam_sample_{idx:02d}_{wnid}.png"
        save_gradcam_figure(
            original_np,
            heatmap_fp32_np,
            heatmap_int8_np,
            fp32_title=f"{model_label} FP32 GradCAM",
            int8_title=f"{model_label} INT8 GradCAM",
            save_path=save_path,
        )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    generator: Optional[torch.Generator] = None
    if args.shuffle or args.seed is not None:
        generator = torch.Generator()
        if args.seed is not None:
            generator.manual_seed(args.seed)

    print(f"Loading Tiny-ImageNet from {data_root} ...")
    wnids = load_wnids(data_root)
    imagenet_indices, wnid_order = build_imagenet_mapping(data_root, wnids)
    dataset, loader, classes = build_dataloader(
        data_root,
        args.batch_size,
        args.num_workers,
        args.max_images,
        args.shuffle,
        generator,
    )

    folder_wnids = classes
    folder_to_tiny = {i: wnid_order[w] for i, w in enumerate(folder_wnids)}

    model_fp32, model_int8, target_layer_fp32, target_layer_int8, model_label = load_models(args.model)

    print(f"Loading FP32 {model_label} model (CPU) ...")
    print("Evaluating FP32 model ...")
    metrics_fp32 = evaluate_model(model_fp32, loader, imagenet_indices, folder_to_tiny)
    print(
        f"FP32 {model_label} - Top-1: {metrics_fp32['top1']:.2f}% | Top-5: {metrics_fp32['top5']:.2f}% | Samples: {metrics_fp32['samples']:.0f}"
    )

    print(f"Loading INT8 quantized {model_label} model (CPU) ...")
    print("Evaluating INT8 model ...")
    metrics_int8 = evaluate_model(model_int8, loader, imagenet_indices, folder_to_tiny)
    print(
        f"INT8 {model_label} - Top-1: {metrics_int8['top1']:.2f}% | Top-5: {metrics_int8['top5']:.2f}% | Samples: {metrics_int8['samples']:.0f}"
    )

    delta_top1 = metrics_fp32["top1"] - metrics_int8["top1"]
    delta_top5 = metrics_fp32["top5"] - metrics_int8["top5"]
    print(
        f"Performance delta (FP32 - INT8): Top-1 Δ={delta_top1:.2f} pts | Top-5 Δ={delta_top5:.2f} pts"
    )

    metrics_path = results_dir / f"tiny_imagenet_{args.model}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "fp32": metrics_fp32,
                "int8": metrics_int8,
                "delta": {"top1": delta_top1, "top5": delta_top5},
            },
            f,
            indent=2,
        )
    print(f"Saved metric summary to {metrics_path}")

    print("Generating GradCAM comparisons ...")
    gradcam_dir = results_dir / "gradcam" / args.model
    generate_gradcam_samples(
        dataset,
        imagenet_indices,
        folder_to_tiny,
        wnids,
        model_fp32,
        model_int8,
        target_layer_fp32,
        target_layer_int8,
        gradcam_dir,
        num_samples=args.gradcam_samples,
        model_label=model_label,
    )
    print(f"GradCAM images saved to {gradcam_dir}")


if __name__ == "__main__":
    main()
