import argparse
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision import transforms
from torch.profiler import profile, ProfilerActivity, record_function

from pytorch_grad_cam.utils.image import preprocess_image

# Import MPS profiler if available
try:
    import torch.mps.profiler as mps_profiler
    MPS_PROFILER_AVAILABLE = True
except ImportError:
    MPS_PROFILER_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch Model Profiler')
    
    parser.add_argument('--model', type=str, default='resnet101',
                       choices=['resnet18', 'resnet101', 'alexnet'],
                       help='Model to use for profiling (default: resnet101)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to run on (default: auto - uses CUDA/MPS if available)')
    
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp16', 'fp32'],
                       help='Precision to use (default: fp32)')
    
    parser.add_argument('--image-url', type=str,
                       default="https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png",
                       help='URL of the image to use for profiling')
    
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of times to replicate the input in batch dimension (default: 1)')
    
    return parser.parse_args()


def load_image(url, batch_size=1):
    """Load and preprocess image from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = np.array(Image.open(response.raw))
        
        # Convert to float32 and normalize
        image = np.float32(image) / 255
        input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Replicate the input tensor along batch dimension
        if batch_size > 1:
            input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)
            print(f"Replicated input tensor to batch size: {batch_size}")
        
        return input_tensor
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        raise


def load_model(model_name):
    """Load the specified model."""
    print(f'Using model: {model_name}')
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    model.eval()
    return model


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but was requested")
        device = torch.device("cuda")
    elif device_arg == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available but was requested")
        device = torch.device("mps")
    else:  # cpu
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    elif device.type == 'mps':
        print("Using Apple Metal Performance Shaders (MPS)")
        if MPS_PROFILER_AVAILABLE:
            print("MPS profiler available for detailed GPU analysis")
        else:
            print("MPS profiler not available - using standard profiler only")
    
    return device


def cleanup_mps_artifacts():
    """Clean up MPS profiler artifacts like 0000-. directories and move gputrace files to log directory."""
    import os
    import shutil
    import glob
    
    # Move any .gputrace directories from 0000-* paths to log directory
    for pattern in ['0000-*']:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                # Extract the gputrace directory name
                for item in os.listdir(path):
                    if item.endswith('.gputrace'):
                        source = os.path.join(path, item)
                        destination = os.path.join('./log', item)
                        if os.path.exists(source):
                            print(f"Moving MPS trace from {source} to {destination}")
                            shutil.move(source, destination)
                
                print(f"Cleaning up MPS profiler artifact: {path}")
                shutil.rmtree(path)
    
    # Also clean up any standalone .gputrace directories in root
    for path in glob.glob('*.gputrace'):
        if os.path.isdir(path):
            destination = os.path.join('./log', path)
            print(f"Moving MPS trace from {path} to {destination}")
            shutil.move(path, destination)


def run_profiling(model, input_tensor, model_name, device):
    """Run the profiling session."""
    # Setup profiler activities
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    # Setup profiler scheduler
    scheduler = torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1)
    
    print("Starting profiling...")
    
    # Use MPS profiler for MPS devices if available
    if device.type == 'mps' and MPS_PROFILER_AVAILABLE:
        print("Using MPS profiler for Apple Silicon...")
        
        # Check if Metal capture is enabled
        import os
        import time
        
        # Check if metal capture is actually enabled and usable
        if mps_profiler.is_metal_capture_enabled():
            try:
                # Create unique filename with timestamp
                timestamp = str(int(time.time()))
                # Use simple filename without path to avoid MPS profiler path issues
                capture_file = f'{model_name}_mps_{timestamp}'
                
                print(f"Metal capture enabled. Starting GPU trace capture: {capture_file}")
                # Use MPS metal capture for GPU profiling
                with mps_profiler.metal_capture(capture_file):
                    with profile(
                        activities=activities,
                        schedule=scheduler,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
                        with_stack=True
                    ) as prof:
                        with torch.no_grad():
                            for step in range(12):
                                _ = model(input_tensor)
                                prof.step()
                
                # MPS profiler creates the .gputrace directory automatically
                print(f"MPS GPU trace directory created: {capture_file}.gputrace")
                print("You can open this directory with Xcode's Instruments or Metal Debugger")
            except Exception as e:
                print(f"MPS metal capture failed: {e}")
                print("Falling back to standard profiling...")
                # Fall back to standard profiling
                with profile(
                    activities=activities,
                    schedule=scheduler,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
                    with_stack=True
                ) as prof:
                    with torch.no_grad():
                        for step in range(12):
                            _ = model(input_tensor)
                            prof.step()
        else:
            print("Metal capture not enabled. Using MPS OS Signpost tracing.")
            print("To enable Metal GPU capture, run: export MTL_CAPTURE_ENABLED=1")
            
            # Use MPS OS Signpost tracing (doesn't require MTL_CAPTURE_ENABLED)
            with mps_profiler.profile():
                with profile(
                    activities=activities,
                    schedule=scheduler,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
                    with_stack=True
                ) as prof:
                    with torch.no_grad():
                        for step in range(12):
                            _ = model(input_tensor)
                            prof.step()
            
            print("MPS OS Signpost tracing completed. Use Instruments to view traces.")
    else:
        # Standard profiling for CPU/CUDA or when MPS profiler is not available
        with profile(
            activities=activities,
            schedule=scheduler,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for step in range(12):
                    _ = model(input_tensor)
                    prof.step()
    
    print(f"Profiling completed. Results saved to ./log/{model_name}")
    
    # Clean up MPS profiler artifacts if using MPS
    if device.type == 'mps' and MPS_PROFILER_AVAILABLE:
        cleanup_mps_artifacts()


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Precision: {args.precision}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image URL: {args.image_url}")
    print()
    
    # Load image
    print("Loading image...")
    input_tensor = load_image(args.image_url, args.batch_size)
    print(f'Input tensor shape: {input_tensor.shape}')
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Setup device
    device = setup_device(args.device)
    
    # Apply precision settings
    if args.precision == 'fp16':
        print("Using half precision (FP16)")
        model = model.half()
        input_tensor = input_tensor.half()
    else:
        print("Using full precision (FP32)")
    
    # Move to device
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Print dtype information
    print(f"Model parameter dtype: {next(model.parameters()).dtype}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print()
    
    # Run profiling
    run_profiling(model, input_tensor, args.model, device)


if __name__ == "__main__":
    main()


