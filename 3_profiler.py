import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision import transforms
from torch.profiler import profile, ProfilerActivity, record_function

from pytorch_grad_cam.utils.image import preprocess_image

# Load image
cat_and_dog_image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
cat_and_dog = np.array(Image.open(requests.get(cat_and_dog_image_url, stream=True).raw))
# Display the image using matplotlib
# plt.figure(figsize=(5, 4))
# plt.imshow(cat_and_dog)
# plt.axis('off')  # Hide axes
# plt.title('cat_and_dog')
# plt.show()

# Convert to float32 first for matplotlib compatibility, then convert back to float16 for model
cat_and_dog = np.float32(cat_and_dog) / 255  # Keep float32 for model
input_tensor = preprocess_image(cat_and_dog, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print(f'Input tensor shape: {input_tensor.shape}')

# Choose model
model_name = "alexnet"  # Change to "alexnet" to use AlexNet
if model_name == "resnet18":
    print(f'Using model: {model_name}')
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif model_name == "alexnet":
    print(f'Using model: {model_name}')
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
else:
    raise ValueError("Unsupported model name. Supported models: 'resnet18', 'alexnet'.")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]

scheduler = torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1)

with profile(
    activities=activities,
    schedule=scheduler,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
    # record_shapes=True,
    # profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        for step in range(12):
            desc = 'Profiling ...'
            _ = model(input_tensor)
            prof.step()


