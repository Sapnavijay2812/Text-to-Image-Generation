# Text-to-Image-Generation

# Project Overview:

This project aims to develop a text-to-image generative model that translates natural language descriptions into high-fidelity images. Leveraging state-of-the-art techniques such as transformer architectures, attention mechanisms, and adversarial training, the model generates images that accurately reflect the given textual descriptions. The project includes a user-friendly web interface for practical applications, allowing users to input text and receive corresponding images.

# Installation Instructions:

To set up the environment and run the project, follow these steps:

*Clone the repository to your local machine:
```
git clone https://github.com/your-repo/text-to-image-generation.git
cd text-to-image-generation
```
Install the required libraries and dependencies:
```
!pip install --upgrade diffusers transformers -q

```
You should see output similar to this:
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 934.9/934.9 kB 17.0 MB/s eta 0:00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 89.8 MB/s eta 0:00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 28.5 MB/s eta 0:00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 108.0 MB/s eta 0:00:00
```
Set up your environment:

```
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
```
Define your configuration settings:
```
class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
````
*
# Usage:

Examples of how to use the code:

Load the Stable Diffusion model:

```
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)
```

Define the function to generate images from text prompts:
```
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image
```

Generate an image from a sample prompt:
```
generate_image("astronaut in space", image_gen_model)

```

# Dependencies:

The following libraries and dependencies are required to run this project:

*diffusers
transformers
torch
pandas
numpy
matplotlib
tqdm
cv2*

Ensure all dependencies are installed and properly configured before running the project.



