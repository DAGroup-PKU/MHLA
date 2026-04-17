import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device)

sana = SanaPipeline("configs/sana_config/512ms/Sana_600M_img512_MHLA.yaml")
sana.from_pretrained("hf://DAGroup-PKU/MHLA/SANA_MHLA/SANA_MHLA.pth")

prompt = "A cute cat."

output_path = os.path.join(os.getcwd(), "output.png")

num_samples = 50
width, height = 512, 512

generator.manual_seed(42)
image = sana(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=20,
    generator=generator,
)
save_image(image, output_path, nrow=1, normalize=True, value_range=(-1, 1))