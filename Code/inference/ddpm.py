import torch
from torchvision import transforms
from PIL import Image
import os

DDPM_MODEL_PATH = "C:\Users\mahat\OneDrive\Documents\Vessel-Dectection\Code\DDPM\DDPM_unconditional_best_weights_1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpm_model = torch.load(DDPM_MODEL_PATH, map_location=DEVICE)
ddpm_model.eval()

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

def denoise_image(image_path, output_folder="temp"):
    os.makedirs(output_folder, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        denoised = ddpm_model.sample(x)

    output_image = to_pil(denoised.squeeze().cpu())
    output_path = os.path.join(output_folder, "enhanced_" + os.path.basename(image_path))
    output_image.save(output_path)
    print(f"[DDPM] Enhanced image saved at: {output_path}")
    return output_path

if __name__ == "__main__":
    input_path = "temp/image0.jpg"
    denoise_image(input_path)
