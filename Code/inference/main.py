from ddpm_inference import denoise_image
from yolov7_inference import detect_vessels

input_image = "temp/demo_img.jpg"
enhanced_image = denoise_image(input_image)
detect_vessels(enhanced_image)