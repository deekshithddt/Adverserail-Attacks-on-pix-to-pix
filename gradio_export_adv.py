import os
import gradio as gr
import numpy as np
import torch
from PIL import Image

from fgsm_attack_pix2pix import (
    cw_attack,
    deepfool_attack,
    denorm_to_uint8,
    fgsm_attack,
    pgd_attack,
    pixel_eps_to_model_eps,
    to_model_tensor,
)
from predict_pix2pix import DEFAULT_MODEL_PATH, load_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_generator(model_path, DEVICE)

try:
    MODEL = load_model(DEFAULT_MODEL_PATH)
except FileNotFoundError:
    MODEL = None
    print("Warning: Model file not found. Attack will fail until model is provided.")

def generate_all_adv_images(upload_image, epsilon_px_255, steps):
    if upload_image is None:
        raise gr.Error("Please upload an image.")
    if MODEL is None:
        raise gr.Error("Model not loaded. Please ensure pix2pix_sketch_G.pth exists.")

    image_size = 512

    img_color = Image.fromarray(upload_image).convert("RGB")
    img_gray = img_color.convert("L")

    # input tensor
    x = to_model_tensor(img_gray, image_size, DEVICE)
    
    epsilon_px = float(epsilon_px_255) / 255.0
    epsilon_model = pixel_eps_to_model_eps(epsilon_px)
    alpha_model_pgd = pixel_eps_to_model_eps(epsilon_px / max(int(steps), 1))
    alpha_model_cw = pixel_eps_to_model_eps(0.01)

    clean_img_out = denorm_to_uint8(x)
    
    # Run all 4 attacks independently
    _, x_fgsm, _, _ = fgsm_attack(MODEL, x, epsilon_model)
    _, x_pgd, _, _ = pgd_attack(MODEL, x, epsilon_model, int(steps), alpha_model_pgd, True, True)
    _, x_deepfool, _, _ = deepfool_attack(MODEL, x, int(steps), 0.35, 0.15)
    _, x_cw, _, _ = cw_attack(MODEL, x, epsilon_model, int(steps), alpha_model_cw, 1.5)

    return (
        clean_img_out,
        denorm_to_uint8(x_fgsm),
        denorm_to_uint8(x_pgd),
        denorm_to_uint8(x_deepfool),
        denorm_to_uint8(x_cw)
    )

custom_css = """
body {
  background: linear-gradient(135deg, #e0f2fe 0%, #eef2ff 100%);
}
.gradio-container {
  max-width: 1100px !important;
}
"""

with gr.Blocks(title="Adversarial Image Exporter", css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Adversarial Image Exporter (All Attacks)</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload an image to simultaneously generate all 4 <b>adversarial images</b>. Hover over any of the corrupted Grayscale photos and click the download icon to save it for your demonstrations.</p>")

    with gr.Row():
        upload_img = gr.Image(type="numpy", label="Upload Image")
        with gr.Column():
            gr.Markdown("### Attack Settings")
            epsilon = gr.Slider(1, 128, value=32, step=1, label="Epsilon (Strength) /255")
            steps = gr.Slider(1, 100, value=10, step=1, label="Steps (For PGD/DeepFool/CW)")
            run_btn = gr.Button("Generate All Adversarial Images", variant="primary")

    with gr.Row():
        clean_out = gr.Image(type="numpy", label="1. Clean (No Attack)", interactive=False)
        fgsm_out = gr.Image(type="numpy", label="2. FGSM (Download!)", interactive=False)
        pgd_out = gr.Image(type="numpy", label="3. PGD (Download!)", interactive=False)

    with gr.Row():
        deepfool_out = gr.Image(type="numpy", label="4. DeepFool (Download!)", interactive=False)
        cw_out = gr.Image(type="numpy", label="5. CW (Download!)", interactive=False)
        
        # Add an empty column purely for layout balancing.
        gr.Column()

    run_btn.click(
        fn=generate_all_adv_images,
        inputs=[upload_img, epsilon, steps],
        outputs=[clean_out, fgsm_out, pgd_out, deepfool_out, cw_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)
