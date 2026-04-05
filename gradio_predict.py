import os
import uuid

import gradio as gr
import numpy as np
from PIL import Image

from predict_pix2pix import DEFAULT_MODEL_PATH, load_generator, predict_sketch


DEVICE = "cuda"
try:
    import torch

    if not torch.cuda.is_available():
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_generator(model_path, DEVICE)


MODEL = load_model(DEFAULT_MODEL_PATH)
DISPLAY_HEIGHT = 640


def run_prediction(input_image, image_size):
    if input_image is None:
        raise gr.Error("Please upload an image.")

    in_h, in_w = input_image.shape[:2]
    temp_input = f"_gradio_input_{uuid.uuid4().hex}.png"
    Image.fromarray(input_image).save(temp_input)
    try:
        _, sketch = predict_sketch(
            MODEL, temp_input, device=DEVICE, image_size=int(image_size)
        )
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

    # Ensure output pixel size exactly matches input pixel size.
    if sketch.shape[0] != in_h or sketch.shape[1] != in_w:
        sketch = np.array(
            Image.fromarray(sketch).resize((in_w, in_h), Image.Resampling.LANCZOS)
        )

    return sketch


custom_css = """
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  padding-left: 8px !important;
  padding-right: 8px !important;
}

/* Force image content to fill each panel in Gradio 3 */
#input_panel img,
#input_panel canvas,
#output_panel img,
#output_panel canvas {
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
  object-position: center !important;
}

#input_panel .image-container,
#output_panel .image-container,
#input_panel .svelte-1ipelgc,
#output_panel .svelte-1ipelgc {
  min-height: 640px !important;
}
"""


with gr.Blocks(title="Image to Sketch Generator", css=custom_css) as demo:
    gr.Markdown("# Image to Sketch Generator")
    gr.Markdown("Upload an image and generate a pencil sketch.")

    with gr.Row():
        input_img = gr.Image(
            type="numpy",
            label="Input Image",
            height=DISPLAY_HEIGHT,
            elem_id="input_panel",
        )
        output_img = gr.Image(
            type="numpy",
            label="Predicted Sketch",
            height=DISPLAY_HEIGHT,
            elem_id="output_panel",
        )

    image_size = gr.Slider(
        minimum=256,
        maximum=1024,
        value=512,
        step=32,
        label="Model Input Size",
    )

    run_btn = gr.Button("Generate Sketch")
    run_btn.click(
        fn=run_prediction,
        inputs=[input_img, image_size],
        outputs=output_img,
    )


if __name__ == "__main__":
    demo.launch(share=True)
