import os

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter

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
DISPLAY_HEIGHT = 280


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_generator(model_path, DEVICE)


MODEL = load_model(DEFAULT_MODEL_PATH)


def _resize_same(arr, w, h):
    if arr.shape[0] != h or arr.shape[1] != w:
        return np.array(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))
    return arr


def _finalize_output(clean_out, adv_out, blend_strength, blur_radius):
    blend = float(blend_strength)
    final_out = ((1.0 - blend) * clean_out.astype(np.float32)) + (
        blend * adv_out.astype(np.float32)
    )
    final_out = np.clip(final_out, 0, 255).astype(np.uint8)
    if float(blur_radius) > 0:
        final_out = np.array(
            Image.fromarray(final_out).filter(
                ImageFilter.GaussianBlur(radius=float(blur_radius))
            )
        )
    return final_out


def _visibility_rescue(clean_out, out_img):
    # If attack collapses to near-black/low-dynamic output, recover visible structure
    # for UI display while keeping distortion evident.
    mean_v = float(out_img.mean())
    p2 = float(np.percentile(out_img, 2))
    p98 = float(np.percentile(out_img, 98))
    dynamic = p98 - p2
    if mean_v < 30.0 or dynamic < 25.0:
        norm = (out_img.astype(np.float32) - p2) / max(dynamic, 1.0)
        norm = np.clip(norm, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
        # Keep changes visible but avoid total black frame.
        rescued = 0.65 * norm.astype(np.float32) + 0.35 * clean_out.astype(np.float32)
        return np.clip(rescued, 0, 255).astype(np.uint8)
    return out_img


def _tone_match_to_clean(clean_out, out_img):
    # Match attacked output brightness/contrast to clean sketch for better visibility.
    c = clean_out.astype(np.float32)
    o = out_img.astype(np.float32)
    c_mean, c_std = float(c.mean()), float(c.std())
    o_mean, o_std = float(o.mean()), float(o.std())
    o_std = max(o_std, 1.0)
    matched = (o - o_mean) * (c_std / o_std) + c_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def _amplify_difference(clean_out, out_img, factor=1.0):
    # Display-only amplification: makes subtle attack differences easier to see.
    c = clean_out.astype(np.float32)
    o = out_img.astype(np.float32)
    amplified = c + float(factor) * (o - c)
    return np.clip(amplified, 0, 255).astype(np.uint8)


def _mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def _psnr(a, b):
    mse = _mse(a, b)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_global(a, b):
    # Lightweight global SSIM (single-channel), no extra dependencies.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if abs(den) < 1e-12:
        return 1.0
    return float(num / den)


def _metric_row(name, clean, out, ran):
    if not ran or out is None:
        return [name, "skipped", "skipped", "skipped"]
    mse = _mse(clean, out)
    psnr = _psnr(clean, out)
    ssim = _ssim_global(clean, out)
    return [name, f"{mse:.4f}", f"{psnr:.4f}", f"{ssim:.4f}"]


def run_all_attacks(
    upload_image,
    run_fgsm,
    run_pgd,
    run_deepfool,
    run_cw,
    epsilon_px_255,
    steps,
    alpha_px_255,
    random_start,
    project,
    image_size,
    cw_c,
    deepfool_shift_threshold,
    overshoot,
    blend_strength,
    blur_radius,
    vis_boost,
):
    input_image = upload_image
    if input_image is None:
        raise gr.Error("Please upload an image.")

    orig_h, orig_w = input_image.shape[:2]
    img_color = Image.fromarray(input_image).convert("RGB")
    img_gray = img_color.convert("L")

    x = to_model_tensor(img_gray, int(image_size), DEVICE)
    epsilon_px = float(epsilon_px_255) / 255.0
    epsilon_model = pixel_eps_to_model_eps(epsilon_px)
    alpha_px = None if alpha_px_255 <= 0 else float(alpha_px_255) / 255.0
    alpha_model = None if alpha_px is None else pixel_eps_to_model_eps(alpha_px)
    used_alpha_px = alpha_px if alpha_px is not None else epsilon_px / max(int(steps), 1)

    y_clean_ref, _, _, _ = fgsm_attack(MODEL, x, epsilon_model)
    clean_out = _resize_same(denorm_to_uint8(y_clean_ref), orig_w, orig_h)

    fgsm_out = None
    pgd_out = None
    deepfool_out = None
    cw_out = None
    loss_fgsm = None
    loss_pgd = None
    loss_deepfool = None
    loss_cw = None

    if bool(run_fgsm):
        _, _, y_fgsm, loss_fgsm = fgsm_attack(MODEL, x, epsilon_model)
        fgsm_out = _resize_same(denorm_to_uint8(y_fgsm), orig_w, orig_h)
        fgsm_out = _finalize_output(clean_out, fgsm_out, blend_strength, blur_radius)
        fgsm_out = _amplify_difference(clean_out, fgsm_out, factor=float(vis_boost))

    if bool(run_pgd):
        _, _, y_pgd, loss_pgd = pgd_attack(
            model=MODEL,
            x=x,
            epsilon=epsilon_model,
            steps=int(steps),
            alpha=alpha_model,
            random_start=bool(random_start),
            project=bool(project),
        )
        pgd_out = _resize_same(denorm_to_uint8(y_pgd), orig_w, orig_h)
        pgd_out = _finalize_output(clean_out, pgd_out, blend_strength, blur_radius)
        pgd_out = _visibility_rescue(clean_out, pgd_out)
        pgd_out = _tone_match_to_clean(clean_out, pgd_out)

    if bool(run_deepfool):
        _, _, y_deepfool, loss_deepfool = deepfool_attack(
            model=MODEL,
            x=x,
            steps=int(steps),
            shift_threshold=float(deepfool_shift_threshold),
            overshoot=float(overshoot),
        )
        deepfool_out = _resize_same(denorm_to_uint8(y_deepfool), orig_w, orig_h)
        deepfool_out = _finalize_output(clean_out, deepfool_out, blend_strength, blur_radius)
        deepfool_out = _amplify_difference(clean_out, deepfool_out, factor=float(vis_boost))

    if bool(run_cw):
        _, _, y_cw, loss_cw = cw_attack(
            model=MODEL,
            x=x,
            epsilon=epsilon_model,
            steps=int(steps),
            alpha=alpha_model,
            c=float(cw_c),
        )
        cw_out = _resize_same(denorm_to_uint8(y_cw), orig_w, orig_h)
        cw_out = _finalize_output(clean_out, cw_out, blend_strength, blur_radius)
        cw_out = _visibility_rescue(clean_out, cw_out)
        cw_out = _tone_match_to_clean(clean_out, cw_out)

    details = (
        f"Device: {DEVICE}\n"
        f"Epsilon: {epsilon_px:.6f} ({epsilon_px_255:.0f}/255)\n"
        f"Steps: {int(steps)}\n"
        f"Alpha: {used_alpha_px:.6f} ({used_alpha_px*255:.2f}/255)\n"
        f"Run FGSM: {bool(run_fgsm)}\n"
        f"Run PGD: {bool(run_pgd)}\n"
        f"Run DeepFool: {bool(run_deepfool)}\n"
        f"Run CW: {bool(run_cw)}\n"
        f"Random start (PGD): {bool(random_start)}\n"
        f"Projection (PGD): {bool(project)}\n"
        f"CW c: {float(cw_c):.3f}\n"
        f"DeepFool threshold: {float(deepfool_shift_threshold):.4f}\n"
        f"DeepFool overshoot: {float(overshoot):.4f}\n"
        f"FGSM/DeepFool visibility boost: {float(vis_boost):.2f}\n"
        f"Blend strength: {float(blend_strength):.3f}\n"
        f"Blur radius: {float(blur_radius):.2f}\n"
        f"L1 shift FGSM: {f'{loss_fgsm:.6f}' if loss_fgsm is not None else 'skipped'}\n"
        f"L1 shift PGD: {f'{loss_pgd:.6f}' if loss_pgd is not None else 'skipped'}\n"
        f"L1 shift DeepFool: {f'{loss_deepfool:.6f}' if loss_deepfool is not None else 'skipped'}\n"
        f"L1 shift CW: {f'{loss_cw:.6f}' if loss_cw is not None else 'skipped'}"
    )

    metrics_rows = [
        _metric_row("FGSM", clean_out, fgsm_out, bool(run_fgsm)),
        _metric_row("PGD", clean_out, pgd_out, bool(run_pgd)),
        _metric_row("DeepFool", clean_out, deepfool_out, bool(run_deepfool)),
        _metric_row("CW", clean_out, cw_out, bool(run_cw)),
    ]

    return clean_out, fgsm_out, pgd_out, deepfool_out, cw_out, details, metrics_rows


custom_css = """
body {
  background: linear-gradient(135deg, #fff3d6 0%, #dff7ff 45%, #ffe3ef 100%);
}
.gradio-container {
  max-width: 1300px !important;
}
#main-title {
  text-align: center;
  font-size: 40px;
  font-weight: 800;
  color: #1f2937;
  margin-bottom: 4px;
}
#subtitle {
  text-align: center;
  color: #374151;
  margin-bottom: 16px;
}
"""


with gr.Blocks(title="Sketch Generator Attack", css=custom_css) as demo:
    gr.Markdown("<div id='main-title'>Sketch Generator - Adversarial Attack</div>")
    gr.Markdown("<div id='subtitle'>Clean sketch + FGSM + PGD + DeepFool + CW outputs side by side.</div>")

    with gr.Row():
        upload_img = gr.Image(
            type="numpy",
            label="Upload Image",
            height=DISPLAY_HEIGHT,
            sources=["upload", "webcam", "clipboard"],
        )
        clean_img = gr.Image(type="numpy", label="Sketch Output (Clean)", height=DISPLAY_HEIGHT)
        fgsm_img = gr.Image(type="numpy", label="FGSM Output", height=DISPLAY_HEIGHT)
        pgd_img = gr.Image(type="numpy", label="PGD Output", height=DISPLAY_HEIGHT)
        deepfool_img = gr.Image(type="numpy", label="DeepFool Output", height=DISPLAY_HEIGHT)
        cw_img = gr.Image(type="numpy", label="CW Output", height=DISPLAY_HEIGHT)

    details = gr.Textbox(label="Attack Details", lines=8)
    metrics_tbl = gr.Dataframe(
        headers=["Attack", "MSE (higher=worse)", "PSNR (lower=worse)", "SSIM (lower=worse)"],
        datatype=["str", "str", "str", "str"],
        row_count=4,
        col_count=(4, "fixed"),
        label="Distortion Metrics (vs Clean Sketch)",
    )

    with gr.Row():
        run_fgsm = gr.Checkbox(value=True, label="Run FGSM")
        run_pgd = gr.Checkbox(value=True, label="Run PGD")
        run_deepfool = gr.Checkbox(value=True, label="Run DeepFool")
        run_cw = gr.Checkbox(value=True, label="Run CW")

    with gr.Row():
        epsilon = gr.Slider(1, 96, value=32, step=1, label="Epsilon (/255)")
        steps = gr.Slider(1, 200, value=40, step=1, label="Steps")
        alpha = gr.Slider(
            0.0,
            32.0,
            value=1.0,
            step=0.5,
            label="Alpha (/255, 0 = auto)",
        )

    with gr.Row():
        random_start = gr.Checkbox(value=True, label="Random Start (PGD)")
        project = gr.Checkbox(value=True, label="Project to Epsilon Ball (PGD)")
        cw_c = gr.Slider(0.1, 20.0, value=1.5, step=0.1, label="CW c")
        deepfool_shift_threshold = gr.Slider(
            0.01, 0.8, value=0.35, step=0.01, label="DeepFool Shift Threshold"
        )
        overshoot = gr.Slider(0.0, 0.4, value=0.15, step=0.01, label="DeepFool Overshoot")

    with gr.Row():
        image_size = gr.Slider(256, 1024, value=512, step=32, label="Model Input Size")
        blend_strength = gr.Slider(
            0.0, 1.0, value=0.85, step=0.01, label="Distortion Blend Strength"
        )
        blur_radius = gr.Slider(
            0.0, 3.0, value=0.0, step=0.1, label="Blur Radius"
        )
        vis_boost = gr.Slider(
            1.0, 4.0, value=2.2, step=0.1, label="FGSM/DeepFool Visibility Boost"
        )

    run_btn = gr.Button("Run Attack", variant="primary")
    run_btn.click(
        fn=run_all_attacks,
        inputs=[
            upload_img,
            run_fgsm,
            run_pgd,
            run_deepfool,
            run_cw,
            epsilon,
            steps,
            alpha,
            random_start,
            project,
            image_size,
            cw_c,
            deepfool_shift_threshold,
            overshoot,
            blend_strength,
            blur_radius,
            vis_boost,
        ],
        outputs=[clean_img, fgsm_img, pgd_img, deepfool_img, cw_img, details, metrics_tbl],
    )


if __name__ == "__main__":
    demo.launch(share=True)
