import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import uuid
import cv2
import gradio as gr
import numpy as np
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
DISPLAY_HEIGHT = 280

# ==========================================
# UI & METRICS UTILITIES
# ==========================================
def format_html_table(headers, rows):
    html = "<table style='width:100%; border-collapse: collapse; text-align: left;'>"
    html += "<tr style='background-color:#f3f4f6; color:#111;'>"
    for h in headers:
        html += f"<th style='padding: 8px; border-bottom: 2px solid #ddd;'>{h}</th>"
    html += "</tr>"
    for r in rows:
        html += "<tr>"
        for v in r:
            html += f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>{v}</td>"
        html += "</tr>"
    html += "</table>"
    return html

def _resize_same(arr, w, h):
    if arr.shape[0] != h or arr.shape[1] != w:
        return np.array(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))
    return arr

def _finalize_output(clean_out, adv_out, blend_strength, blur_radius):
    blend = float(blend_strength)
    final_out = ((1.0 - blend) * clean_out.astype(np.float32)) + (blend * adv_out.astype(np.float32))
    final_out = np.clip(final_out, 0, 255).astype(np.uint8)
    if float(blur_radius) > 0:
        final_out = np.array(Image.fromarray(final_out).filter(ImageFilter.GaussianBlur(radius=float(blur_radius))))
    return final_out

def _visibility_rescue(clean_out, out_img):
    mean_v = float(out_img.mean())
    p2 = float(np.percentile(out_img, 2))
    p98 = float(np.percentile(out_img, 98))
    dynamic = p98 - p2
    if mean_v < 30.0 or dynamic < 25.0:
        norm = (out_img.astype(np.float32) - p2) / max(dynamic, 1.0)
        norm = np.clip(norm, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
        rescued = 0.65 * norm.astype(np.float32) + 0.35 * clean_out.astype(np.float32)
        return np.clip(rescued, 0, 255).astype(np.uint8)
    return out_img

def _tone_match_to_clean(clean_out, out_img):
    c = clean_out.astype(np.float32)
    o = out_img.astype(np.float32)
    c_mean, c_std = float(c.mean()), float(c.std())
    o_mean, o_std = float(o.mean()), float(o.std())
    o_std = max(o_std, 1.0)
    matched = (o - o_mean) * (c_std / o_std) + c_mean
    return np.clip(matched, 0, 255).astype(np.uint8)

def _amplify_difference(clean_out, out_img, factor=1.0):
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
    if mse <= 1e-12: return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))

def _ssim_global(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a = a.mean(); mu_b = b.mean(); var_a = a.var(); var_b = b.var()
    cov_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255.0) ** 2; c2 = (0.03 * 255.0) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if abs(den) < 1e-12: return 1.0
    return float(num / den)

def _metric_row(name, clean, out, ran):
    if not ran or out is None: return [name, "skipped", "skipped", "skipped"]
    mse = _mse(clean, out)
    psnr = _psnr(clean, out)
    ssim = _ssim_global(clean, out)
    return [name, f"{mse:.4f}", f"{psnr:.4f}", f"{ssim:.4f}"]

def generate_heatmap(clean_img, adv_img):
    if adv_img is None: return None
    diff = np.abs(adv_img.astype(np.float32) - clean_img.astype(np.float32))
    diff = np.clip(diff * 30, 0, 255).astype(np.uint8)
    try:
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    except Exception:
        heatmap = np.stack([diff, np.zeros_like(diff), np.zeros_like(diff)], axis=-1)
    return heatmap

def save_temp_png(np_arr, prefix):
    if np_arr is None: return None
    filename = f"{prefix}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(os.getcwd(), filename)
    Image.fromarray(np_arr).save(filepath, format="PNG")
    return filepath

# ==========================================
# TAB 1 LOGIC: NORMAL PREDICT
# ==========================================
def tab1_predict(input_image, image_size):
    if input_image is None: raise gr.Error("Please upload an image.")
    temp_input = f"_tab1_{uuid.uuid4().hex}.png"
    Image.fromarray(input_image).save(temp_input)
    try:
        _, sketch = predict_sketch(MODEL, temp_input, device=DEVICE, image_size=int(image_size))
    finally:
        if os.path.exists(temp_input): os.remove(temp_input)
    return sketch

# ==========================================
# TAB 2 LOGIC: EXPORTER / ARTIST TOOL
# ==========================================
def tab2_export_attacks(image, eps_255, steps):
    if image is None: raise gr.Error("Please upload an image to attack.")
    
    img_pil = Image.fromarray(image).convert("L")
    x = to_model_tensor(img_pil, 512, DEVICE)
    
    epsilon_px = float(eps_255) / 255.0
    epsilon_model = pixel_eps_to_model_eps(epsilon_px)
    alpha_model_pgd = pixel_eps_to_model_eps(epsilon_px / max(int(steps), 1))
    alpha_model_cw = pixel_eps_to_model_eps(0.01)
    
    clean_tensor_uint8 = denorm_to_uint8(x)
    
    # Running bounded steps to keep UI snappy
    _, x_fgsm, _, _ = fgsm_attack(MODEL, x, epsilon_model)
    _, x_pgd, _, _ = pgd_attack(MODEL, x, epsilon_model, int(steps), alpha_model_pgd, True, True)
    _, x_df, _, _ = deepfool_attack(MODEL, x, int(steps), 0.35, 0.15)
    _, x_cw, _, _ = cw_attack(MODEL, x, epsilon_model, int(steps), alpha_model_cw, 1.5)
    
    adv_fgsm = denorm_to_uint8(x_fgsm)
    adv_pgd = denorm_to_uint8(x_pgd)
    adv_df = denorm_to_uint8(x_df)
    adv_cw = denorm_to_uint8(x_cw)

    dl_fgsm = save_temp_png(adv_fgsm, "adv_fgsm")
    dl_pgd = save_temp_png(adv_pgd, "adv_pgd")
    dl_df = save_temp_png(adv_df, "adv_df")
    dl_cw = save_temp_png(adv_cw, "adv_cw")
    
    hw_fgsm = generate_heatmap(clean_tensor_uint8, adv_fgsm)
    hw_pgd = generate_heatmap(clean_tensor_uint8, adv_pgd)
    hw_df = generate_heatmap(clean_tensor_uint8, adv_df)
    hw_cw = generate_heatmap(clean_tensor_uint8, adv_cw)
    
    return [
        adv_fgsm, hw_fgsm, dl_fgsm,
        adv_pgd, hw_pgd, dl_pgd,
        adv_df, hw_df, dl_df,
        adv_cw, hw_cw, dl_cw
    ]

# ==========================================
# TAB 3 LOGIC: DETAILED ATTACK ANALYSIS
# ==========================================
def tab3_attack_analysis(
    upload_image, run_fgsm, run_pgd, run_deepfool, run_cw, epsilon_px_255, steps, alpha_px_255, 
    random_start, project, image_size, cw_c, deepfool_shift_threshold, overshoot, blend_strength, blur_radius, vis_boost
):
    if upload_image is None: raise gr.Error("Please upload an image.")

    orig_h, orig_w = upload_image.shape[:2]
    img_gray = Image.fromarray(upload_image).convert("L")

    x = to_model_tensor(img_gray, int(image_size), DEVICE)
    epsilon_px = float(epsilon_px_255) / 255.0
    epsilon_model = pixel_eps_to_model_eps(epsilon_px)
    alpha_px = None if alpha_px_255 <= 0 else float(alpha_px_255) / 255.0
    alpha_model = None if alpha_px is None else pixel_eps_to_model_eps(alpha_px)
    used_alpha_px = alpha_px if alpha_px is not None else epsilon_px / max(int(steps), 1)

    y_clean_ref, _, _, _ = fgsm_attack(MODEL, x, epsilon_model)
    clean_out = _resize_same(denorm_to_uint8(y_clean_ref), orig_w, orig_h)

    # Variables
    fgsm_out = None; pgd_out = None; deepfool_out = None; cw_out = None
    loss_fgsm = None; loss_pgd = None; loss_df = None; loss_cw = None

    if bool(run_fgsm):
        _, _, y_fgsm, loss_fgsm = fgsm_attack(MODEL, x, epsilon_model)
        fgsm_out = _resize_same(denorm_to_uint8(y_fgsm), orig_w, orig_h)
        fgsm_out = _finalize_output(clean_out, fgsm_out, blend_strength, blur_radius)
        fgsm_out = _amplify_difference(clean_out, fgsm_out, factor=float(vis_boost))

    if bool(run_pgd):
        _, _, y_pgd, loss_pgd = pgd_attack(MODEL, x, epsilon_model, int(steps), alpha_model, bool(random_start), bool(project))
        pgd_out = _resize_same(denorm_to_uint8(y_pgd), orig_w, orig_h)
        pgd_out = _finalize_output(clean_out, pgd_out, blend_strength, blur_radius)
        pgd_out = _visibility_rescue(clean_out, pgd_out)
        pgd_out = _tone_match_to_clean(clean_out, pgd_out)

    if bool(run_deepfool):
        _, _, y_df, loss_df = deepfool_attack(MODEL, x, int(steps), float(deepfool_shift_threshold), float(overshoot))
        deepfool_out = _resize_same(denorm_to_uint8(y_df), orig_w, orig_h)
        deepfool_out = _finalize_output(clean_out, deepfool_out, blend_strength, blur_radius)
        deepfool_out = _amplify_difference(clean_out, deepfool_out, factor=float(vis_boost))

    if bool(run_cw):
        _, _, y_cw, loss_cw = cw_attack(MODEL, x, epsilon_model, int(steps), alpha_model, float(cw_c))
        cw_out = _resize_same(denorm_to_uint8(y_cw), orig_w, orig_h)
        cw_out = _finalize_output(clean_out, cw_out, blend_strength, blur_radius)
        cw_out = _visibility_rescue(clean_out, cw_out)
        cw_out = _tone_match_to_clean(clean_out, cw_out)

    details = (
        f"Device: {DEVICE}\nEpsilon: {epsilon_px:.6f} ({epsilon_px_255:.0f}/255)\nSteps: {int(steps)}\n"
        f"Alpha: {used_alpha_px:.6f} ({used_alpha_px*255:.2f}/255)\n"
        f"Run FGSM/PGD/DF/CW: {bool(run_fgsm)}/{bool(run_pgd)}/{bool(run_deepfool)}/{bool(run_cw)}\n"
        f"L1 shifts -> FGSM: {f'{loss_fgsm:.6f}' if loss_fgsm else 'skipped'} | PGD: {f'{loss_pgd:.6f}' if loss_pgd else 'skipped'} | "
        f"DF: {f'{loss_df:.6f}' if loss_df else 'skipped'} | CW: {f'{loss_cw:.6f}' if loss_cw else 'skipped'}"
    )

    metrics_rows = [
        _metric_row("FGSM", clean_out, fgsm_out, bool(run_fgsm)),
        _metric_row("PGD", clean_out, pgd_out, bool(run_pgd)),
        _metric_row("DeepFool", clean_out, deepfool_out, bool(run_deepfool)),
        _metric_row("CW", clean_out, cw_out, bool(run_cw)),
    ]
    metrics_html = format_html_table(["Attack", "MSE (higher=worse)", "PSNR (lower=worse)", "SSIM (lower=worse)"], metrics_rows)

    return clean_out, fgsm_out, pgd_out, deepfool_out, cw_out, details, metrics_html


# ==========================================
# TAB 4 LOGIC: SECURITY FIREWALL (DEFENSE)
# ==========================================
def apply_defense(img_array):
    pil_img = Image.fromarray(img_array).convert("L")
    purified = pil_img.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.array(purified)

def tab4_defense_scenario(bad_image, clean_ref_image=None):
    if bad_image is None: raise gr.Error("Please upload the Hacked Image.")
    
    purified_img_array = apply_defense(bad_image)
    
    temp_hacked = f"_tab4_hacked_{uuid.uuid4().hex}.png"
    temp_purified = f"_tab4_purified_{uuid.uuid4().hex}.png"
    Image.fromarray(bad_image).save(temp_hacked)
    Image.fromarray(purified_img_array).save(temp_purified)
    
    _, sketch_ruined = predict_sketch(MODEL, temp_hacked, device=DEVICE, image_size=512)
    _, sketch_restored = predict_sketch(MODEL, temp_purified, device=DEVICE, image_size=512)
    
    metrics = [["No Clean Reference Uploaded", "N/A", "N/A", "N/A"]]
    if clean_ref_image is not None:
        temp_clean = f"_clean_{uuid.uuid4().hex}.png"
        Image.fromarray(clean_ref_image).save(temp_clean)
        _, sketch_clean = predict_sketch(MODEL, temp_clean, device=DEVICE, image_size=512)
        metrics = [
            _metric_row("Hacked (Without Firewall)", sketch_clean, sketch_ruined, True),
            _metric_row("Restored (With Firewall)", sketch_clean, sketch_restored, True),
        ]
        if os.path.exists(temp_clean): os.remove(temp_clean)
        
    metrics_html = format_html_table(["Stage", "MSE", "PSNR", "SSIM"], metrics)
    
    if os.path.exists(temp_hacked): os.remove(temp_hacked)
    if os.path.exists(temp_purified): os.remove(temp_purified)
        
    return sketch_ruined, purified_img_array, sketch_restored, metrics_html

# ==========================================
# UI LAYOUT
# ==========================================
custom_css = """
body { background: linear-gradient(135deg, #fff3d6 0%, #dff7ff 45%, #ffe3ef 100%); }
.gradio-container { max-width: 1300px !important; }
#main-title { text-align: center; font-size: 40px; font-weight: 800; color: #1f2937; margin-bottom: 4px; }
#subtitle { text-align: center; color: #374151; margin-bottom: 16px; }
"""

with gr.Blocks(title="AI Cyber-Forensics Dashboard", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("<div id='main-title'>🛡️ AI Cyber-Forensics Toolkit</div>")
    gr.Markdown("<div id='subtitle'>A complete toolchain integrating Database Sketches, Attack Modeler, Deep Analysis, and Security Firewall.</div>")
    
    with gr.Tabs():
        
        # --- TAB 1: Normal Generation ---
        with gr.TabItem("1. Normal Sketch (gradio_predict)"):
            gr.Markdown("Standard interface for generating sketches from suspect photographs.")
            with gr.Row():
                input_img1 = gr.Image(type="numpy", label="Input Photograph", sources=["upload", "webcam", "clipboard"], height=400)
                output_img1 = gr.Image(type="numpy", label="Generated Sketch", height=400)
            
            image_size1 = gr.Slider(minimum=256, maximum=1024, value=512, step=32, label="Model Input Size")
            btn1 = gr.Button("Generate Sketch", variant="primary")
            btn1.click(fn=tab1_predict, inputs=[input_img1, image_size1], outputs=output_img1)
            
        # --- TAB 2: Attack Exporter / Artist Tool ---
        with gr.TabItem("2. Attack Exporter (Artist Tool)"):
            gr.Markdown("Generate the raw **poisoned photos** (grayscale) that fool the network. Features Epsilon slider and Heatmaps to show noise payload.")
            with gr.Row():
                input_img2 = gr.Image(type="numpy", label="Upload Photo to Poison", sources=["upload", "webcam"], height=300)
                with gr.Column():
                    eps_slider2 = gr.Slider(1, 128, value=32, step=1, label="Attack Scale / Power (Epsilon) / 255")
                    steps_slider2 = gr.Slider(1, 100, value=10, step=1, label="Steps (For PGD/DeepFool/CW)")
                    btn2 = gr.Button("Build Hacked Images", variant="stop")
            
            outputs_t2 = []
            for title in ["FGSM", "PGD", "DeepFool", "CW"]:
                gr.Markdown(f"### {title} Poisoned Payload")
                with gr.Row():
                    adv_disp = gr.Image(type="numpy", label=f"Corrupted Photo (Grayscale)", height=250, interactive=False)
                    heat_disp = gr.Image(type="numpy", label="Thermal Threat Imprint (Noise Heatmap)", height=250, interactive=False)
                    dl_file = gr.File(label=f"Download {title} File to Fool Model")
                    outputs_t2.extend([adv_disp, heat_disp, dl_file])
            btn2.click(fn=tab2_export_attacks, inputs=[input_img2, eps_slider2, steps_slider2], outputs=outputs_t2)

        # --- TAB 3: Detailed Attack Analysis ---
        with gr.TabItem("3. Attack Analysis (gradio_attack)"):
            gr.Markdown("Deep mathematical evaluation of all 4 attacked **sketches** side-by-side using advanced parameters.")
            with gr.Row():
                upload_img3 = gr.Image(type="numpy", label="Upload Image", height=DISPLAY_HEIGHT, sources=["upload", "webcam"])
                clean_img_sk = gr.Image(type="numpy", label="Sketch Output (Clean)", height=DISPLAY_HEIGHT)
                fgsm_img_sk = gr.Image(type="numpy", label="FGSM Output", height=DISPLAY_HEIGHT)
                pgd_img_sk = gr.Image(type="numpy", label="PGD Output", height=DISPLAY_HEIGHT)
                deepfool_img_sk = gr.Image(type="numpy", label="DeepFool Output", height=DISPLAY_HEIGHT)
                cw_img_sk = gr.Image(type="numpy", label="CW Output", height=DISPLAY_HEIGHT)

            details3 = gr.Textbox(label="Attack Details", lines=6)
            metrics_tbl3 = gr.HTML(label="Distortion Metrics (vs Clean Sketch)")

            with gr.Row():
                run_fgsm = gr.Checkbox(value=True, label="Run FGSM"); run_pgd = gr.Checkbox(value=True, label="Run PGD")
                run_deepfool = gr.Checkbox(value=True, label="Run DeepFool"); run_cw = gr.Checkbox(value=True, label="Run CW")

            with gr.Row():
                epsilon3 = gr.Slider(1, 96, value=32, step=1, label="Epsilon (/255)")
                steps3 = gr.Slider(1, 200, value=40, step=1, label="Steps")
                alpha3 = gr.Slider(0.0, 32.0, value=1.0, step=0.5, label="Alpha (/255, 0 = auto)")

            with gr.Row():
                random_start = gr.Checkbox(value=True, label="Random Start (PGD)")
                project = gr.Checkbox(value=True, label="Project to Epsilon Ball (PGD)")
                cw_c = gr.Slider(0.1, 20.0, value=1.5, step=0.1, label="CW c")
                deepfool_shift_threshold = gr.Slider(0.01, 0.8, value=0.35, step=0.01, label="DeepFool Shift Threshold")
                overshoot = gr.Slider(0.0, 0.4, value=0.15, step=0.01, label="DeepFool Overshoot")

            with gr.Row():
                image_size3 = gr.Slider(256, 1024, value=512, step=32, label="Model Input Size")
                blend_strength = gr.Slider(0.0, 1.0, value=0.85, step=0.01, label="Distortion Blend Strength")
                blur_radius = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Blur Radius")
                vis_boost = gr.Slider(1.0, 4.0, value=2.2, step=0.1, label="FGSM/DeepFool Visibility Boost")

            btn3 = gr.Button("Run Attack Analysis", variant="primary")
            inputs_t3 = [upload_img3, run_fgsm, run_pgd, run_deepfool, run_cw, epsilon3, steps3, alpha3, random_start, project, image_size3, cw_c, deepfool_shift_threshold, overshoot, blend_strength, blur_radius, vis_boost]
            outputs_t3 = [clean_img_sk, fgsm_img_sk, pgd_img_sk, deepfool_img_sk, cw_img_sk, details3, metrics_tbl3]
            btn3.click(fn=tab3_attack_analysis, inputs=inputs_t3, outputs=outputs_t3)

        # --- TAB 4: Security Firewall (Defense) ---
        with gr.TabItem("4. Security Firewall (Defense)"):
            gr.Markdown("Upload a single downloaded **Hacked Photo** (from Tab 2) to evaluate if our Firewall algorithm can scrub the noise payload and restore the Database Sketch.")
            with gr.Row():
                input_hacked4 = gr.Image(type="numpy", label="Upload HACKED Photo", height=300)
                clean_ref4 = gr.Image(type="numpy", label="Optional: Upload Clean Reference for Metrics", height=300)
            btn4 = gr.Button("Execute Security Firewall Defense", variant="primary")
            
            def_metrics = gr.HTML(label="Defense Effectiveness")
            
            with gr.Row():
                ruined_sk4 = gr.Image(type="numpy", label="Model Outcome WITHOUT Firewall", height=400)
                purified_ph4 = gr.Image(type="numpy", label="Firewall Processing Image (Scrubbed)", height=400)
                restored_sk4 = gr.Image(type="numpy", label="Model Outcome WITH Firewall (Restored!)", height=400)
                
            btn4.click(fn=tab4_defense_scenario, inputs=[input_hacked4, clean_ref4], outputs=[ruined_sk4, purified_ph4, restored_sk4, def_metrics])

if __name__ == "__main__":
    demo.launch(share=True)
