# Adversarial Attacks on Pix2Pix

This project explores how a pix2pix-style image-to-image model behaves under adversarial perturbations.

The main idea is:

- take an input image
- generate a sketch using a trained pix2pix generator
- study how small input perturbations can change the generated sketch output

The repository includes both normal sketch generation and an interface for adversarial attack experiments.

## What This Project Does

This project uses a trained generator model stored in `pix2pix_sketch_G.pth` to convert an input image into a grayscale sketch-like output.

It provides two main workflows:

1. Normal inference
   Generate a sketch from an image using a command-line script or a Gradio web interface.

2. Adversarial analysis
   Compare clean model output with attacked output using methods such as FGSM, PGD, DeepFool, and CW through a Gradio interface.

## Project Structure

```text
Adversarial-Attacks-on-pix-to-pix/
|-- README.md
|-- .gitignore
|-- predict_pix2pix.py
|-- gradio_predict.py
|-- gradio_attack_pix2pix.py
|-- gradio_export_adv.py
|-- pix2pix_sketch_G.pth
|-- pixtopix (1).ipynb
|-- test.jpg
|-- test2.jpg
|-- test3.jpg
|-- test_sketch.jpg
|-- fgsm_outputs/              # generated attack outputs, ignored by git
|-- .venv/                     # local virtual environment, ignored by git
|-- __pycache__/               # Python cache, ignored by git
`-- FS2K/                      # local dataset folder, ignored by git
```

## File Explanation

### `predict_pix2pix.py`

This is the core inference script.

It contains:

- the generator network definition
- model loading logic
- image preprocessing and postprocessing
- command-line inference support

Use this file when you want to generate a sketch from a single image directly from the terminal.

### `gradio_predict.py`

This file launches a Gradio app for normal sketch generation.

It allows you to:

- upload an image
- run the pix2pix model
- view the generated sketch in the browser

This is the easiest entry point for quick testing.

### `gradio_attack_pix2pix.py`

This file launches a Gradio app for adversarial attack comparison.

It is designed to show:

- clean sketch output
- FGSM output
- PGD output
- DeepFool output
- CW output
- simple distortion metrics such as MSE, PSNR, and SSIM

### `gradio_export_adv.py`

This file launches a dedicated Gradio app to generate and download raw adversarial examples for all attacks simultaneously. It is designed to export the raw, uncompressed grayscale adversarial image.
This is particularly important because web browsers often inadvertently destroy microscopic adversarial noise through image compression when uploading photos directly to web UIs.

*(Note: All attack utilities depend on `fgsm_attack_pix2pix.py`, which contains the zero-gradient trap fix allowing the math to escape a local minimum when the input and target are identical).*

### `pix2pix_sketch_G.pth`

This is the trained generator model checkpoint used for inference.

### `pixtopix (1).ipynb`

This notebook likely contains the training or experimentation workflow used during development.

### Sample images

- `test.jpg`
- `test2.jpg`
- `test3.jpg`
- `test_sketch.jpg`

These are example files that can be used to test the model.

### `FS2K/`

This is the dataset folder used locally during development.

It is intentionally excluded from GitHub so that the repository stays smaller and easier to clone.

## Model Architecture

The model in `predict_pix2pix.py` is a lightweight encoder-decoder generator with skip connections, similar in spirit to a U-Net / pix2pix generator.

High-level flow:

- input image is converted to grayscale
- image is resized to the configured model input size
- image is normalized to `[-1, 1]`
- generator predicts a sketch image
- output is denormalized and postprocessed

The output is a single-channel sketch-style image.

## How Inference Works

The normal prediction pipeline is:

1. Load the trained generator weights
2. Read the input image
3. Convert it to grayscale
4. Resize it to the configured size, usually `512 x 512`
5. Run the generator
6. Convert the output back into a visible sketch image
7. Save or display the result

## Setup

Create and activate a virtual environment, then install the required packages.

Typical packages used by this project are:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `gradio`
- `opencv-python` (optional, but used when available)

Example:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install torch torchvision numpy pillow gradio opencv-python
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Git Bash:

```bash
source .venv/Scripts/activate
```

## How to Run

### 1. Run normal sketch prediction from CLI

```bash
python predict_pix2pix.py --input test.jpg
```

Optional arguments:

- `--model` to specify a different `.pth` file
- `--output` to control where the sketch is saved
- `--image-size` to change model input size
- `--cpu` to force CPU inference
- `--show` to display the result

Example:

```bash
python predict_pix2pix.py --input test.jpg --output output_sketch.png --show
```

### 2. Run the Gradio sketch generator UI

```bash
python gradio_predict.py
```

This opens a browser-based interface where you can upload an image and generate a sketch.

### 3. Run the Gradio adversarial attack UI

```bash
python gradio_attack_pix2pix.py
```

### 4. Export pure adversarial images

```bash
python gradio_export_adv.py
```

This app produces downloadable adversarial images. 
**Note on Web Browsers:** When using web interfaces (like Gradio), browsers often compress uploaded images automatically. This compression accidentally destroys the delicate adversarial noise! To truly test the attacked image, you must download it from this export app and run it directly through the terminal to bypass browser compression:

```bash
python predict_pix2pix.py --input <downloaded_adv_image>.png --show
```

## Output Files

Normal inference saves generated sketches next to the input image unless another output path is provided.

Adversarial results are typically saved inside `fgsm_outputs/` when attack scripts are used.

## Current Repository State

At the moment, the repository includes:

- the trained model
- inference code
- the basic Gradio prediction app
- the adversarial comparison app
- sample test images

The `FS2K` dataset is not part of the GitHub-tracked project contents.

The adversarial attack interface is fully implemented and reproducible, with all attack methods (FGSM, PGD, DeepFool, CW) safely contained in `fgsm_attack_pix2pix.py`.

## Summary

In short, this project is a pix2pix-based sketch generation system with an added adversarial analysis workflow.

It is useful for:

- image-to-sketch generation experiments
- testing robustness of generative models
- comparing clean and adversarial outputs
- demonstrating attack methods on image-to-image networks
