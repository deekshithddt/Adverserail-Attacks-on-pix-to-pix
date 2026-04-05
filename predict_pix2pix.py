import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "pix2pix_sketch_G.pth")


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = Down(1, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(512, 128)
        self.u3 = Up(256, 64)
        self.u4 = Up(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1, d3], 1))
        u3 = self.u3(torch.cat([u2, d2], 1))
        u4 = self.u4(torch.cat([u3, d1], 1))

        return torch.tanh(self.final(u4))


def load_generator(model_path, device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_sketch(model, img_path, device, image_size=512):
    img_color = Image.open(img_path).convert("RGB")
    img_gray = img_color.convert("L")
    orig_w, orig_h = img_color.size

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    input_tensor = transform(img_gray).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = model(input_tensor)

    out = fake.squeeze().cpu().numpy()
    out = (out + 1) / 2
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    if cv2 is not None:
        out = cv2.GaussianBlur(out, (5, 5), 0)
        out = cv2.normalize(out, None, 50, 255, cv2.NORM_MINMAX)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(out, -1, kernel)
        out = cv2.addWeighted(out, 0.9, np.full_like(out, 255), 0.1, 0)
        final_sketch = cv2.resize(out, (orig_w, orig_h))
    else:
        # PIL fallback when OpenCV is unavailable
        img = Image.fromarray(out)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=140, threshold=3))
        final_sketch = np.array(img.resize((orig_w, orig_h), Image.Resampling.LANCZOS))

    return img_color, final_sketch


def default_output_path(input_path):
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".png"
    return f"{base}_sketch{ext}"


def parse_args():
    parser = argparse.ArgumentParser(description="Pix2Pix sketch prediction")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL_PATH,
        help="Path to generator weights (.pth)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output sketch path (default: <input>_sketch.<ext>)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Model input size used during training (default: 512)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display input and output windows",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output or default_output_path(args.input)

    model = load_generator(args.model, device)
    input_image, sketch_image = predict_sketch(
        model, args.input, device=device, image_size=args.image_size
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(output_path, sketch_image)
    else:
        Image.fromarray(sketch_image).save(output_path)
    print(f"Model: {os.path.abspath(args.model)}")
    print(f"Input: {os.path.abspath(args.input)}")
    print(f"Output: {os.path.abspath(output_path)}")
    print(f"Device: {device}")

    if args.show:
        if cv2 is not None:
            input_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("Input", input_bgr)
            cv2.imshow("Pencil Sketch", sketch_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.title("Input")
                plt.imshow(input_image)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.title("Pencil Sketch")
                plt.imshow(sketch_image, cmap="gray")
                plt.axis("off")
                plt.show()
            except Exception:
                pass


if __name__ == "__main__":
    main()
