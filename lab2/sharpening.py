import cv2
import numpy as np
from pathlib import Path

def sharpen_image(img):
    """
    Apply spatial-domain sharpening using a convolution kernel.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def run_on_folder(input_dir="inputs"):
    input_dir = Path(input_dir)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print("No images found.")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        sharpened = sharpen_image(img)

        out_path = input_dir / f"{img_path.stem}_sharpened{img_path.suffix}"
        cv2.imwrite(str(out_path), sharpened)

        print(f"Sharpened saved: {out_path.name}")

if __name__ == "__main__":
    run_on_folder("inputs")