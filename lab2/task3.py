import cv2
import numpy as np
from pathlib import Path

def harris_corners_manual(
    img_bgr,
    k=0.04,
    sigma=1.5,
    window_sigma=1.5,
    thresh_ratio=0.01,
    nms_ksize=3
):

    # Grayscale + float
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Smoothing (Gaussian)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Gradients (Sobel)
    Ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Sxx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma)

    # Harris response R = det(M) - k * trace(M)^2
    detM = (Sxx * Syy) - (Sxy * Sxy)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)

    # Threshold
    Rmax = np.max(R)
    if Rmax <= 0:
        # corner çıkmayabilir (çok düz görüntü vb.)
        return R, np.zeros_like(R, dtype=np.uint8), []

    thresh = thresh_ratio * Rmax
    corners_mask = (R > thresh).astype(np.uint8)

    # Non-maximum suppression (NMS)
    kernel = np.ones((nms_ksize, nms_ksize), np.uint8)
    R_dilated = cv2.dilate(R, kernel)
    nms_mask = (R == R_dilated) & (R > thresh)

    ys, xs = np.where(nms_mask)
    points = list(zip(xs.tolist(), ys.tolist()))

    out_mask = np.zeros_like(gray, dtype=np.uint8)
    out_mask[ys, xs] = 255

    return R, out_mask, points


def draw_corners(img_bgr, points, radius=3, color=(0, 0, 255)):
    out = img_bgr.copy()
    for (x, y) in points:
        cv2.circle(out, (x, y), radius, color, -1)
    return out


def run_on_folder(input_dir="images", output_dir="outputs/task3_harris"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"No images found in {input_dir.resolve()}")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read: {img_path.name}")
            continue

        R, mask, pts = harris_corners_manual(
            img,
            k=0.04,
            sigma=1.2,
            window_sigma=1.5,
            thresh_ratio=0.0001,
            nms_ksize=3
        )

        overlay = draw_corners(img, pts, radius=3)

        # Save outputs
        cv2.imwrite(str(output_dir / f"{img_path.stem}_harris_overlay.png"), overlay)
        # R'yi görmek için normalize ederek kaydedelim
        R_pos = np.maximum(R, 0)
        R_log = np.log1p(R_pos)
        R_vis = cv2.normalize(R_log, None, 0, 255, cv2.NORM_MINMAX)
        R_vis = R_vis.astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_harris_response.png"), R_vis)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_harris_mask.png"), mask)

        print(f"{img_path.name}: corners={len(pts)} saved to {output_dir}")

if __name__ == "__main__":
    run_on_folder("inputs", "outputs/")