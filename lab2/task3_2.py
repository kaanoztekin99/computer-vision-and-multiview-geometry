import cv2
import numpy as np
from pathlib import Path

def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    raise RuntimeError("SIFT not available. Install opencv-contrib-python.")

def sift_outputs(img_bgr, max_keypoints=800):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sift = create_sift()
    try:
        sift = cv2.SIFT_create(nfeatures=max_keypoints)
    except TypeError:
        pass

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # RICH KEYPOINT VISUALIZATION
    rich_vis = cv2.drawKeypoints(
        img_bgr, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # HARRIS-LIKE BINARY MASK
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if keypoints is not None:
        for kp in keypoints:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                mask[y, x] = 255

    # OVERLAY
    overlay = img_bgr.copy()
    ys, xs = np.where(mask == 255)
    for (x, y) in zip(xs, ys):
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)

    return rich_vis, mask, overlay, keypoints, descriptors

def run_on_folder(input_dir="images", output_dir="outputs/task3_sift"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print("No images found.")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rich, mask, overlay, kps, desc = sift_outputs(img)

        cv2.imwrite(str(output_dir / f"{img_path.stem}_sift_rich.png"), rich)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_sift_mask.png"), mask)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_sift_overlay.png"), overlay)

        n_kp = 0 if kps is None else len(kps)
        print(f"{img_path.name}: SIFT keypoints = {n_kp}")

if __name__ == "__main__":
    run_on_folder("inputs", "outputs/task3_sift_all")