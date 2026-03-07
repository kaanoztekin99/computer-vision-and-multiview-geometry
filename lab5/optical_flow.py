import os
import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_images(input_dir: str, img1_name: str, img2_name: str):
    img1_path = os.path.join(input_dir, img1_name)
    img2_path = os.path.join(input_dir, img2_name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Could not read first image: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read second image: {img2_path}")

    return img1, img2


def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement can help corner detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def detect_features(gray_img: np.ndarray, max_corners: int = 1000):
    points = cv2.goodFeaturesToTrack(
        gray_img,
        maxCorners=max_corners,
        qualityLevel=0.003,
        minDistance=5,
        blockSize=3,
        useHarrisDetector=False
    )

    if points is None:
        raise ValueError("No good feature points were detected in the first image.")

    return points


def compute_lk_flow(
    gray1: np.ndarray,
    gray2: np.ndarray,
    points1: np.ndarray,
    win_size=(21, 21),
    max_level=3,
    max_iter=30,
    eps=0.01
):
    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    )

    points2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, points1, None, **lk_params
    )

    return points2, status, error


def draw_detected_features(img: np.ndarray, points: np.ndarray, title_text="Detected Features"):
    vis = img.copy()

    for pt in points:
        x, y = pt.ravel()
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

    cv2.putText(
        vis, f"{title_text} | count={len(points)}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    return vis


def draw_tracks(img2, points1, points2, status, title_text=""):
    vis = img2.copy()

    if points2 is None or status is None:
        return vis

    good_new = points2[status.flatten() == 1]
    good_old = points1[status.flatten() == 1]

    for new_pt, old_pt in zip(good_new, good_old):
        a, b = new_pt.ravel()
        c, d = old_pt.ravel()

        a, b, c, d = int(a), int(b), int(c), int(d)

        cv2.line(vis, (c, d), (a, b), (0, 255, 0), 2)
        cv2.circle(vis, (a, b), 4, (0, 0, 255), -1)
        cv2.circle(vis, (c, d), 3, (255, 0, 0), -1)

    cv2.putText(
        vis, f"{title_text} | tracked={len(good_new)}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )

    return vis


def save_image(path: str, image: np.ndarray) -> None:
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def create_side_by_side(images, labels):
    labeled = []

    for img, label in zip(images, labels):
        temp = img.copy()
        cv2.putText(
            temp, label, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        labeled.append(temp)

    heights = [img.shape[0] for img in labeled]
    min_h = min(heights)

    resized = []
    for img in labeled:
        h, w = img.shape[:2]
        scale = min_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, min_h)))

    return cv2.hconcat(resized)


def print_tracking_stats(name, points1, status):
    total = len(points1)
    valid = int(np.sum(status)) if status is not None else 0
    print(f"{name}: detected={total}, tracked={valid}, lost={total - valid}")


def main():
    input_dir = "input"
    output_dir = "outputs"
    ensure_dir(output_dir)

    # For Lucas-Kanade
    img1_name = "3_1.jpeg"
    img2_name = "3_2.jpeg"

    img1, img2 = read_images(input_dir, img1_name, img2_name)

    gray1 = preprocess_gray(img1)
    gray2 = preprocess_gray(img2)

    points1 = detect_features(gray1, max_corners=1000)
    print(f"Detected feature count in first image: {len(points1)}")

    feature_vis = draw_detected_features(img1, points1, "Detected Features in Image 1")
    save_image(os.path.join(output_dir, "01_detected_features.jpg"), feature_vis)

    pair_vis = create_side_by_side(
        [img1, img2],
        ["Input Image 1", "Input Image 2"]
    )
    save_image(os.path.join(output_dir, "00_input_pair.jpg"), pair_vis)

    # Experiment 1: weak setting
    p2_low, st_low, err_low = compute_lk_flow(
        gray1, gray2, points1,
        win_size=(15, 15),
        max_level=0,
        max_iter=5,
        eps=0.03
    )
    print_tracking_stats("Low iterations + No pyramid", points1, st_low)
    vis_low = draw_tracks(img2, points1, p2_low, st_low, "Low iterations + No pyramid")
    save_image(os.path.join(output_dir, "02_lk_low_iterations_no_pyramid.jpg"), vis_low)

    # Experiment 2: more iterations
    p2_iter, st_iter, err_iter = compute_lk_flow(
        gray1, gray2, points1,
        win_size=(15, 15),
        max_level=0,
        max_iter=30,
        eps=0.01
    )
    print_tracking_stats("High iterations + No pyramid", points1, st_iter)
    vis_iter = draw_tracks(img2, points1, p2_iter, st_iter, "High iterations + No pyramid")
    save_image(os.path.join(output_dir, "03_lk_high_iterations_no_pyramid.jpg"), vis_iter)

    # Experiment 3: coarse-to-fine pyramid
    p2_pyr, st_pyr, err_pyr = compute_lk_flow(
        gray1, gray2, points1,
        win_size=(21, 21),
        max_level=3,
        max_iter=30,
        eps=0.01
    )
    print_tracking_stats("High iterations + Pyramid", points1, st_pyr)
    vis_pyr = draw_tracks(img2, points1, p2_pyr, st_pyr, "High iterations + Pyramid")
    save_image(os.path.join(output_dir, "04_lk_high_iterations_with_pyramid.jpg"), vis_pyr)

    comparison = create_side_by_side(
        [vis_low, vis_iter, vis_pyr],
        [
            "Low iter / No pyramid",
            "High iter / No pyramid",
            "High iter / Pyramid"
        ]
    )
    save_image(os.path.join(output_dir, "05_comparison_panel.jpg"), comparison)

    print("Processing completed successfully.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()