import os
import cv2
import numpy as np
from sklearn.cluster import MeanShift


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_image_from_input(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(input_dir, file_name)
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            print(f"Loaded image: {image_path}")
            return image, file_name

    raise FileNotFoundError("No valid image found in the input/ folder.")


def resize_if_needed(image, max_dimension=300):
    """
    MeanShift is very expensive on large images.
    """
    h, w = image.shape[:2]
    max_current_dim = max(h, w)

    if max_current_dim <= max_dimension:
        return image

    scale = max_dimension / max_current_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"Image resized from {w}x{h} to {new_w}x{new_h} for faster MeanShift processing.")
    return resized


def build_feature_space(image, spatial_radius, color_radius):
    """
    Creates a 5D feature vector for each pixel:
    [x/spatial_radius, y/spatial_radius, L/color_radius, a/color_radius, b/color_radius]
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = image.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    features = np.zeros((h * w, 5), dtype=np.float32)
    features[:, 0] = xx.reshape(-1) / spatial_radius
    features[:, 1] = yy.reshape(-1) / spatial_radius
    features[:, 2] = image_lab[:, :, 0].reshape(-1) / color_radius
    features[:, 3] = image_lab[:, :, 1].reshape(-1) / color_radius
    features[:, 4] = image_lab[:, :, 2].reshape(-1) / color_radius

    return features


def reconstruct_image(image, labels):
    """
    Reconstruct segmented image by assigning each cluster
    the mean BGR color of the pixels in that cluster.
    """
    h, w = image.shape[:2]
    flat_img = image.reshape(-1, 3).astype(np.float32)

    segmented = np.zeros_like(flat_img)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        mean_color = np.mean(flat_img[mask], axis=0)
        segmented[mask] = mean_color

    segmented = segmented.reshape(h, w, 3).astype(np.uint8)
    return segmented


def run_meanshift(image, spatial_radius, color_radius, bandwidth=1.0):
    """
    Runs MeanShift clustering in the 5D feature space.
    """
    features = build_feature_space(image, spatial_radius, color_radius)

    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        cluster_all=True
    )

    labels = ms.fit_predict(features)
    segmented = reconstruct_image(image, labels)
    num_clusters = len(np.unique(labels))

    return segmented, num_clusters


def save_segmented_only(segmented, output_path):
    cv2.imwrite(output_path, segmented)


def main():
    input_dir = "input"
    output_dir = "outputs"

    ensure_dir(output_dir)

    image, image_name = load_image_from_input(input_dir)
    print(f"Original image shape: {image.shape[1]}x{image.shape[0]}")

    # MeanShift on full-resolution phone images is usually too slow, that's why we need image resizing
    image = resize_if_needed(image, max_dimension=300)
    print(f"Processing image shape: {image.shape[1]}x{image.shape[0]}")

    parameter_sets = [
        (8, 12),
        (16, 18),
        (24, 28),
        (40, 36),
        (56, 45)
    ]

    for i, (spatial_r, color_r) in enumerate(parameter_sets, start=1):
        print(f"\nRunning MeanShift for spatial_radius={spatial_r}, color_radius={color_r}")

        segmented, clusters = run_meanshift(
            image=image,
            spatial_radius=spatial_r,
            color_radius=color_r,
            bandwidth=1.0
        )

        segmented_name = f"meanshift_{i}_sr{spatial_r}_cr{color_r}.png"
        segmented_path = os.path.join(output_dir, segmented_name)

        save_segmented_only(segmented, segmented_path)

        print(f"Saved segmented image: {segmented_path}")
        print(f"Number of clusters: {clusters}")


if __name__ == "__main__":
    main()