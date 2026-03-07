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
        raise FileNotFoundError(f"Could not read image: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read image: {img2_path}")

    return img1, img2


def detect_and_describe_sift(image, nfeatures=2000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features_knn(desc1, desc2, ratio=0.75):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def draw_matches(img1, kp1, img2, kp2, matches, max_to_draw=100):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_to_draw]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def points_from_matches(kp1, kp2, matches):
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def normalize_points(points):
    """
    Hartley normalization:
    Shift points to zero mean and scale so that mean distance to origin is sqrt(2).
    """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    dist = np.sqrt(np.sum(shifted ** 2, axis=1))
    mean_dist = np.mean(dist)

    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    points_h = np.column_stack([points, np.ones(points.shape[0])])
    norm_h = (T @ points_h.T).T
    norm_points = norm_h[:, :2] / norm_h[:, 2:3]

    return norm_points, T


def compute_homography_dlt(src_pts, dst_pts):
    """
    Compute homography H such that:
        dst ~ H * src
    using normalized DLT + SVD.
    Requires at least 4 correspondences.
    """
    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)

    A = []
    for (x, y), (xp, yp) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


def apply_homography(H, points):
    points_h = np.column_stack([points, np.ones(points.shape[0])])
    projected_h = (H @ points_h.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    return projected


def ransac_homography(src_pts, dst_pts, iterations=2000, threshold=4.0, seed=42):
    """
    Robust homography estimation with RANSAC.
    Returns:
        best_H, best_inlier_mask
    """
    if src_pts.shape[0] != dst_pts.shape[0]:
        raise ValueError("Source and destination point counts must match.")
    if src_pts.shape[0] < 4:
        raise ValueError("At least 4 matches are required for homography.")

    rng = np.random.default_rng(seed)
    num_points = src_pts.shape[0]

    best_H = None
    best_inlier_mask = None
    best_inlier_count = 0
    best_error = np.inf

    for _ in range(iterations):
        sample_idx = rng.choice(num_points, size=4, replace=False)
        src_sample = src_pts[sample_idx]
        dst_sample = dst_pts[sample_idx]

        try:
            H_candidate = compute_homography_dlt(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue

        projected = apply_homography(H_candidate, src_pts)
        errors = np.linalg.norm(projected - dst_pts, axis=1)
        inlier_mask = errors < threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count < 4:
            continue

        mean_error = np.mean(errors[inlier_mask]) if inlier_count > 0 else np.inf

        if (inlier_count > best_inlier_count) or (
            inlier_count == best_inlier_count and mean_error < best_error
        ):
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_error = mean_error
            best_H = H_candidate

    if best_H is None or best_inlier_mask is None:
        raise RuntimeError("RANSAC failed to estimate a valid homography.")

    # Recompute H using all inliers
    refined_H = compute_homography_dlt(src_pts[best_inlier_mask], dst_pts[best_inlier_mask])
    return refined_H, best_inlier_mask


def draw_inlier_matches(img1, kp1, img2, kp2, matches, inlier_mask, max_to_draw=150):
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:max_to_draw]

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def warp_and_stitch(img1, img2, H_1_to_2):
    """
    Warp img1 into img2 coordinate system and create a simple panorama.
    Uses inverse warping internally via cv2.warpPerspective.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.array([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ], dtype=np.float64)

    corners_img2 = np.array([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ], dtype=np.float64)

    warped_corners_img1 = apply_homography(H_1_to_2, corners_img1)
    all_corners = np.vstack([warped_corners_img1, corners_img2])

    x_min, y_min = np.floor(np.min(all_corners, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(all_corners, axis=0)).astype(int)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    pano_width = x_max - x_min
    pano_height = y_max - y_min

    warped_img1 = cv2.warpPerspective(img1, T @ H_1_to_2, (pano_width, pano_height))

    panorama = warped_img1.copy()
    panorama[ty:ty + h2, tx:tx + w2] = img2

    # Optional simple overlay handling:
    # If you want slightly better blending than hard overwrite:
    mask1 = np.any(warped_img1 > 0, axis=2)
    mask2 = np.zeros((pano_height, pano_width), dtype=bool)
    mask2[ty:ty + h2, tx:tx + w2] = True

    overlap = mask1 & mask2

    # Average in overlap region
    if np.any(overlap):
        img2_canvas = np.zeros_like(panorama)
        img2_canvas[ty:ty + h2, tx:tx + w2] = img2

        panorama = panorama.astype(np.float32)
        img2_canvas = img2_canvas.astype(np.float32)

        panorama[overlap] = 0.5 * panorama[overlap] + 0.5 * img2_canvas[overlap]
        panorama[~mask1 & mask2] = img2_canvas[~mask1 & mask2]
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    return warped_img1, panorama


def main():
    input_dir = "input"
    output_dir = "output"
    ensure_dir(output_dir)

    img1_name = "foto1.jpeg"
    img2_name = "foto2.jpeg"

    img1, img2 = read_images(input_dir, img1_name, img2_name)

    print("Detecting SIFT features...")
    kp1, desc1 = detect_and_describe_sift(img1, nfeatures=2500)
    kp2, desc2 = detect_and_describe_sift(img2, nfeatures=2500)

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    if desc1 is None or desc2 is None:
        raise RuntimeError("Could not compute descriptors for one or both images.")

    print("Matching features...")
    good_matches = match_features_knn(desc1, desc2, ratio=0.75)
    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches to compute homography.")

    raw_match_vis = draw_matches(img1, kp1, img2, kp2, good_matches, max_to_draw=100)
    cv2.imwrite(os.path.join(output_dir, "01_raw_matches.jpg"), raw_match_vis)

    pts1, pts2 = points_from_matches(kp1, kp2, good_matches)

    print("Running RANSAC + DLT(SVD) homography estimation...")
    H_1_to_2, inlier_mask = ransac_homography(
        pts1, pts2,
        iterations=2500,
        threshold=4.0,
        seed=42
    )

    inlier_count = int(np.sum(inlier_mask))
    print(f"Inliers: {inlier_count} / {len(good_matches)}")
    print("Estimated homography (img1 -> img2):")
    print(H_1_to_2)

    inlier_vis = draw_inlier_matches(img1, kp1, img2, kp2, good_matches, inlier_mask, max_to_draw=150)
    cv2.imwrite(os.path.join(output_dir, "02_inlier_matches.jpg"), inlier_vis)

    warped_img1, panorama = warp_and_stitch(img1, img2, H_1_to_2)

    cv2.imwrite(os.path.join(output_dir, "03_warped_image1.jpg"), warped_img1)
    cv2.imwrite(os.path.join(output_dir, "04_panorama.jpg"), panorama)

    print("Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()