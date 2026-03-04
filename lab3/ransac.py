import os
import cv2
import numpy as np

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sift_match(img1, img2, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        return kp1, kp2, [], None, None

    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_knn = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    return kp1, kp2, good, desc1, desc2

def ransac_filter(kp1, kp2, matches, reproj_thresh=3.0):
    """
    Estimate homography using RANSAC and return inlier matches.
    reproj_thresh: pixel threshold for considering a match as inlier.
    """
    if len(matches) < 4:
        return None, [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=reproj_thresh)
    if mask is None:
        return None, [], None

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]

    return H, inlier_matches, mask

def main():
    input_dir = "input"
    output_dir = "output"
    ensure_dir(output_dir)

    img1_path = os.path.join(input_dir, "image3.jpeg")
    img2_path = os.path.join(input_dir, "image4.jpeg")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Check input/image3.jpeg and input/image4.jpeg")

    # SIFT + ratio test
    kp1, kp2, good_matches, _, _ = sift_match(img1, img2, ratio=0.75)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")
    print(f"Matches after ratio test: {len(good_matches)}")

    # RANSAC outlier rejection (Homography)
    H, inlier_matches, inlier_mask = ransac_filter(kp1, kp2, good_matches, reproj_thresh=3.0)
    if H is None:
        print("Not enough matches or homography could not be estimated.")
        return

    print(f"Inliers after RANSAC: {len(inlier_matches)} / {len(good_matches)}")

    # Draw only inliers
    vis_inliers = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        sorted(inlier_matches, key=lambda m: m.distance)[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    out_matches = os.path.join(output_dir, "sift_matches_ransac_inliers.png")
    cv2.imwrite(out_matches, vis_inliers)
    print(f"Saved inlier matches to: {out_matches}")

    cv2.imshow("RANSAC Inlier Matches (Homography)", vis_inliers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Warp img1 onto img2 using H
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w2, h2))
    out_warp = os.path.join(output_dir, "img1_warped_to_img2.png")
    cv2.imwrite(out_warp, warped)
    print(f"Saved warped image to: {out_warp}")

if __name__ == "__main__":
    main()
