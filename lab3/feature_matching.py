import cv2
import os

def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    img1_path = os.path.join(input_dir, "image3.jpeg")
    img2_path = os.path.join(input_dir, "image4.jpeg")
    out_path = os.path.join(output_dir, "sift_feature_matching.png")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Check input/image1.jpeg and input/image2.jpeg paths")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #  SIFT detect + describe
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    #  FLANN KD-tree matcher
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #  KNN matching
    matches_knn = flann.knnMatch(desc1, desc2, k=2)

    #  Lowe ratio test
    ratio = 0.75
    good_matches = []
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    print(f"Good matches after ratio test: {len(good_matches)}")

    # visualize
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches[:60],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # save output
    cv2.imwrite(out_path, vis)
    print(f"Output saved to: {out_path}")

    # show
    cv2.imshow("SIFT Feature Matching", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
