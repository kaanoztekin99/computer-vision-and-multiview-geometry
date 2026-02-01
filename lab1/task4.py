import cv2
import numpy as np


def open_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def create_stereo_sgbm(num_disparities=128, block_size=7):
    # num_disparities must be divisible by 16
    num_disparities = max(16, (num_disparities // 16) * 16)
    block_size = max(3, block_size | 1)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo


def disparity_to_relative_depth(disp, eps=1e-6):
    """
    Relative depth from disparity (no calibration):
        depth ~ 1 / disparity
    disp: float32 disparity values (can be <=0 in invalid areas)
    """
    disp_pos = np.maximum(disp, eps)
    depth = 1.0 / disp_pos
    # mark invalid disparity as 0 depth for display
    depth[disp <= 0] = 0.0
    return depth


def normalize_to_uint8(img):
    """Normalize a float image to 0..255 uint8 for visualization."""
    vis = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return vis.astype(np.uint8)


def main():
    capL = open_camera(0)
    capR = open_camera(1)

    if capL is None or capR is None:
        print("Error: Stereo depth needs TWO cameras (e.g., indices 0 and 1).")
        print("If you only have one webcam, you need a deep model (MiDaS etc.).")
        return

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("numDisp (x16)", "Controls", 8, 20, lambda x: None)  # 8*16=128
    cv2.createTrackbar("blockSize", "Controls", 7, 21, lambda x: None)       # odd
    cv2.createTrackbar("smooth (0/1)", "Controls", 1, 1, lambda x: None)     # apply median blur

    print("Stereo depth estimation started. Press 'q' to quit.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Error: Can't read from one of the cameras.")
            break

        # Ensure same size
        h, w = frameL.shape[:2]
        frameR = cv2.resize(frameR, (w, h), interpolation=cv2.INTER_LINEAR)

        nd = cv2.getTrackbarPos("numDisp (x16)", "Controls") * 16
        bs = cv2.getTrackbarPos("blockSize", "Controls")
        smooth = cv2.getTrackbarPos("smooth (0/1)", "Controls")

        stereo = create_stereo_sgbm(num_disparities=nd, block_size=bs)

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Disparity (fixed-point /16)
        disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        # Optional smoothing for nicer display
        if smooth == 1:
            disp = cv2.medianBlur(disp, 5)

        # Visualize disparity
        disp_vis = normalize_to_uint8(disp)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Relative depth (inverse disparity)
        depth = disparity_to_relative_depth(disp)
        depth_vis = normalize_to_uint8(depth)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        cv2.imshow("Left", frameL)
        # different colormaps
        cv2.imshow("Disparity (JET)", disp_color)
        cv2.imshow("Relative Depth (TURBO)", depth_color)  # brighter != always nearer; it's normalized

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()