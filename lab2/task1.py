import cv2
import numpy as np
import sys

def initialize_camera(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def high_pass_fft(gray, radius=30):
    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # High-pass mask: 1 everywhere, 0 at center square/circle-ish region
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0.0

    # Apply mask
    fshift_filtered = fshift * mask

    # IFFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize to display
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

def main():
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    radius = 30
    print("Press 'q' to quit. Press '+' / '-' to change cutoff radius.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hp = high_pass_fft(gray, radius=radius)

        cv2.imshow("Original (Gray)", gray)
        cv2.imshow("High-pass (FFT)", hp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            radius = min(radius + 2, min(gray.shape)//2 - 1)
        elif key == ord('-'):
            radius = max(radius - 2, 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()