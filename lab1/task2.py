import numpy as np
import matplotlib.pyplot as plt
from task1 import fov_rad

def fov_curve_deg(sensor_width: float, focal_lengths: np.ndarray) -> np.ndarray:
    focal_lengths = np.asarray(focal_lengths, dtype=float)
    if np.any(focal_lengths <= 0):
        raise ValueError("All focal lengths must be > 0")
    return np.degrees([fov_rad(sensor_width, fl) for fl in focal_lengths])

def plot_two_camera_fov(
    w1: float = 6.4,     
    w2: float = 36.0,    
    f_min: float = 1.5,
    f_max: float = 80.0,
    n: int = 400
):
    f = np.linspace(f_min, f_max, n)
    theta1 = fov_curve_deg(w1, f)
    theta2 = fov_curve_deg(w2, f)

    plt.figure()
    plt.plot(f, theta1, label=f"Camera 1 (w={w1} mm)")
    plt.plot(f, theta2, label=f"Camera 2 (w={w2} mm)")
    plt.xlabel("Focal length f (mm)")
    plt.ylabel("Horizontal FOV θ (degrees)")
    plt.title("Field of View vs Focal Length for Different Sensor Widths")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    plot_two_camera_fov(w1=6.4, w2=36.0, f_min=1.5, f_max=80.0)

if __name__ == "__main__":
    main()