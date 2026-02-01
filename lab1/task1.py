import numpy as np

def fov_rad(sensor_width: float, focal_length: float) -> float:
    """
    Horizontal FOV (radians) for a camera:
        theta = 2 * arctan( w / (2f) )
    """
    if sensor_width <= 0:
        raise ValueError("sensor_width must be > 0")
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0")
    return 2.0 * np.arctan(sensor_width / (2.0 * focal_length))

def fov_deg(sensor_width: float, focal_length: float) -> float:
    return float(np.degrees(fov_rad(sensor_width, focal_length)))

def main():
    w = 6.4
    f = 4.0
    theta = fov_deg(w, f)
    print(f"sensor width w = {w} mm")
    print(f"focal length f = {f} mm")
    print(f"Horizontal FOV θ = {theta:.2f} degrees")

if __name__ == "__main__":
    main()