import numpy as np
import matplotlib.pyplot as plt

# x1 = (x, y, z), x2 = (x+dx, y, z)
# Pinhole: x' = f * x / z  =>  |x2' - x1'| = f * dx / z
def projected_x(f: float, x: float, z: float) -> float:
    if z <= 0:
        raise ValueError("z must be > 0 (point must be in front of camera).")
    return f * (x / z)

def projected_distance(f: float, dx: float, z: float) -> float:
    if z <= 0:
        raise ValueError("z must be > 0")
    return abs(f * dx / z)

def curve_distance_vs_f(dx: float, z: float, f_values: np.ndarray) -> np.ndarray:
    f_values = np.asarray(f_values, dtype=float)
    if np.any(f_values <= 0):
        raise ValueError("All focal lengths must be > 0")
    return np.abs(f_values * dx / z)

def curve_distance_vs_z(dx: float, f: float, z_values: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    if np.any(z_values <= 0):
        raise ValueError("All depths z must be > 0")
    return np.abs(f * dx / z_values)


def main():
    dx = 100.0
    z  = 2000.0
    f0 = 35.0 

    d0 = projected_distance(f0, dx, z)
    print(f"dx = {dx}, z = {z}, f = {f0}  =>  |x2' - x1'| = {d0:.4f} (same unit as f)")

    f_values = np.linspace(5.0, 100.0, 400)
    dist_f = curve_distance_vs_f(dx, z, f_values)

    plt.figure()
    plt.plot(f_values, dist_f)
    plt.xlabel("Focal length f")
    plt.ylabel("Projected distance |x2' - x1'|")
    plt.title(f"Projected distance vs focal length (dx={dx}, z={z})")
    plt.grid(True)
    plt.show()

    z_values = np.linspace(200.0, 8000.0, 400)
    dist_z = curve_distance_vs_z(dx, f0, z_values)

    plt.figure()
    plt.plot(z_values, dist_z)
    plt.xlabel("Depth z")
    plt.ylabel("Projected distance |x2' - x1'|")
    plt.title(f"Projected distance vs depth (dx={dx}, f={f0})")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()