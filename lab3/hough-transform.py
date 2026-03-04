import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_hough_accumulator(points: np.ndarray, num_theta: int, num_r: int):
    """
    Build Hough accumulator for line representation:
        r = x cos(theta) + y sin(theta)

    Returns:
        acc (num_r, num_theta)
        thetas (num_theta,)
        rs (num_r,)
        r_min, r_max
    """
    thetas = np.linspace(0, np.pi, num_theta, endpoint=False)

    max_r = np.sqrt((points[:, 0] ** 2 + points[:, 1] ** 2).max())
    r_min, r_max = -max_r, max_r
    rs = np.linspace(r_min, r_max, num_r)

    acc = np.zeros((num_r, num_theta), dtype=np.int32)

    for (x, y) in points:
        r_curve = x * np.cos(thetas) + y * np.sin(thetas)
        r_idx = np.round((r_curve - r_min) / (r_max - r_min) * (num_r - 1)).astype(int)
        r_idx = np.clip(r_idx, 0, num_r - 1)
        acc[r_idx, np.arange(num_theta)] += 1

    return acc, thetas, rs, r_min, r_max


def find_peak(acc: np.ndarray, thetas: np.ndarray, rs: np.ndarray):
    peak_r_idx, peak_theta_idx = np.unravel_index(np.argmax(acc), acc.shape)
    return rs[peak_r_idx], thetas[peak_theta_idx], acc[peak_r_idx, peak_theta_idx]


def plot_xy_with_line(points, r_star, theta_star, output_dir, fname="xy_space_with_best_line.png"):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])

    for (x, y) in points:
        plt.text(x + 0.05, y + 0.05, f"({x},{y})")

    plt.title("Points in (x,y) space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    x_vals = np.linspace(points[:, 0].min() - 1, points[:, 0].max() + 1, 400)

    if abs(np.sin(theta_star)) > 1e-9:
        y_vals = (r_star - x_vals * np.cos(theta_star)) / np.sin(theta_star)
        plt.plot(x_vals, y_vals, linewidth=2)
    else:
        x0 = r_star / np.cos(theta_star)
        plt.axvline(x0, linewidth=2)

    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()


def plot_hough_curves(points, thetas, r_star, theta_star, output_dir, fname="hough_space_curves.png"):
    plt.figure()
    for (x, y) in points:
        r_curve = x * np.cos(thetas) + y * np.sin(thetas)
        plt.plot(thetas, r_curve)

    plt.scatter([theta_star], [r_star], s=80)
    plt.title("Hough Space (theta vs r)")
    plt.xlabel("theta (rad)")
    plt.ylabel("r")
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()


def plot_accumulator(acc, thetas, r_min, r_max, output_dir, fname="hough_accumulator.png"):
    plt.figure()

    im = plt.imshow(
        acc,
        aspect="auto",
        origin="lower",
        extent=[thetas.min(), thetas.max(), r_min, r_max],
        cmap="hot",          # (alternative: "inferno", "magma")
        interpolation="nearest"
    )

    plt.title("Hough Accumulator")
    plt.xlabel("theta (rad)")
    plt.ylabel("r")
    plt.colorbar(im, label="Votes")

    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()


def main():
    output_dir = "output"
    ensure_dir(output_dir)

    points_A = np.array([(2, 2), (3, 1.5), (6, 0)], dtype=float)
    points_B = np.array([(2, 2), (5, 3), (6, 0)], dtype=float)

    # choosing a set of points
    points = points_B

    num_theta = 1800
    num_r = 1500

    acc, thetas, rs, r_min, r_max = build_hough_accumulator(points, num_theta, num_r)
    r_star, theta_star, votes = find_peak(acc, thetas, rs)

    print(f"Best (r, theta) = ({r_star:.4f}, {theta_star:.4f}), votes={votes}/{len(points)}")

    plot_xy_with_line(points, r_star, theta_star, output_dir)
    plot_hough_curves(points, thetas, r_star, theta_star, output_dir)
    plot_accumulator(acc, thetas, r_min, r_max, output_dir)


if __name__ == "__main__":
    main()
