import matplotlib.pyplot as plt
import numpy as np

UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

LETTER_F = np.array([
    [0.2, 0.1], [0.4, 0.1], [0.4, 0.45], [0.65, 0.45], [0.65, 0.6],
    [0.4, 0.6], [0.4, 0.75], [0.8, 0.75], [0.8, 0.9], [0.2, 0.9],
])


def closed(points):
    return np.vstack([points, points[:1]])


def grid_lines(count=4):
    lines = []
    for t in np.linspace(0, 1, count + 1)[1:-1]:
        lines.append(np.array([[t, 0.0], [t, 1.0]]))
        lines.append(np.array([[0.0, t], [1.0, t]]))
    return lines


def draw_arrow(axis, end, color):
    axis.annotate("", xy=end, xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color=color, linewidth=2))


def plot_transformation(A, title="", axis=None):
    A = np.array(A, dtype=float)
    if axis is None:
        _, axis = plt.subplots(figsize=(5, 5))

    square_image = UNIT_SQUARE @ A.T
    letter_image = LETTER_F @ A.T

    for line in grid_lines():
        axis.plot(*line.T, color="lightgray", linewidth=0.8)
        axis.plot(*(line @ A.T).T, color="tab:blue", linewidth=0.8, alpha=0.6)

    axis.plot(*closed(UNIT_SQUARE).T, color="gray", linestyle="--")
    axis.fill(*square_image.T, color="tab:blue", alpha=0.15)
    axis.plot(*closed(square_image).T, color="tab:blue")

    axis.fill(*LETTER_F.T, color="gray", alpha=0.5)
    axis.fill(*letter_image.T, color="tab:orange", alpha=0.9)

    draw_arrow(axis, A[:, 0], "red")
    draw_arrow(axis, A[:, 1], "green")

    all_points = np.vstack([UNIT_SQUARE, square_image])
    lower = all_points.min(axis=0) - 0.5
    upper = all_points.max(axis=0) + 0.5
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.axhline(0, color="black", linewidth=0.5)
    axis.axvline(0, color="black", linewidth=0.5)
    axis.set_aspect("equal")
    axis.set_title(title)
    return axis


def signed_area(corners):
    x = corners[:, 0]
    y = corners[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return 0.5 * np.sum(x * y_next - x_next * y)


def main():
    figure_2_matrices = {
        "(a) stretch": np.array([[2, 0], [0, 1]]),
        "(b) rotation by 90°": np.array([[0, -1], [1, 0]]),
        "(c) reflection in the x-axis": np.array([[1, 0], [0, -1]]),
        "(d) shear": np.array([[1, 1], [0, 1]]),
    }

    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    for axis, (title, A) in zip(axes.flat, figure_2_matrices.items()):
        plot_transformation(A, title, axis)
    figure.tight_layout()

    singular_matrix = np.array([[1, 2], [2, 4]])
    plot_transformation(singular_matrix, "singular matrix [[1, 2], [2, 4]]")

    all_matrices = dict(figure_2_matrices)
    all_matrices["singular"] = singular_matrix
    all_matrices["Figure 4"] = np.array([[3, 1], [1, 2]])

    print(f"{'matrix':<30}{'signed area':>12}{'det(A)':>10}")
    for name, A in all_matrices.items():
        area = signed_area(UNIT_SQUARE @ A.T)
        determinant = np.linalg.det(A)
        print(f"{name:<30}{area:>12.3f}{determinant:>10.3f}")

    plt.show()


if __name__ == "__main__":
    main()
