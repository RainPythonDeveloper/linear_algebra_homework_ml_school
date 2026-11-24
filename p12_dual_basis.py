import matplotlib.pyplot as plt
import numpy as np


def dual_basis(P):
    return np.linalg.inv(P)


def vandermonde(points, degree):
    return np.vander(points, degree + 1, increasing=True)


def lagrange_basis(nodes):
    V = vandermonde(nodes, len(nodes) - 1)
    return np.linalg.inv(V)


def interpolate(f, nodes):
    lagrange_polynomials = lagrange_basis(nodes)
    return lagrange_polynomials @ f(nodes)


def evaluate(coefficients, points):
    return vandermonde(points, len(coefficients) - 1) @ coefficients


def plot_interpolation(axis, f, nodes, title):
    coefficients = interpolate(f, nodes)
    t = np.linspace(nodes[0], nodes[-1], 1000)
    axis.plot(t, f(t), label="f")
    axis.plot(t, evaluate(coefficients, t), label="interpolation p")
    axis.plot(nodes, f(nodes), "o", color="black", label="nodes")
    axis.set_title(title)
    axis.legend()


def main():
    rng = np.random.default_rng(0)

    P = rng.normal(size=(5, 5))
    rows = dual_basis(P)
    print("r_i b_j is the identity matrix:", np.allclose(rows @ P, np.eye(5)))

    x = rng.normal(size=5)
    reconstructed = sum((rows[i] @ x) * P[:, i] for i in range(5))
    print("x = sum (r_i x) b_i:", np.allclose(x, reconstructed))

    nodes = np.array([-1.0, 0.0, 1.0])
    lagrange_polynomials = lagrange_basis(nodes)
    print("\nLagrange polynomials for nodes -1, 0, 1 (coefficients of 1, x, x^2):")
    for node, coefficients in zip(nodes, lagrange_polynomials.T):
        print(f"L_{node:+.0f}:", coefficients)

    values = np.array([1.0, 2.0, 5.0])
    print("polynomial with values 1, 2, 5 at -1, 0, 1:", lagrange_polynomials @ values)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    nodes = np.linspace(-1, 1, 9)
    plot_interpolation(axes[0], lambda t: np.sin(3 * t), nodes, "sin(3t), 9 nodes on [-1, 1]")

    nodes = np.linspace(-5, 5, 21)
    plot_interpolation(axes[1], lambda t: 1 / (1 + t ** 2), nodes, "1 / (1 + t^2), 21 nodes on [-5, 5]")
    axes[1].set_ylim(-2, 2)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
