import time

import numpy as np


def lu(A):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)

    for k in range(n - 1):
        pivot = U[k, k]
        if pivot == 0:
            raise ValueError("zero pivot: LU without row swaps does not exist")

        column_below = U[k + 1:, k].copy()
        row_right = U[k, k + 1:]

        L[k + 1:, k] = column_below / pivot
        U[k + 1:, k + 1:] -= np.outer(column_below, row_right) / pivot
        U[k + 1:, k] = 0.0

    return L, U


def solve_lu(L, U, b):
    n = len(b)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x


def sherman_morrison(inverse, u, v):
    inverse_u = inverse @ u
    v_inverse = v @ inverse
    return inverse - np.outer(inverse_u, v_inverse) / (1 + v @ inverse_u)


def check_example_6_107():
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    L, U = lu(A)
    print("L =")
    print(L)
    print("U =")
    print(U)
    print("solution of Ax = (5, 13, 31):", solve_lu(L, U, np.array([5, 13, 31])))


def check_random_matrices(rng):
    all_close = True
    for _ in range(10):
        A = rng.normal(size=(6, 6))
        L, U = lu(A)
        b = rng.normal(size=6)
        if not np.allclose(L @ U, A):
            all_close = False
        if not np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b)):
            all_close = False
    print("L @ U == A and solve_lu == np.linalg.solve for random 6x6:", all_close)


def check_small_pivot():
    A = np.array([[1e-17, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])

    L, U = lu(A)
    print("without row swap:", solve_lu(L, U, b))
    print("U =")
    print(U)

    swapped_A = A[[1, 0]]
    swapped_b = b[[1, 0]]
    L, U = lu(swapped_A)
    print("after swapping the rows:", solve_lu(L, U, swapped_b))


def check_sherman_morrison(rng):
    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))

    start = time.perf_counter()
    inverse = np.linalg.inv(X.T @ X)
    for row in new_rows:
        inverse = sherman_morrison(inverse, row, row)
    update_time = time.perf_counter() - start

    start = time.perf_counter()
    current_X = X
    for row in new_rows:
        current_X = np.vstack([current_X, row])
        inverse_from_scratch = np.linalg.inv(current_X.T @ current_X)
    scratch_time = time.perf_counter() - start

    print("updated inverse equals inverse from scratch:", np.allclose(inverse, inverse_from_scratch))
    print(f"time with Sherman-Morrison updates: {update_time:.4f} s")
    print(f"time recomputing from scratch:       {scratch_time:.4f} s")


def main():
    rng = np.random.default_rng(0)

    print("--- Example 6.107 ---")
    check_example_6_107()
    print("\n--- random matrices ---")
    check_random_matrices(rng)
    print("\n--- tiny pivot ---")
    check_small_pivot()
    print("\n--- Sherman-Morrison ---")
    check_sherman_morrison(rng)


if __name__ == "__main__":
    main()
