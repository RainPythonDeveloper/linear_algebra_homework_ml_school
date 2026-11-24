import numpy as np


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    num_rows, num_cols = R.shape
    pivots = []
    pivot_row = 0

    for col in range(num_cols):
        if pivot_row == num_rows:
            break

        best_row = pivot_row + np.argmax(np.abs(R[pivot_row:, col]))
        if abs(R[best_row, col]) < tol:
            continue

        R[[pivot_row, best_row]] = R[[best_row, pivot_row]]
        R[pivot_row] = R[pivot_row] / R[pivot_row, col]

        for row in range(num_rows):
            if row != pivot_row:
                R[row] = R[row] - R[row, col] * R[pivot_row]

        pivots.append(col)
        pivot_row += 1

    R[np.abs(R) < tol] = 0.0
    return R, pivots


def rank(A):
    _, pivots = rref(A)
    return len(pivots)


def null_space_basis(A):
    R, pivots = rref(A)
    num_cols = R.shape[1]
    free_columns = [col for col in range(num_cols) if col not in pivots]

    basis = []
    for free_col in free_columns:
        vector = np.zeros(num_cols)
        vector[free_col] = 1.0
        for row, pivot_col in enumerate(pivots):
            vector[pivot_col] = -R[row, free_col]
        basis.append(vector)
    return basis


def col_space_basis(A):
    A = np.array(A, dtype=float)
    _, pivots = rref(A)
    return A[:, pivots]


def main():
    R, pivots = rref([[1, 2, -1], [3, -1, 4]])
    print("RREF of [[1, 2, -1], [3, -1, 4]]:")
    print(R)
    print("pivots:", pivots)

    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R, pivots = rref(F)
    print("\nRREF of F:")
    print(R)
    print("pivots:", pivots)
    print("rank(F):", rank(F))
    print("null space basis of F:", null_space_basis(F))
    print("column space basis of F:")
    print(col_space_basis(F))

    rng = np.random.default_rng(0)
    all_checks_passed = True
    for _ in range(20):
        A = rng.normal(size=(5, 7))
        null_basis = null_space_basis(A)
        kernel_is_correct = all(np.allclose(A @ v, 0) for v in null_basis)
        rank_nullity_holds = rank(A) + len(null_basis) == 7
        rank_matches_numpy = rank(A) == np.linalg.matrix_rank(A)
        if not (kernel_is_correct and rank_nullity_holds and rank_matches_numpy):
            all_checks_passed = False
    print("\n20 random 5x7 matrices, all checks passed:", all_checks_passed)

    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    null_basis = null_space_basis(A)
    print("\n6x8 matrix built as (6x4) @ (4x8):")
    print("rank:", rank(A))
    print("null space dimension:", len(null_basis))
    print("A @ v close to zero:", all(np.allclose(A @ v, 0) for v in null_basis))


if __name__ == "__main__":
    main()
