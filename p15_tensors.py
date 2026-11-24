import time

import numpy as np


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


def check_reshape():
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)

    indices_match = all(
        T[i, j, k] == flat[12 * i + 4 * j + k]
        for i in range(2)
        for j in range(3)
        for k in range(4)
    )
    print("T[i, j, k] == T.reshape(-1)[12*i + 4*j + k]:", indices_match)

    first_unfolding = T.reshape(2, 12)
    second_unfolding = T.transpose(1, 0, 2).reshape(3, 8)
    print("T.reshape(2, 12):")
    print(first_unfolding)
    print("rank:", np.linalg.matrix_rank(first_unfolding))
    print("T.transpose(1, 0, 2).reshape(3, 8):")
    print(second_unfolding)
    print("rank:", np.linalg.matrix_rank(second_unfolding))


def check_einsum_table(rng):
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 5))
    S = rng.normal(size=(4, 4))
    x = rng.normal(size=3)
    y = rng.normal(size=4)
    z = rng.normal(size=3)

    checks = {
        "matrix product": np.allclose(np.einsum("ij,jk->ik", A, B), A @ B),
        "trace": np.allclose(np.einsum("ii->", S), np.trace(S)),
        "dot product": np.allclose(np.einsum("i,i->", x, z), x @ z),
        "outer product": np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y)),
        "transpose": np.allclose(np.einsum("ij->ji", A), A.T),
        "bilinear form": np.allclose(np.einsum("i,ij,j->", x, A, y), x @ A @ y),
    }
    for name, passed in checks.items():
        print(f"{name:<15}", passed)


def check_batches(rng):
    A = rng.normal(size=(100, 4, 4))
    B = rng.normal(size=(100, 4, 4))

    traces = np.einsum("bii->b", A)
    products = np.einsum("bij,bjk->bik", A, B)

    traces_loop = np.array([np.trace(A[b]) for b in range(100)])
    products_loop = np.array([A[b] @ B[b] for b in range(100)])

    print("batch of traces matches loop:", np.allclose(traces, traces_loop))
    print("batch of products matches loop:", np.allclose(products, products_loop))


def check_kronecker(rng):
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 5))
    X = rng.normal(size=(4, 5))

    direct = (A @ X @ B.T).reshape(-1)
    with_kron = np.kron(A, B) @ X.reshape(-1)
    print("(A X B^T).reshape(-1) == kron(A, B) @ X.reshape(-1):", np.allclose(direct, with_kron))

    n = 100
    A = rng.normal(size=(n, n))
    B = rng.normal(size=(n, n))
    X = rng.normal(size=(n, n))

    start = time.perf_counter()
    kron_matrix = np.kron(A, B)
    kron_result = kron_matrix @ X.reshape(-1)
    kron_time = time.perf_counter() - start

    start = time.perf_counter()
    direct_result = (A @ X @ B.T).reshape(-1)
    direct_time = time.perf_counter() - start

    print(f"\nnumbers stored in kron(A, B) for 100x100 matrices: {kron_matrix.size:,}")
    print(f"memory: {kron_matrix.nbytes / 1e6:.0f} MB")
    print(f"multiplications with kron: {n ** 4:,}")
    print(f"multiplications for A X B^T: {2 * n ** 3:,}")
    print("results agree:", np.allclose(kron_result, direct_result))
    print(f"time with kron: {kron_time:.4f} s, time for A X B^T: {direct_time:.6f} s")


def check_rank_one(rng):
    u = rng.normal(size=3)
    v = rng.normal(size=4)
    w = rng.normal(size=5)
    T = rank_one(u, v, w)
    uv = np.outer(u, v)

    slices_are_multiples = all(np.allclose(T[:, :, k], w[k] * uv) for k in range(5))
    print("every slice T[:, :, k] equals w[k] * u v^T:", slices_are_multiples)


def main():
    rng = np.random.default_rng(0)

    print("--- reshaping ---")
    check_reshape()
    print("\n--- Table 1 with np.einsum ---")
    check_einsum_table(rng)
    print("\n--- batches ---")
    check_batches(rng)
    print("\n--- Kronecker product ---")
    check_kronecker(rng)
    print("\n--- rank one tensor ---")
    check_rank_one(rng)


if __name__ == "__main__":
    main()
