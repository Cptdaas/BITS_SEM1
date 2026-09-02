# ============================================================
# Q2 - Dataset, Rank, Covariance and Power Method
#
# The dataset has 500 rows and 6 features.
#
# f1, f2, f3 and f4 are random standard normal features.
# f5 = 2f1 + 3f2
# f6 = f3 - 2f4
#
# Rank and Power Method are implemented manually.
# np.linalg.eigh() is used only in part (d).
# ============================================================

import sys
from pathlib import Path

import numpy as np


class OutputTee:
    """Write program output to the terminal and a response file."""

    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.file = output_file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


response_file = open(
    Path(__file__).with_name("Q2_reponse.txt"),
    "w",
    encoding="utf-8"
)
sys.stdout = OutputTee(response_file)


# Fixing the seed so that results can be reproduced
np.random.seed(42)

np.set_printoptions(
    precision=8,
    suppress=True,
    linewidth=140
)

TOL = 1e-8


# ------------------------------------------------------------
# Calculate vector norm
# ------------------------------------------------------------

def vnorm(x):
    """Returns the Euclidean norm of vector x."""
    return (x @ x) ** 0.5


# ------------------------------------------------------------
# Q2(1)
# Generate the dataset X
# ------------------------------------------------------------

n = 500


# Four random features from standard normal distribution
f1 = np.random.randn(n)
f2 = np.random.randn(n)
f3 = np.random.randn(n)
f4 = np.random.randn(n)


# Two dependent features
f5 = 2 * f1 + 3 * f2
f6 = f3 - 2 * f4


# Create the dataset
X = np.column_stack([
    f1,
    f2,
    f3,
    f4,
    f5,
    f6
])


print("Q2(1) Dataset X")

print("\nShape of X =", X.shape)

print("\nFirst 10 rows of X:")
print(X[:10])


# ------------------------------------------------------------
# Q2(2)
# Find rank of X using Gaussian elimination
# ------------------------------------------------------------

def rank_via_elimination(M):
    """
    Finds the rank by counting the number of pivots
    obtained during Gaussian elimination.
    """

    M = M.astype(float).copy()

    rows, cols = M.shape

    pivot_row = 0
    rank = 0

    # Process each column
    for col in range(cols):

        if pivot_row >= rows:
            break

        # Find the largest absolute value below the
        # current pivot row
        candidate = (
            np.argmax(
                np.abs(M[pivot_row:, col])
            )
            + pivot_row
        )

        # No pivot in this column
        if abs(M[candidate, col]) < TOL:
            continue

        # Swap rows
        M[[pivot_row, candidate]] = (
            M[[candidate, pivot_row]]
        )

        # Eliminate values below pivot
        for r in range(pivot_row + 1, rows):

            M[r, :] -= (
                M[r, col]
                / M[pivot_row, col]
            ) * M[pivot_row, :]

        rank += 1
        pivot_row += 1

    return rank


print("\nQ2(2) Rank of X")

print(
    "rank(X) =",
    rank_via_elimination(X)
)

print(
    "Expected rank = 4 because f5 and f6 "
    "are linear combinations of the first four features."
)


# ------------------------------------------------------------
# Q2(3a)
# Covariance matrix
#
# C = (1/n) X^T X
# ------------------------------------------------------------

C = (X.T @ X) / n


print("\nQ2(3a) Covariance matrix")

print("Shape of C =", C.shape)

print("\nC =")
print(C)


# ------------------------------------------------------------
# Q2(3b)
# Power Method for dominant eigenvalue
# ------------------------------------------------------------

def power_method(
    M,
    max_iter=100000,
    tol=1e-12,
    seed=0
):
    """
    Finds the dominant eigenvalue and eigenvector
    using the Power Method.
    """

    rng = np.random.default_rng(seed)

    # Start with a random vector
    v = rng.standard_normal(
        M.shape[0]
    )

    # Make the vector unit length
    v = v / vnorm(v)

    lam_old = 0.0

    history = []

    for k in range(
        1,
        max_iter + 1
    ):

        # Multiply matrix by current vector
        w = M @ v

        # Normalise the result
        v = w / vnorm(w)

        # Rayleigh quotient
        lam = v @ (M @ v)

        history.append(lam)

        # Check convergence
        if abs(lam - lam_old) < tol:
            return (
                lam,
                v,
                k,
                history
            )

        lam_old = lam

    return (
        lam,
        v,
        max_iter,
        history
    )


# Find largest eigenvalue and corresponding eigenvector
lam1, v1, it1, _ = power_method(
    C,
    seed=1
)


print("\nQ2(3b) Power Method")

print("\nlambda_1 =", lam1)

print("\nv_1 =")
print(v1)

print(
    "\nNumber of iterations =",
    it1
)


# ------------------------------------------------------------
# Q2(3c)
# Power Method with deflation
#
# After finding an eigenvector, its direction is removed
# from the matrix before finding the next eigenpair.
#
# M = C - (sum vj vj^T) C
# ------------------------------------------------------------

def power_method_deflation(
    C,
    k,
    seed=1
):
    """
    Finds k eigenpairs using the Power Method
    with deflation.
    """

    eigvals = []
    eigvecs = []

    for i in range(k):

        # Start with C for every new eigenvalue
        M = C.copy()

        # Remove the directions already found
        for vj in eigvecs:

            M = (
                M
                - np.outer(vj, vj) @ M
            )

        # Apply Power Method
        lam, v, _, _ = power_method(
            M,
            seed=seed + i
        )

        eigvals.append(lam)
        eigvecs.append(v)

    return (
        np.array(eigvals),
        np.array(eigvecs)
    )


# Rank is 4, so we find 4 non-zero eigenvalues
k = 4


pm_vals, pm_vecs = power_method_deflation(
    C,
    k
)


print(
    "\nQ2(3c) Power Method with Deflation"
)


for i in range(k):

    print(
        f"\nlambda_{i + 1} = "
        f"{pm_vals[i]:.8f}"
    )

    print(
        f"v_{i + 1} ="
    )

    print(pm_vecs[i])


# ------------------------------------------------------------
# Q2(3d)
# Use Python library function to find eigenvalues
# and eigenvectors.
#
# np.linalg.eigh() is allowed in this part.
# ------------------------------------------------------------

w, V = np.linalg.eigh(C)


# Arrange eigenvalues from largest to smallest
order = np.argsort(w)[::-1]

w = w[order]
V = V[:, order]


print(
    "\nQ2(3d) Eigenvalues and eigenvectors "
    "using np.linalg.eigh()"
)


print("\nAll eigenvalues:")
print(w)


for i in range(k):

    print(
        f"\nlambda_{i + 1} (library) = "
        f"{w[i]:.8f}"
    )

    print(
        f"v_{i + 1} (library) ="
    )

    print(V[:, i])


# ------------------------------------------------------------
# Compare Power Method results with library results
# ------------------------------------------------------------

print(
    "\nComparison of Power Method and "
    "library results:"
)


for i in range(k):

    # Eigenvectors can differ only by their sign
    same_dir = np.sign(
        pm_vecs[i] @ V[:, i]
    )

    eig_diff = abs(
        pm_vals[i] - w[i]
    )

    vec_diff = vnorm(
        pm_vecs[i]
        - same_dir * V[:, i]
    )

    print(
        f"\nlambda_{i + 1}: "
        f"PM = {pm_vals[i]:.8f}, "
        f"Library = {w[i]:.8f}"
    )

    print(
        f"Eigenvalue difference = "
        f"{eig_diff:.2e}"
    )

    print(
        f"Eigenvector difference = "
        f"{vec_diff:.2e}"
    )


# ------------------------------------------------------------
# Q2(3e)
# Number of iterations required to reach
# accuracy of 10^-7
# ------------------------------------------------------------

def iters_to_accuracy(
    C,
    eigvecs_found,
    true_lambda,
    target=1e-7,
    seed=1,
    idx=0
):
    """
    Finds the number of Power Method iterations
    required to reach the specified accuracy.
    """

    M = C.copy()

    # Deflate eigenvectors already obtained
    for vj in eigvecs_found:

        M = (
            M
            - np.outer(vj, vj) @ M
        )

    # Run Power Method and store its history
    _, _, _, hist = power_method(
        M,
        seed=seed + idx
    )

    # Check when the required accuracy is reached
    for k_iter, lam in enumerate(
        hist,
        1
    ):

        if abs(
            lam - true_lambda
        ) < target:

            return k_iter

    return len(hist)


print(
    "\nQ2(3e) Iterations required "
    "for accuracy 10^-7"
)


# Store eigenvectors found so far
found = []


for i in range(k):

    it = iters_to_accuracy(
        C,
        found,
        w[i],
        idx=i
    )

    # Ratio of consecutive eigenvalues
    if i + 1 < len(w):

        gap_ratio = abs(
            w[i + 1] / w[i]
        )

    else:

        gap_ratio = 0

    print(
        f"\nlambda_{i + 1}: "
        f"{it} iterations"
    )

    print(
        f"Eigenvalue gap ratio = "
        f"{gap_ratio:.4f}"
    )

    # Add current eigenvector for next deflation
    found.append(
        pm_vecs[i]
    )

sys.stdout = sys.stdout.terminal
response_file.close()