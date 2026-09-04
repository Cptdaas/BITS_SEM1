# ============================================================
# Q1 - Finding solutions of linear systems
#
# This program:
# 1. Constructs the augmented matrix [A | b]
# 2. Finds REF and RREF using Gaussian elimination
# 3. Identifies pivot and non-pivot columns
# 4. Finds a particular solution
# 5. Finds solutions of Ax = 0
# 6. Finds and verifies the general solution
#
# No built-in functions such as solve(), matrix_rank()
# or rref() are used.
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
    Path(__file__).with_name("Q1_reponse.txt"),
    "w",
    encoding="utf-8"
)
sys.stdout = OutputTee(response_file)


# Fixed seed so that the same random values are generated
np.random.seed(42)

np.set_printoptions(
    precision=8,
    suppress=True,
    linewidth=140
)

# Tolerance used for checking zero values
TOL = 1e-9


# ------------------------------------------------------------
# Function to convert a matrix into Row Echelon Form (REF)
# ------------------------------------------------------------

def to_ref(M):
    """
    Performs Gaussian elimination and returns REF.
    Partial pivoting is used while selecting pivots.
    """

    M = M.astype(float).copy()

    rows, cols = M.shape

    pivot_row = 0
    pivot_cols = []

    # Check columns from left to right
    for col in range(cols):

        if pivot_row >= rows:
            break

        # Select the row with the largest absolute value
        # in the current column
        candidate = (
            np.argmax(
                np.abs(M[pivot_row:, col])
            )
            + pivot_row
        )

        # Division by zero handling:
        # If the candidate is approximately zero, there is
        # no pivot in this column, so move to the next column.
        if abs(M[candidate, col]) < TOL:
            continue

        # Move the selected row to the pivot position
        M[[pivot_row, candidate]] = (
            M[[candidate, pivot_row]]
        )

        # Make all entries below the pivot zero
        for r in range(pivot_row + 1, rows):

            factor = (
                M[r, col]
                / M[pivot_row, col]
            )

            M[r, :] = (
                M[r, :]
                - factor * M[pivot_row, :]
            )

        pivot_cols.append(col)
        pivot_row += 1

    return M, pivot_cols


# ------------------------------------------------------------
# Function to convert a matrix into Reduced Row Echelon
# Form (RREF)
# ------------------------------------------------------------

def to_rref(M):
    """
    Converts a matrix to RREF using the REF obtained
    from Gaussian elimination.
    """

    R, pivot_cols = to_ref(M)

    # Start from the last pivot and move upwards
    for i in reversed(range(len(pivot_cols))):

        col = pivot_cols[i]
        row = i

        # Make the pivot equal to 1
        R[row, :] = (
            R[row, :]
            / R[row, col]
        )

        # Make all entries above the pivot zero
        for r in range(row):

            R[r, :] = (
                R[r, :]
                - R[r, col] * R[row, :]
            )

    return R, pivot_cols


# ------------------------------------------------------------
# Find pivot columns, non-pivot columns, particular solution
# and solutions of Ax = 0
# ------------------------------------------------------------

def analyse_system(A, b):
    """
    Finds the REF, RREF, pivot columns, free columns,
    particular solution and null-space basis.
    """

    m, n = A.shape

    # Construct augmented matrix [A | b]
    aug = np.hstack([
        A,
        b.reshape(-1, 1)
    ])

    # Find REF and RREF
    REF, _ = to_ref(aug)
    RREF, piv_all = to_rref(aug)

    # If the last column is a pivot column, then the system
    # contains an equation of the form 0 = non-zero.
    consistent = not any(
        c == n for c in piv_all
    )

    # Pivot columns belonging to A
    pivot_cols = [
        c for c in piv_all
        if c < n
    ]

    # Columns without pivots are free columns
    free_cols = [
        c for c in range(n)
        if c not in pivot_cols
    ]

    # --------------------------------------------------------
    # Particular solution
    #
    # Set all free variables equal to zero.
    # --------------------------------------------------------

    x_p = np.zeros(n)

    for i, c in enumerate(pivot_cols):
        x_p[c] = RREF[i, n]

    # --------------------------------------------------------
    # Solutions of Ax = 0
    #
    # One null-space vector is obtained for every free
    # variable.
    # --------------------------------------------------------

    null_basis = []

    for f in free_cols:

        v = np.zeros(n)

        # Set the selected free variable equal to 1
        v[f] = 1.0

        # Calculate the corresponding pivot variables
        for i, c in enumerate(pivot_cols):
            v[c] = -RREF[i, f]

        null_basis.append(v)

    return (
        REF,
        RREF,
        pivot_cols,
        free_cols,
        x_p,
        null_basis,
        consistent
    )


# ------------------------------------------------------------
# Q1(3)
# Random 5 x 7 matrix A and vector b
# ------------------------------------------------------------

A = np.random.randn(5, 7)

b = np.random.randn(5)


print("Q1(3) Random system")

print("\nA =")
print(A)

print("\nb =")
print(b)


# ------------------------------------------------------------
# Construct augmented matrix and perform REF/RREF
# ------------------------------------------------------------

(
    REF,
    RREF,
    pivot_cols,
    free_cols,
    x_p,
    null_basis,
    consistent
) = analyse_system(A, b)


print("\nAugmented matrix [A | b] =")
print(
    np.hstack([
        A,
        b.reshape(-1, 1)
    ])
)


print("\nREF of [A | b] =")
print(REF)


print("\nRREF of [A | b] =")
print(RREF)


# ------------------------------------------------------------
# Pivot and non-pivot columns
# ------------------------------------------------------------

print("\nConsistent system =", consistent)

print("Pivot columns     =", pivot_cols)

print("Non-pivot columns =", free_cols)


# ------------------------------------------------------------
# Particular solution
# ------------------------------------------------------------

print("\nParticular solution x_p =")
print(x_p)

print("\nVerification of A x_p = b")
print("A @ x_p =", A @ x_p)
print("b       =", b)


# ------------------------------------------------------------
# Solutions of Ax = 0
# ------------------------------------------------------------

print("\nSolutions of Ax = 0:")

for j, v in enumerate(null_basis, 1):

    print(
        f"n{j} =",
        v
    )

    print(
        f"A @ n{j} =",
        A @ v
    )


# ------------------------------------------------------------
# General solution
#
# x = x_p + c1*n1 + c2*n2 + ...
# ------------------------------------------------------------

coeffs = np.random.randn(
    len(null_basis)
)


x_general = (
    x_p
    + sum(
        c * v
        for c, v in zip(
            coeffs,
            null_basis
        )
    )
)


print("\nRandom values of free variables:")
print(coeffs)


print("\nGeneral solution:")
print("x = x_p + c1*n1 + c2*n2 + ...")

print("\nOne general solution obtained using random coefficients:")
print(x_general)


# ------------------------------------------------------------
# Verification of the general solution
# ------------------------------------------------------------

print("\nVerification of general solution")

print("A @ x_general =")
print(A @ x_general)

print("\nb =")
print(b)

print(
    "\nMaximum absolute error =",
    np.max(
        np.abs(
            A @ x_general - b
        )
    )
)

sys.stdout = sys.stdout.terminal
response_file.close()