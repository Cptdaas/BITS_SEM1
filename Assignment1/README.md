# Assignment: MFML

This folder contains the first MFML assignment. The notebook uses NumPy to demonstrate Gaussian elimination, null spaces, matrix rank, covariance matrices, and eigenvalue computation.

## Folder contents

- `assignment1MFML.ipynb`: The completed assignment notebook. It contains two executable Python cells for Q1 and Q2.
- `requirements.txt`: Python packages required by the notebook (`numpy` and `pandas`).
- `.gitignore`: Prevents virtual environments, Python cache files, Jupyter checkpoints, editor files, and macOS metadata from being committed.
- `README.md`: Assignment explanation and environment setup instructions.

## Question 1: Linear system solutions

### Question

For a randomly generated matrix $A$ with shape $5 \times 7$ and vector $b$ with five entries:

1. Build the augmented matrix $[A \mid b]$.
2. Reduce it to row echelon form (REF) and reduced row echelon form (RREF) without using built-in `rref`, `matrix_rank`, or `solve` functions.
3. Identify pivot and free-variable columns.
4. Find one particular solution to $Ax=b$.
5. Find a basis for the null space of $A$ and verify the general solution.

### Response

The notebook implements REF and RREF using Gaussian elimination with partial pivoting. A tolerance is used to avoid treating very small floating-point values as pivots.

The verified output shows:

- The system is consistent.
- Pivot columns are `[0, 1, 2, 3, 4]`.
- Free columns are `[5, 6]`, so there are two free variables.
- One particular solution is obtained by setting both free variables to zero.
- Two null-space basis vectors are constructed, one for each free variable.
- The general solution is

  `x = x_p + c1*n1 + c2*n2`

  where `c1` and `c2` are arbitrary scalars.
- The numerical check gives a maximum absolute error of approximately `1.11e-15`, confirming that `A @ x = b` up to floating-point precision.

## Question 2: Dataset rank and covariance eigenpairs

### Question

Construct a dataset $X$ with shape $500 \times 6$ using four independent standard-normal features and two dependent features:

- $f5 = 2f1 + 3f2$
- $f6 = f3 - 2f4$

Then:

1. Compute the rank without using `np.linalg.matrix_rank`.
2. Compute $C = (1/n)X^T X$.
3. Find eigenpairs using the hand-written power method.
4. Use deflation to obtain the top four eigenpairs.
5. Compare the result with `np.linalg.eigh`.
6. Measure iterations needed to reach eigenvalue accuracy `1e-7`.

### Response

The first four features are independent, while `f5` and `f6` are exact linear combinations of them. Therefore, the dataset has rank 4 even though it has six columns.

The verified output shows:

- `X.shape == (500, 6)`.
- `rank(X) == 4`.
- The covariance matrix has shape `(6, 6)`.
- Its four non-zero eigenvalues are approximately `12.47020120`, `5.98130177`, `1.05483099`, and `0.95881690`; the remaining two are numerically zero.
- The hand-written power method agrees with `np.linalg.eigh` to approximately `1e-12` in eigenvalue values. Eigenvectors may differ by sign, which is expected because `v` and `-v` represent the same direction.
- The measured iteration counts for the four eigenvalues are `13`, `5`, `72`, and `1` under the notebook's convergence procedure.

The third eigenvalue takes many iterations because its gap ratio is close to one (`0.9090`); power-method convergence becomes slower when successive eigenvalues are close.

## Setup on macOS

Open a terminal and move into the assignment folder:

```bash
cd "/Users/vinod/Documents/BITS_SEM1/BITS_SEM1/Assignment"
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip` and install the assignment requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the installed packages:

```bash
python -m pip list
```

When finished, leave the virtual environment with:

```bash
deactivate
```

## Run in VS Code

1. Open the `Assignment` folder in VS Code.
2. Install or enable the Python and Jupyter extensions if needed.
3. Choose the interpreter located at `Assignment/.venv/bin/python` using **Python: Select Interpreter**.
4. Open `assignment1MFML.ipynb`.
5. Select the `.venv` kernel and run the two populated cells from top to bottom.

The fixed random seed in the notebook makes the generated data and reported values reproducible.
