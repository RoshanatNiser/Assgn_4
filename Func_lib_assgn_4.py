
# Assignment 4: Cholesky decomposition and Jacobi
# Name: Roshan Yadav
# Roll No: 2311144

import math

def read_matrix(filename):
    """Read matrix from a file"""
    with open(filename, 'r') as f:
        matrix = []
        for line in f:
            # Convert each line into a list of floats
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix

def read_vector(filename):
    """Read a vector from a file."""
    with open(filename, 'r') as f:
        vector = []
        for line in f:
            vector.append(float(line.strip()))
    return vector

def jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve Ax = b using Jacobi method with a given maximum iter as max_iter.
    """
    # Number of equations
    n = len(b)
    
    # Create copies to avoid modifying originals
    A_copy = [row[:] for row in A]
    b_copy = b[:]

    # Step 0: check if any diagonal element of A is zero. If yes then swap the rows.
    for i in range(n):
        if A_copy[i][i] == 0:
            if i == 0:
                # Swap with last row
                A_copy[i], A_copy[-1] = A_copy[-1], A_copy[i]
                b_copy[i], b_copy[-1] = b_copy[-1], b_copy[i]
            else:
                # Swap with previous row
                A_copy[i], A_copy[i-1] = A_copy[i-1], A_copy[i]
                b_copy[i], b_copy[i-1] = b_copy[i-1], b_copy[i]

    # Step 1: Initialize guess vector x
    if x0 is None:
        x0 = [0.0] * n  

    x = x0[:]  

    for k in range(max_iter):
        x_new = [0.0] * n 
        # Step 2: Compute each x[i] using previous iteration values
        for i in range(n):
            # Calculate sum of a_ij * x_j for j != i
            sum_terms = 0.0
            for j in range(n):
                if j != i:
                    sum_terms += A_copy[i][j] * x[j]
            # Update x_new[i] using Jacobi formula
            x_new[i] = (b_copy[i] - sum_terms) / A_copy[i][i]

        # Step 3: Compute maximum difference for convergence check
        diff = max(abs(x_new[i] - x[i]) for i in range(n))

        if diff < tol:
            print(f"Jacobi converged in {k+1} iterations")
            return x_new, k + 1

        # Step 4: Prepare for next iteration
        x = x_new

    print("Jacobi reached maximum iterations")
    return x

def cholesky_solve(A, b):
    """
    This function solves system of linear equations using Cholesky decomposition
    with forward-backward substitution.
    """
    n = len(A)
    # Step 0: check if any diagonal element of A is zero. If yes then swap the rows.
    for i in range(n):
        if A[i][i] == 0:
            if i == 0:
                # Swap with last row
                A[i], A[-1] = A[-1], A[i]
                b[i], b[-1] = b[-1], b[i]
            else:
                # Swap with previous row
                A[i], A[i-1] = A[i-1], A[i]
                b[i], b[i-1] = b[i-1], b[i]

    # Step 1: Cholesky decomposition - compute L such that A = L * L^T
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # diagonal elements
                sum_sq = sum(L[i][k] ** 2 for k in range(j))
                value = A[i][i] - sum_sq
                if value <= 0:
                    raise ValueError(f"Matrix not positive definite at ({i},{i}). Value: {value}")
                L[i][j] = math.sqrt(value)  
            else:  # off-diagonal elements (j < i)
                sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - sum_prod) / L[j][j]

    # Step 2: Forward substitution - solve L * y = b
    y = [0.0] * n
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L[i][i]

    # Step 3: Backward substitution - solve L^T * x = y
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        sum_val = sum(L[j][i] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_val) / L[i][i]

    return x, L
