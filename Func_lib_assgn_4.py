# Assignment 4: Cholesky decompostion and Jacobi
# Name: Roshan Yadav
# Roll No: 2311144

def read_matrix(filename):
    """Read matrix from a file"""
    with open( filename , 'r' ) as f :
        matrix =[]
        for line in f :
            # Convert each line into a list of floats
            row = [ float(num) for num in line.strip().split() ]
            matrix.append(row)
    return matrix

def read_vector(filename):
    """
    Read a vector from a file.
    """
    with open(filename, 'r') as f:
        vector = []
        for line in f:
            vector.append(float(line.strip()))
    return vector



def jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve Ax = b using Jacobi method with a given maximum iter as max_iter.
    """

    #Number of equations
    n = len(b)  

    # Step 0: check if any diagonal element of A is zero. If yes then swap the rows.

    for i in range(n):
        if A[i][i] ==0:
            k=A[i]
            if i == 0:
                m = A[-1]
                j=A[i]
                A[i]=m
                A[-1]=j
                m= b[i-1]
                j=b[-1]
                b[i]=m
                b[-1]=j
            else:
                m= A[i-1]
                j=A[i]
                A[i]=m
                A[i-1]=j
                m= b[i-1]
                j=b[i]
                b[i]=m
                b[i-1]=j

    # Step 1: Initialize guess vector x
    if x0 is None:
        x0 = [0.0] * n  # start with zeros

    x = x0[:]  # make a copy

    for k in range(max_iter):
        x_new = [0.0] * n  # to store new values

        # Step 2: Compute each x[i] using previous iteration values
        for i in range(n):
            # Calculate sum of a_ij * x_j for j != i
            sum_terms = 0.0
            for j in range(n):
                if j != i:
                    sum_terms += A[i][j] * x[j]
            # Update x_new[i] using Jacobi formula
            x_new[i] = (b[i] - sum_terms) / A[i][i]

        # Step 3: Compute maximum difference for convergence check
        diff = max(abs(x_new[i] - x[i]) for i in range(n))

        if diff < tol:
            print(f"Jacobi converged in {k+1} iterations")
            return x_new, k + 1

        # Step 4: Prepare for next iteration
        x = x_new

    print("Max iteration done")
    return x


def cholesky_solve(A, b):
    """
    This fuction solves system of linear equations using Cholesky decomposition
    with forward-backward substitution.
    """
    n = len(A)

    # Step 0: check if any diagonal element of A is zero. If yes then swap the rows.
    for i in range(n):
        if A[i][i] == 0:
            # Find a row with non-zero element in column i
            for k in range(i+1, n):
                if A[k][i] != 0:
                    # Swap rows in matrix A
                    A[i], A[k] = A[k], A[i]
                    # Swap corresponding elements in vector b
                    b[i], b[k] = b[k], b[i]
                    break
    
    # Step 1: Initialize L matrix
    L = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.0)
        L.append(row)
    
    # Step 2: Cholesky decomposition A = L * L^T
    for i in range(n):
        for j in range(i + 1):  # Only lower triangular part
            if i == j:  # Diagonal elements
                sum_sq = 0.0
                for k in range(j):
                    sum_sq += L[i][k] * L[i][k]
                L[i][j] = (A[i][i] - sum_sq) ** 0.5
            else:  # Below diagonal elements
                sum_prod = 0.0
                for k in range(j):
                    sum_prod += L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - sum_prod) / L[j][j]

    # Step 3: Forward Substitution - Solve Ly = b
    y = []
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i][j] * y[j]
        y.append((b[i] - sum_val) / L[i][i])
    
    # Step 4: Backward Substitution - Solve L^T * x = y
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += L[j][i] * x[j]  # L^T[i][j] = L[j][i]
        x[i] = (y[i] - sum_val) / L[i][i]
    
    return x, L
