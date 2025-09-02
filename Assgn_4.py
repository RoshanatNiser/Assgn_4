# Assignment 4: Cholesky decompostion and Jacobi
# Name: Roshan Yadav
# Roll No: 2311144

from Func_lib_assgn_4 import *

A=read_matrix('F.txt')
b=read_vector('N.txt')

print(A)
print(b)

# Question_1:Cholesky Decomposition
print("Cholesky Decomposition")
x,L=cholesky_solve(A,b)
print(x) # Output: [-1.1102230246251565e-16, 0.9999999999999999, 1.0, 1.0000000000000004]

# Question_2: Jacobi
print("Jacobi Method")
r=jacobi(A, b)
print(r) # Output: [0.0, 0.9999994039535522, 0.9999997019767761, 0.9999997019767761]
