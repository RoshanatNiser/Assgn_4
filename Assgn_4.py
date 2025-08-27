# Assignment 4: Cholesky decompostion and Jacobi
# Name: Roshan Yadav
# Roll No: 2311144

from Func_lib_assgn_4 import *

A=read_matrix('F.txt')
b=read_vector('N.txt')

print(A)
print(b)

# Question_1:
p=cholesky_solve(A,b)
x,L= p
print(x) # Output: 

# Question_2: Jacobi
r=jacobi(A, b)
print(r) # Output: [-4.984900771468886e+120, 4.0254513349757756e+120, 5.119708280191826e+121, -1.0710410128027627e+121] 