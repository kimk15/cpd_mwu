import numpy as np
from numpy.linalg import norm

"""
Generates collinear factors
"""
def generate_collinear_factors(shape, rank, congruence):
	# Grab tensor shape
	I, J, K = shape

	# Generate A, B, C
	A = generate_Collinear_Matrix(I, rank, congruence)
	B = generate_Collinear_Matrix(J, rank, congruence)
	C = generate_Collinear_Matrix(K, rank, congruence)

	return A, B, C

"""
Generates a collinear factor matrix
"""
def generate_Collinear_Matrix(I,R,congruence):
    # Generate F matrix using cholesky
    F = generate_F((R,R), congruence)
    # Generate P
    P = generate_P((I, R))

    return P @ F

"""
Generates A: Entries are generated from standard normal distribution
"""
def generate_A(size):
	return np.random.standard_normal(size)	

"""
Generate F: F is generated as stated in PARAFAC2 - PART I by kiers, berge, bro.
"""
def generate_F(size, congruence):
	# Fill an array with congruence
	A = np.ones(size) * congruence
	# Fill with one
	np.fill_diagonal(A, 1)
	# Compute choelesky decomposition
	F = np.linalg.cholesky(A)

	return F.T

"""
Generates P:
	Sample from standard normal -> orthnormalize column using QR
"""
def generate_P(size):
	# Sample from standard normal distribution
	P = np.random.standard_normal(size)
	#Orthnormalize columns
	Q, _ = np.linalg.qr(np.random.standard_normal(size))

	return Q

"""
congruence(): computes congruence between two vectors
"""
def congruence(x, y):
	return np.dot(x,y)/(norm(x)*norm(y))
