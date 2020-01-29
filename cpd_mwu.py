from helper import rand_init
from helper import sample
from helper import bern
from helper import update_weights
from helper import update_factors
from helper import residual_error
from helper import norm

import numpy as np
"""
Computes CPD for 3 dimensional tensors.
Returns A,B,C
"""
def CPD_MWU(X, sketching_rates, lamb, eps, nu, rank, update, num_iterations=100):
	# Keep residual errors
	error = []

	# Initialize weights
	weights = np.array([1] * len(sketching_rates)) / (len(sketching_rates))

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A, B, C = rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)\

	# Append initialization residual error
	error.append(residual_error(X,A,B,C))

	# Run CPD_MWU for num iterations
	for i in range(num_iterations):
		# Select sketching rate with probability proportional to w_i
		s = sample(sketching_rates, weights)

		# Solve Ridge Regression for A,B,C
		A, B, C = update_factors(A, B, C, X, lamb, s, rank)

		# Update weights
		if bern(eps) == 1 and len(sketching_rates) > 1 and update:
			update_weights(A, B, C, X, lamb, weights, sketching_rates, rank, nu)

		# Hold Residual Error
		error.append(residual_error(X,A,B,C))
		# print("Iteration", i, ":", error[-1])
	
	return A,B,C, np.array(error)

