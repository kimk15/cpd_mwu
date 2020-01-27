import tensorly as tl
import tensorly.random as tl_rand
import tensorly.kruskal_tensor as tl_kruskal

from helper import rand_init
from helper import sample
from helper import bern
from helper import update_weights
from helper import update_factors
from helper import residual_error
from helper import norm
"""
Computes CPD for 3 dimensional tensors.
Returns A,B,C
"""
def CPD_MWU(X, sketching_rates, lamb, eps, nu, rank, F, num_iterations=100):
	# Keep residual errors
	error = []
	diff_norm = []

	# Initialize weights
	weights = [1] * len(sketching_rates)

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
		if bern(eps) == 1:
			update_weights(A, B, C, X, lamb, weights, sketching_rates, rank, nu)

		error.append(residual_error(X,A,B,C))
		diff_norm.append(norm(X-tl_kruskal.kruskal_to_tensor([A,B,C])))

		print("Iteration", i, ":", norm(X), diff_norm[-1], error[-1])
		F[0] = A
		F[1] = B
		F[2] = C
	return A,B,C, error, diff_norm




# Todo:
#	- run simulation on rank 5 tensor and plot out error metrics
#	- check update_weights