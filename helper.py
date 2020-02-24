from numpy import eye
from numpy.linalg import pinv, norm
import numpy as np
import time
import random
import tensorly.tenalg as tl_alg
import tensorly.base as tl_base
import tensorly.kruskal_tensor as tl_kruskal

"""
Randomly generates numpy matrix G with entries [0,1)
"""
def rand_init(dim, rank):
	return np.random.rand(dim, rank)

"""
Randomly selects a sketching weight using weight distribution
"""
def sample(sketching_rate, weights):
	r = random.uniform(0, sum(weights))
	total_sum = 0

	for i, w in enumerate(weights):
		total_sum += w
		if total_sum > r:
			return sketching_rate[i]

	return weights[-1]
"""
Unfolds a tensors following the Kolda and Bader definition
Source: https://stackoverflow.com/questions/49970141/using-numpy-reshape-to-perform-3rd-rank-tensor-unfold-operation
"""
def unfold(tensor, mode=0):
	return tl_base.unfold(tensor, mode)

"""
Returns a singular sample from bernoulli distribution.
"""
def bern(eps):
	return np.random.binomial(n=1, p=eps)

"""
Updates weights according to CPD reading
"""
def update_weights(A, B, C, X, Id, norm_x, lamb, weights, sketching_rates, rank, nu, eps):
	for i, w in enumerate(weights):
		start = time.time()
		s = sketching_rates[i]
		A_new, B_new, C_new = update_factors(A, B, C, X, Id, lamb, s, rank)
		total_time = time.time() - start
		weights[i] *= np.exp(-nu/eps*(residual_error(X, norm_x, A_new, B_new, C_new) - residual_error(X, norm_x, A, B, C)) / (total_time))
	
	weights /= sum(weights)
	return

"""
Updates factor matrices through ridge regression
"""
def update_factors(A, B, C, X, Id, lamb, s, rank):
	# Update A
	X_unfold = tl_base.unfold(X, 0)
	dim_1, dim_2 = X_unfold.shape
	idx = generate_sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=0).T)[:, idx]
	A = (lamb * A + X_unfold[:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	# Update B
	X_unfold = tl_base.unfold(X, 1)
	dim_1, dim_2 = X_unfold.shape
	idx = generate_sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=1).T)[:, idx]
	B = (lamb * B + X_unfold[:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	# Update C
	X_unfold = tl_base.unfold(X, 2)
	dim_1, dim_2 = X_unfold.shape
	idx = generate_sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=2).T)[:, idx]
	C = (lamb * C + X_unfold[:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	return A,B,C
"""
Generates sketching indices
"""
def generate_sketch_indices(s, total_col):
	return np.random.choice(range(total_col), size=int(s*total_col), replace=False, p=None)

"""
Computes residual error
"""
def residual_error(X, norm_x, A, B, C):
	X_bar = tl_kruskal.kruskal_to_tensor([A,B,C])
	return norm(X-X_bar)/norm_x

"""
Computes norm of a tensor
"""
def norm(X):
	return np.linalg.norm(X)
	

