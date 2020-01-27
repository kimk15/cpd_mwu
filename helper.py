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
def update_weights(A, B, C, X, lamb, weights, sketching_rates, rank, nu):
	for i, w in enumerate(weights):
		start = time.time()
		s = sketching_rates[i]
		A_new, B_new, C_new = update_factors(A, B, C, X, lamb, s, rank)
		total_time = time.time() - start
		weights[i] = (residual_error(X, A_new, B_new, C_new) - residual_error(X, A, B, C)) / total_time

	return

"""
Performs Ridge Regression and optimizes a singular factor matrix
"""
def RS_LS(A, B, C, X_orig, lamb, s, rank, l):
	dim_1, dim_2 = X_orig.shape
	idx = generate_sketch_indices(s, dim_2)

	if l == 'A':
		X = X_orig[:, idx]
		M = (tl_alg.khatri_rao([C,B]).T)[:, idx]
		# return np.matmul(X, np.linalg.pinv(M))
		return np.matmul((A + np.matmul(X,M.T)), np.linalg.pinv(np.matmul(M,M.T) + lamb*np.identity(rank)))
		# return np.matmul((A + np.matmul(np.matmul(np.matmul(X,S),S.T),M.T)), np.linalg.pinv(np.matmul(np.matmul(np.matmul(M,S),S.T),M.T) + lamb*np.identity(rank))) 
	if l == 'B':
		X = X_orig[:, idx]
		M = (tl_alg.khatri_rao([C,A]).T)[:, idx]
		Z = np.linalg.pinv(np.matmul(M,M.T) + lamb*np.identity(rank))
		# return np.matmul(X, np.linalg.pinv(M))
		return np.matmul((B + np.matmul(X,M.T)), np.linalg.pinv(np.matmul(M,M.T) + lamb*np.identity(rank)))
		# return np.matmul((B + np.matmul(np.matmul(np.matmul(X,S),S.T),M.T)), np.linalg.pinv(np.matmul(np.matmul(np.matmul(M,S),S.T),M.T) + lamb*np.identity(rank)))
	if l == 'C':
		X = X_orig[:, idx]
		M = (tl_alg.khatri_rao([B,A]).T)[:, idx]
		Z = np.linalg.pinv(np.matmul(M,M.T) + lamb*np.identity(rank))
		# return np.matmul(X, np.linalg.pinv(M))
		return np.matmul((C + np.matmul(X,M.T)), np.linalg.pinv(np.matmul(M,M.T) + lamb*np.identity(rank)))
	 	# return np.matmul((C + np.matmul(np.matmul(np.matmul(X,S),S.T),M.T)), np.linalg.pinv(np.matmul(np.matmul(np.matmul(M,S),S.T),M.T) + lamb*np.identity(rank)))
	return


"""
Updates factor matricies
"""
def update_factors(A, B, C, X, lamb, s, rank):
	A_new = RS_LS(A, B, C, unfold(X, mode=0), lamb, s, rank, 'A')
	B_new = RS_LS(A_new, B, C, unfold(X, mode=1), lamb, s, rank, 'B')
	C_new = RS_LS(A_new, B_new, C, unfold(X, mode=2), lamb, s, rank, 'C')

	return A_new,B_new,C_new

"""
Generates sketching indices
"""
def generate_sketch_indices(s, total_col):
	return np.random.choice(range(total_col), size=int(s*total_col), replace=False, p=None)

"""
Computes residual error
"""
def residual_error(X, A, B, C):
	X_bar = tl_kruskal.kruskal_to_tensor([A,B,C])
	return norm(X-X_bar)/norm(X)

"""
Computes norm of a tensor
"""
def norm(X):
	return np.linalg.norm(X)
	

