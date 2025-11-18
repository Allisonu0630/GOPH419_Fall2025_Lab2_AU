import numpy as np
import warnings
from numpy.linalg import solve
"""
solve a system Lx=b where L is a lower triangle co-eff. matrix, b is the right rand side vector for matrix where each column is a rhs vector

parameters: 
____________________
- A : array like
	lower triangle
	size: nxn
- b: array like
	right hand sides
	size: (,n) or (n,m)
where m is columns


returns:
___________________
numpy.ndarray
	vector or matrix of solutions x
	this will have the same shape as b

"""

def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel'):
    
#############################################################################################################################################
#solve a system Lx=b where L is a lower triangle co-eff. matrix, b is the right rand side vector for matrix where each column is a rhs vector

#parameters: 

# A : array like
	#lower triangle
	#size: nxn
#b: array like
	#right hand sides
	#size: (,n) or (n,m)
#where m is columns
# x0 : array_like, optional?
           # Initial guess
# tol : float, optional?
           # Relative error tolerance
# alg : str, optional?
            #'seidel' or "jacobi" (case-insensitive)

#returns:

#numpy.ndarray
	#vector or matrix of solutions x
	#this will have the same shape as b
#############################################################


  
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Coefficient matrix A must be square")
    if b.ndim not in [1, 2]:
        raise ValueError("Right hand side b must be 1D or 2D")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have same number of rows as A")

    alg = alg.strip().lower()
    if alg not in ['seidel', 'jacobi']:
        raise ValueError(f"Unknown algorithm '{alg}'")

    n = A.shape[0]
    m = b.shape[1] if b.ndim == 2 else 1
    b = b.reshape(n, m)


    if x0 is None:
        x = np.zeros((n, m))
    else:
        x0 = np.array(x0, dtype=float)
        if x0.shape == b.shape:
            x = x0.copy()
        elif x0.ndim == 1 and x0.shape[0] == n:
            x = np.tile(x0.reshape(-1, 1), (1, m))
        else:
            raise ValueError("x0 shape not compatible with b")

    max_iter = 10000
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            Ai = A[i, :]
            if alg == 'jacobi':
                x_new[i, :] = (b[i, :] - Ai[:i] @ x[:i, :] - Ai[i+1:] @ x[i+1:, :]) / A[i, i]
            else:  # seidel
                x_new[i, :] = (b[i, :] - Ai[:i] @ x_new[:i, :] - Ai[i+1:] @ x[i+1:, :]) / A[i, i]

        rel_error = np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + 1e-12)
        if rel_error < tol:
            return x_new if m > 1 else x_new.flatten()
        x = x_new

    warnings.warn("did not converge after max iterations", RuntimeWarning)
    return x if m > 1 else x.flatten()


def spline_function(xd, yd, order=3):
    """
    make a spline interpolation function of order 1,2,3.

    Parameters:
        xd : array_like
            Independent variable (always increasing)
        yd : array_like
            Dependent variable (same shape as xd)
        order : int
            Spline order (1, 2, 3)

    Returns:
        interp : function
            Callable function interpolates input values
    """
    xd = np.asarray(xd).flatten()
    yd = np.asarray(yd).flatten()

    if xd.shape[0] != yd.shape[0]:
        raise ValueError("xd and yd must have the same length")
    if np.unique(xd).shape[0] != xd.shape[0]:
        raise ValueError("xd contains repeated values")
    if not np.all(np.diff(xd) > 0):
        raise ValueError("xd must be strictly increasing")
    if order not in [1, 2, 3]:
        raise ValueError("order must be 1, 2, or 3")

    n = len(xd)
    h = np.diff(xd)

    if order == 1: ########### linear spline############
        def interp(x):
            x = np.asarray(x)
            if np.any(x < xd[0]) or np.any(x > xd[-1]):
                raise ValueError(f"x out of bounds: [{xd[0]}, {xd[-1]}]")
            result = np.empty_like(x, dtype=float)
            for i in range(n - 1):
                mask = (x >= xd[i]) & (x <= xd[i + 1])
                slope = (yd[i + 1] - yd[i]) / h[i]
                result[mask] = yd[i] + slope * (x[mask] - xd[i])
            return result
        return interp

    elif order == 2: ############quadratic spline###########
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        A[0, 0] = 1
        A[-1, -1] = 1
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 2 * ((yd[i + 1] - yd[i]) / h[i] - (yd[i] - yd[i - 1]) / h[i - 1])

        c = solve(A, rhs)

        def interp(x):
            x = np.asarray(x)
            if np.any(x < xd[0]) or np.any(x > xd[-1]):
                raise ValueError(f"x out of bounds: [{xd[0]}, {xd[-1]}]")
            result = np.empty_like(x, dtype=float)
            for i in range(n - 1):
                mask = (x >= xd[i]) & (x <= xd[i + 1])
                dx = x[mask] - xd[i]
                slope = (yd[i + 1] - yd[i]) / h[i] - c[i] * h[i] / 2
                result[mask] = yd[i] + slope * dx + c[i] * dx**2 / 2
            return result
        return interp

    else:  # order == 3 ###########cubic spline############
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        A[0, 0] = 1
        A[-1, -1] = 1
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 6 * ((yd[i + 1] - yd[i]) / h[i] - (yd[i] - yd[i - 1]) / h[i - 1])

        M = solve(A, rhs)

        def interp(x):
            x = np.asarray(x)
            if np.any(x < xd[0]) or np.any(x > xd[-1]):
                raise ValueError(f"x out of bounds: [{xd[0]}, {xd[-1]}]")
            result = np.empty_like(x, dtype=float)
            for i in range(n - 1):
                mask = (x >= xd[i]) & (x <= xd[i + 1])
                xi, xi1 = xd[i], xd[i + 1]
                hi = h[i]
                yi, yi1 = yd[i], yd[i + 1]
                Mi, Mi1 = M[i], M[i + 1]
                dx = x[mask] - xi
                term1 = (Mi1 / (6 * hi)) * dx**3
                term2 = (Mi / (6 * hi)) * (xi1 - x[mask])**3
                term3 = (yi1 / hi - Mi1 * hi / 6) * dx
                term4 = (yi / hi - Mi * hi / 6) * (xi1 - x[mask])
                result[mask] = term1 + term2 + term3 + term4
            return result
        return interp
