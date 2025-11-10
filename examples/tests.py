import sys
import os
import numpy as np
from numpy.testing import assert_allclose

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.lab_02.linalg_interp import gauss_iter_solve, spline_function
def test_gauss_iter_single_rhs():
    print("Running test_gauss_iter_single_rhs...")
    A = np.array([[4.0, 1.0, 2.0],
                  [3.0, 5.0, 1.0],
                  [1.0, 1.0, 3.0]])
    b = np.array([4.0, 7.0, 3.0])
    expected = np.linalg.solve(A, b)
    for alg in ['seidel', 'jacobi']:
        result = gauss_iter_solve(A, b, tol=1e-10, alg=alg)
        assert_allclose(result, expected, rtol=1e-6)
        print(f"  ✓ {alg} solver matches numpy.linalg.solve")

def test_gauss_iter_inverse():
    print("Running test_gauss_iter_inverse...")
    A = np.array([[4.0, 1.0, 2.0],
                  [3.0, 5.0, 1.0],
                  [1.0, 1.0, 3.0]])
    I = np.eye(3)
    for alg in ['seidel', 'jacobi']:
        A_inv = gauss_iter_solve(A, I, tol=1e-10, alg=alg)
        identity = A @ A_inv
        assert_allclose(identity, I, rtol=1e-10, atol=1e-10)
        print(f"  ✓ {alg} inverse computation passed")

def test_spline_orders():
    print("Running test_spline_orders...")
    x = np.array([0, 1, 2, 3])
    x_test = np.array([0.5, 1.5, 2.5])

    # Linear: y = 2x + 1
    y = 2 * x + 1
    f1 = spline_function(x, y, order=1)
    expected = 2 * x_test + 1
    assert_allclose(f1(x_test), expected, rtol=1e-10)
    print(" ✓ Linear")

def test_spline_exceptions():
    print("Running test_spline_exceptions...")

    # Repeated xd values
    try:
        spline_function([0, 1, 1], [1, 2, 3], order=1)
    except ValueError:
        print("  ✓ Repeated xd error raised")

    # Unsorted xd values
    try:
        spline_function([3, 2, 1], [9, 4, 1], order=2)
    except ValueError:
        print("  ✓ Unsorted xd error raised")

    # Mismatched lengths
    try:
        spline_function([0, 1], [1], order=1)
    except ValueError:
        print("  ✓ Mismatched length error raised")

    # Invalid order
    try:
        spline_function([0, 1, 2], [1, 2, 3], order=4)
    except ValueError:
        print("  ✓ Invalid order error raised")

    # Extrapolation
    try:
        f = spline_function([0, 1, 2], [1, 2, 3], order=1)
        f(-1)
    except ValueError:
        print("  ✓ Extrapolation error raised")

if __name__ == "__main__":
    test_gauss_iter_single_rhs()
    test_gauss_iter_inverse()
    test_spline_orders()
    test_spline_exceptions()