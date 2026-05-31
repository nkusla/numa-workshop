import numpy as np

def quadratic(x: np.ndarray, a: float, b: float):
	return a*x[0]**2 + b*x[1]**2

def quadratic_grad(x: np.ndarray, a: float, b: float):
	df_dx = 2*a*x[0]
	df_dy = 2*b*x[1]
	return np.array([df_dx, df_dy])

def six_hump_camel_function(x: np.ndarray):
    term1 = (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2
    term2 = x[0] * x[1]
    term3 = (-4 + 4 * x[1]**2) * x[1]**2
    return term1 + term2 + term3

def six_hump_camel_gradient(x: np.ndarray):
    df_dx = 2 * (x[0]**5 - 4.2 * x[0]**3 + 4 * x[0] + x[1])
    df_dy = 8 * (x[1]**3 - 2 * x[0]**2 * x[1])
    return np.array([df_dx, df_dy])

def plateau_function(x: np.ndarray):
    return x[0]**2 + x[1]**2 - x[0]*x[1] + 5*x[1]**3 + 3*x[1]**4

def plateau_function_grad(x: np.ndarray):
    df_dx = 2*x[0] - x[1]
    df_dy = 12*x[1]**3 + 15*x[1]**2 + 2*x[1] - x[0]
    return np.array([df_dx, df_dy])

def example_diff_eq(x, y):
	return 2*np.sin(x)

def example_analytical_sol(x, x0, y0):
	return y0 + 2*np.cos(x0) - 2*np.cos(x)
