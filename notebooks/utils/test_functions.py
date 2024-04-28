import numpy as np

def quadratic(x, a, b):
	return a*x[0]**2 + b*x[1]**2

def quadratic_grad(x, a, b):
	df_dx = 2*a*x[0]
	df_dy = 2*b*x[1]
	return np.array([df_dx, df_dy])

def six_hump_camel_function(x):
    term1 = (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2
    term2 = x[0] * x[1]
    term3 = (-4 + 4 * x[1]**2) * x[1]**2
    return term1 + term2 + term3

def six_hump_camel_gradient(x):
    df_dx = 2 * (x[0]**5 - 4.2 * x[0]**3 + 4 * x[0] + x[1])
    df_dy = 8 * (x[1]**3 - 2 * x[0]**2 * x[1])
    return np.array([df_dx, df_dy])