import numpy as np
import matplotlib.pylab as plt
from utils.numerical_algo import NumericalAlgo
from typing import Tuple, Callable, List

DIGITS_COUNT = 10
LINSPACE_SAMPLE = 1000
FIG_WIDTH = 15
FIG_HEIGHT = 8
CONTOUR_LEVELS = 20

def plot_function2D(function: Callable, interval: Tuple[float, float]):
	x = np.linspace(interval[0], interval[1], LINSPACE_SAMPLE)
	y = function(x)
	plt.plot(x, y, color="r")

def set_plot_attributes(title: str, lim: Tuple[float, float]):
	plt.figure(figsize=(15, 8))
	plt.axis('equal')
	plt.title(title)
	plt.xlabel("x")
	plt.axhline(0, color='k', linewidth=.5)
	plt.ylabel("y")
	plt.axvline(0, color='k', linewidth=.5)
	plt.grid(True, linestyle="--")
	plt.xlim(lim)
	plt.ylim(lim)

def visualize_bisection(algo: NumericalAlgo, lim: Tuple[float, float]):
	solution = round(algo.result[-1], DIGITS_COUNT)
	value = round(algo.function(solution), DIGITS_COUNT)
	title = f"Bisection method\nIterations: {len(algo.result)}\nSolution: {solution}\nValue: {value}"

	set_plot_attributes(title, lim)

	x_max = np.max(np.abs(algo.result))
	plot_interval = (-x_max, x_max)
	plot_function2D(algo.function, plot_interval)

	x = np.array(algo.result)
	y = algo.function(x)
	plt.scatter(x, y, s=20, c='b')

	plt.show()

def visualize_newton_rhapson(algo: NumericalAlgo, lim: Tuple[float, float]):
	solution = round(algo.result[-1], DIGITS_COUNT)
	value = round(algo.function(solution), DIGITS_COUNT)

	title = f"Netwon-Rhapson method\nIterations: {len(algo.result)}\nSolution: {solution}\nValue: {value}"

	set_plot_attributes(title, lim)

	x_max = np.max(np.abs(algo.result))
	plot_interval = (-x_max, x_max)
	plot_function2D(algo.function, plot_interval)

	x = np.array(algo.result)
	y = algo.function(x)
	plt.scatter(x, y, s=20, c='b')

	plt.show()

def visualize_secant(algo: NumericalAlgo, lim: Tuple[float, float]):
	pass

def plot_run(algo: NumericalAlgo, x_range: np.ndarray, plot_3d=False):
	run = algo.result
	function = algo.function
	value = np.around(function(run[-1]), DIGITS_COUNT)
	solution = np.around(run[-1], DIGITS_COUNT)
	label = type(algo).__name__

	title = f"\nIterations: {len(algo.result)}\n"
	title += f"Solution: {solution}\n"
	title += f"Value: {value}"

	X1, X2 = np.meshgrid(x_range, x_range)
	Z = function([X1, X2])

	x1 = np.array([x[0] for x in run])
	x2 = np.array([x[1] for x in run])
	z =  np.array([function(x) for x in run])

	fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
	fig.suptitle(title, horizontalalignment = 'center')

	if plot_3d:
		ax1 = fig.add_subplot(122, projection='3d')
		ax1.scatter(x1, x2, z, c='black', marker='.', s=40)
		ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
		ax1.legend([label], loc='upper right')
		ax1.set_xlabel('x_1')
		ax1.set_ylabel('x_2')
		ax1.set_title('3D surface plot', fontsize=10)

	ax2 = fig.add_subplot(121)
	contour = ax2.contour(X1, X2, Z, levels=CONTOUR_LEVELS, alpha=1.0, cmap='viridis')
	ax2.plot(x1, x2, color='black', marker='o', ls='--', linewidth=.5, ms=2)
	ax2.legend([label], loc='upper right')
	ax2.set_xlabel('x_1')
	ax2.set_ylabel('x_2')
	ax2.set_title('Contour plot', fontsize=10)

	plt.tight_layout()
	plt.show()

def plot_multiple_runs(algos: List[NumericalAlgo], x_range: np.ndarray):
	colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']

	plt.figure(figsize=(FIG_HEIGHT, FIG_HEIGHT))

	function = algos[0].function
	X1, X2 = np.meshgrid(x_range, x_range)
	Z = function([X1, X2])
	plt.contour(X1, X2, Z, levels=CONTOUR_LEVELS, alpha=1.0, cmap='viridis')

	plt.xlabel('x_1')
	plt.ylabel('x_2')
	plt.title('Contour plot', fontsize=10)

	labels = [type(a).__name__ for a in algos]

	for i, algo in enumerate(algos):
		run = algo.result

		x1 = np.array([x[0] for x in run])
		x2 = np.array([x[1] for x in run])
		z =  np.array([function(x) for x in run])

		plt.plot(x1, x2, color=colors[i], marker='o', ls='--', linewidth=.5, ms=2, label=labels[i])

	plt.legend(loc='upper right')
	plt.show()