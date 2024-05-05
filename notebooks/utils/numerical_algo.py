from pyparsing import abstractmethod

class NumericalAlgo:
	def __init__(self, function):
		self.function = function
		self.result = []

	@abstractmethod
	def execute(self, n_iter, epsilon):
		pass

class ODESolver:
	def __init__(self, diff_eq):
		self.diff_eq = diff_eq
		self.y = []
		self.x = []

	@abstractmethod
	def solve(self, x0, y0, x_end, step):
		pass