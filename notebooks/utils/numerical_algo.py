from pyparsing import abstractmethod

class NumericalAlgo:
	def __init__(self, function):
		self.function = function
		self.result = []

	@abstractmethod
	def execute(self, n_iter, epsilon):
		pass