
from pyparsing import abstractmethod
import numpy as np
from typing import List

class NumericalAlgo:
	def __init__(self, function):
		self.function = function

	@abstractmethod
	def step(self) -> np.ndarray:
		pass

	@abstractmethod
	def execute(self, n_iter, epsilon) -> List[np.ndarray]:
		pass