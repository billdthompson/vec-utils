import numpy as np 
import pandas as pd

N = 500000 # skipgram max vocabulary size
D = 300 # skipgram vector dimension

# lightweight wrapper for interfacing with .vec skipgram files 
class Skipgram:
	def __init__(self, modelpath=''):
		with open(modelpath, 'r') as f:
			# skip header
			next(f)
			
			self.vectors = np.zeros((N, D))
			self.word = np.empty(N, dtype = object)
			for i, line in enumerate(f):
		
				if i >= N: break

				rowentries = line.rstrip('\n').split(' ')
				self.word[i] = rowentries[0]
				self.vectors[i] = rowentries[1:D + 1]

			self.vectors = self.vectors[:i]
			self.word = pd.DataFrame(dict(word = self.word[:i], idx = range(i))).set_index('word')

	def __getitem__(self, w):
		return self.vectors[self.word.loc[w].idx]