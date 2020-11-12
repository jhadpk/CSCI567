### DO NOT MODIFY THIS FILE ###

import random
import time

import numpy as np


class Dataset:
	def __init__(self,
		         tagfile,
		         datafile,
		         train_test_split=0.8,
		         seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags


class Line:
	def __init__(self, line):
		words = line.split("\n")
		self.id = words[0]
		self.words = []
		self.tags = []

		for idx in range(1, len(words)):
			pair = words[idx].split("\t")
			self.words.append(pair[0])
			self.tags.append(pair[1])
		self.length = len(self.words)
		return

	def show(self):
		print(self.id)
		print(self.length)
		print(self.words)
		print(self.tags)
		return
