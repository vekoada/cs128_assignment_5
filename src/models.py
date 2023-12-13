"""Implementation of various Markov models and Language models for query likelihood document rankings.
"""

import numpy as np
import random

__author__ = "Adam Vekony"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = "Adam Vekony"
__license__ = "MIT"
__email__ = "avekony@westmont.edu"


class MarkovModel:
    def __init__(self, mode: str = 'word', n=3):
        self._mode = mode.lower()
        if self._mode not in ("char", "word"):
            raise ValueError(f"Invalid model type: {mode}. Valid values are 'char' and 'word'.")
        self._n = n
        self._matrix = None

    def _frequency_table(self, seq):
      if self._mode != 'char':
        seq = tuple(seq.split())
      table = {}
      for i in range(len(seq) - self._n):
          preceding = seq[i:i + self._n]
          next = seq[i + self._n]

          if table.get(preceding) is None:
              table[preceding] = {}
              table[preceding][next] = 1
          else:
              if table[preceding].get(next) is None:
                  table[preceding][next] = 1
              else:
                  table[preceding][next] += 1

      return table

    def _transition_matrix(self, table):
        for seq in table.keys():
            total_count = float(sum(table[seq].values()))
            for el_count in table[seq].keys():
                table[seq][el_count] = table[seq][el_count] / total_count

        return table

    def train(self, text):
        self._matrix = self._transition_matrix(self._frequency_table(text))

    def _predict(self, seq):
        seq = seq[-self._n:]
        if self._matrix.get(seq) is None:
              seq = random.choice(list(self._matrix.keys()))
        possibilities = list(self._matrix[seq].keys())
        probs = list(self._matrix[seq].values())

        return np.random.choice(possibilities, p=probs)

    def generate(self, start, max_len=500):
        generation = re.sub('\W+\s*', ' ', start).lower()
        if self._mode != 'char':
            generation = generation.split()
            seq = tuple(start[-self._n:])
        else:
            seq = start[-self._n:]

        for _ in range(max_len):
            next_prediction = self._predict(seq)
            if self._mode != 'char':
                generation.append(next_prediction)
                seq = tuple(generation[-self._n:])
            else:
                generation += next_prediction
                seq = generation[-self._n:]

        return " ".join(generation) if self._mode != 'char' else generation



