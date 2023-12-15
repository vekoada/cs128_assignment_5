"""Implementation of various Markov models and Language models for query likelihood document rankings.
"""

import math
import re
import random
import numpy as np
import pandas as pd

# Implementing query likelihood language model from
# https://nlp.stanford.edu/IR-book/html/htmledition/using-query-likelihood-language-models-in-ir-1.html

__author__ = "Adam Vekony"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = "Adam Vekony"
__license__ = "MIT"
__email__ = "avekony@westmont.edu"


class MarkovModel:
    def __init__(self, mode: str, text: list[str], n: int = 3):
        self._mode = mode.lower()
        self._doc_list = [re.sub('\W+\s*', ' ', doc).lower().strip() for doc in text]
        self._collection = " ".join(self._doc_list)
        if self._mode != 'char':
            self._collection = self._collection.split()

        if self._mode not in ("char", "word"):
            raise ValueError(f"Invalid model type: {mode}. Valid values are 'char' and 'word'.")
        self._n = n
        self._matrix = None

        def count_terms(self, full_corpus):
            if full_corpus is None:
                full_corpus = self._collection
            freq_dict = {}
            for token in full_corpus:
                if token in freq_dict.keys():
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1
            return freq_dict

        self._collection_counts = count_terms(self, full_corpus=self._collection)

    def _frequency_table(self, seq):
        if self._mode != 'char':
            seq = tuple(seq.split())
        table = {}
        for i in range(len(seq) - self._n):
            preceding = seq[i:i + self._n]
            next_item = seq[i + self._n]

            if table.get(preceding) is None:
                table[preceding] = {}
                table[preceding][next_item] = 1
            else:
                if table[preceding].get(next_item) is None:
                    table[preceding][next_item] = 1
                else:
                    table[preceding][next_item] += 1

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

    def generate(self, start, max_len=200):
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

    def _query_probability(self, c_len, query: str, doc: str, l=0.5):
        doc = doc.split()
        query = re.sub('\W+\s*', ' ', query).lower().strip().split()
        term_probs = []
        for term in query:
            prob = doc.count(term) / len(doc) if len(doc) > 0 else 0
            if prob == 0:
                c_prob = self._collection_counts[term] / c_len
                prob = (l * prob) + ((1 - l) * c_prob)
            term_probs.append(prob)

        query_prob = math.prod(term_probs)
        return query_prob

    def _most_probable_doc(self, query: str, l=0.5, corpus_percentage=0.3):
        if self._doc_list is not None and self._collection is not None:
            clen = len(self._collection)
            index = int(corpus_percentage * len(self._doc_list))
            max_qp = 0
            max_doc = ''
            max_index = None
            docs = self._doc_list[:index]
            for i, doc in enumerate(docs):
                qp = self._query_probability(c_len=clen, query=query, doc=doc, l=l)
                if qp > max_qp:
                    max_qp = qp
                    max_doc = doc
                    max_index = i
        return {'index': max_index, 'probability': max_qp, 'text': max_doc}


def main():
    arxiv_data = pd.read_csv(
        "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
    )
    abstracts = arxiv_data['summaries'].tolist()

    def search_terms():
        search_model = MarkovModel(mode='word', text=abstracts, n=1)
        fungi = search_model._most_probable_doc(query='fungi hallucinate', l=0.6, corpus_percentage=1)
        print(f'Query: fungi hallucinate\n{fungi}')
        ai = search_model._most_probable_doc(query='artificial intelligence', l=0.6, corpus_percentage=1)
        print(f'Query: artificial intelligence\n{ai}')
        cancer = search_model._most_probable_doc(query='lymphoma cancer', l=0.6, corpus_percentage=1)
        print(f'Query: lymphoma cancer\n{cancer}')
        transformer = search_model._most_probable_doc(query='attention transformer architecture GRU', l=0.6,
                                                      corpus_percentage=1)
        print(f'Query: attention transformer architecture GRU\n{transformer}')

    def generate_text():
        data = ' '.join(abstracts)
        data = re.sub('\W+\s*', ' ', data).lower()
        model = MarkovModel(mode='word', text=abstracts, n=3)
        model.train(data)
        q = model.generate(start='stereo matching is')
        print(q)
        p = model.generate(start='deep learning has')
        print(p)

        char_model = MarkovModel(mode='char', text=abstracts, n=4)
        char_model.train(data)
        a = char_model.generate(start='stereo matching is')
        print(a)
        b = char_model.generate(start='deep learning has')
        print(b)

    search_terms()
    generate_text()


if __name__ == '__main__':
    main()
