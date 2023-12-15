"""
MarkovModel: A Python class implementing a query likelihood language model based on Markov chains.

This class allows the creation of a language model, specifically a Markov model, for generating text and
calculating the probability of a query given a document. The model supports both character-based and
word-based representations with customizable order. It also provides methods for training the model,
generating text, and estimating query probabilities.
"""

# Imports
import math
import re
import random
import numpy as np
import pandas as pd

# Implementing query likelihood language model as described in:
# https://nlp.stanford.edu/IR-book/html/htmledition/using-query-likelihood-language-models-in-ir-1.html

__author__ = "Adam Vekony"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = "Adam Vekony"
__license__ = "MIT"
__email__ = "avekony@westmont.edu"


class MarkovModel:
    """
    A class representing a Markov Model for language modeling.

    Attributes:
    - _mode (str): The mode of the model, either 'char' for character-based or 'word' for word-based.
    - _doc_list (list): A list of preprocessed documents used for training the model.
    - _collection (str or list): The combined representation of the document collection.
    - _n (int): The order of the Markov model, indicating the number of preceding items considered.
    - _matrix (dict): The transition matrix representing the probabilities of sequences in the model.
    - _collection_counts (dict): The frequency counts of terms in the document collection.

    Methods:
    - __init__(self, mode: str, text: list[str], n: int = 3): Initialize the Markov model with the given parameters.
    - train(self, text): Train the Markov model using the provided text data.
    - generate(self, start, max_len=200): Generate text using the trained Markov model starting from the given seed.
    - _frequency_table(self, seq): Generate a frequency table based on the input sequence.
    - _transition_matrix(self, table): Convert a frequency table into a transition matrix.
    - _predict(self, seq): Predict the next item in the sequence based on the Markov model.
    - _query_probability(self, c_len, query: str, doc: str, l=0.5): Calculate the probability of a query given a document.
    - _most_probable_doc(self, query: str, l=0.5, corpus_percentage=1.0): Find the most probable document in the collection for a given query.
    """
    def __init__(self, mode: str, text: list[str], n: int = 3):
        """
        Initialize the Markov model with the given parameters.

        Parameters:
        - mode (str): The mode of the model, either 'char' for character-based or 'word' for word-based.
        - text (list): List of document strings used for training the model.
        - n (int): The order of the Markov model (default is 3).
        """
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
            """
            Count the frequency of terms in the given corpus.

            Parameters:
            - full_corpus: The corpus for which term frequencies are counted.

            Returns:
            - dict: A dictionary with term frequencies.
            """
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
        """
        Generate a frequency table based on the input sequence.

        Parameters:
        - seq (str): Input sequence for which the frequency table is generated.

        Returns:
        - dict: Frequency table where keys are n-grams and values are dictionaries
          representing the next item and its count.

        Note:
        - For word-based models, the input sequence is converted to a tuple of words.
        """
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
        """
        Convert a frequency table into a transition matrix.

        Parameters:
        - table (dict): Frequency table generated by _frequency_table method.

        Returns:
        - dict: Transition matrix where probabilities replace counts in the frequency table.
        """
        for seq in table.keys():
            total_count = float(sum(table[seq].values()))
            for el_count in table[seq].keys():
                table[seq][el_count] = table[seq][el_count] / total_count

        return table

    def train(self, text):
        """
        Train the Markov model by generating a transition matrix from the input text.

        Parameters:
        - text (list): List of documents used for training the model.
        """
        self._matrix = self._transition_matrix(self._frequency_table(text))

    def _predict(self, seq):
        """
        Predict the next item in the sequence based on the Markov model.

        Parameters:
        - seq (str or tuple): Input sequence for which the next item is predicted.

        Returns:
        - str: The predicted next item.
        """
        seq = seq[-self._n:]
        if self._matrix.get(seq) is None:
            seq = random.choice(list(self._matrix.keys()))
        possibilities = list(self._matrix[seq].keys())
        probs = list(self._matrix[seq].values())

        return np.random.choice(possibilities, p=probs)

    def generate(self, start, max_len=200):
        """
        Generate text using the trained Markov model starting from the given seed.

        Parameters:
        - start (str): The seed text to start the generation.
        - max_len (int): Maximum length of the generated text (default is 200).

        Returns:
        - str: The generated text.
        """
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
        """
        Calculate the probability of a query given a document using a language model.

        Parameters:
        - c_len (int): Length of the entire document collection.
        - query (str): The query string.
        - doc (str): The document string.
        - l (float): Lambda, a parameter for balancing term and collection probabilities (default is 0.5).

        Returns:
        - float: The probability of the query given the document.
        """
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

    def _most_probable_doc(self, query: str, l=0.5, corpus_percentage=1.0):
        """
        Find the most probable document in the collection for a given query.

        Parameters:
        - query (str): The query string.
        - l (float): A parameter for balancing term and collection probabilities (default is 0.5).
        - corpus_percentage (float): The percentage of the document collection to consider (default is 1.0).

        Returns:
        - dict: A dictionary containing the index, probability, and text of the most probable document.
        """
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
