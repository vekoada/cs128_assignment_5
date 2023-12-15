code = perceive

# Assignment 5: Open-ended IR Technique

**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

_Assistant Professor_ Mike Ryu (mryu@westmont.edu)

## Author Information

- **Name(s)**: Adam Vekony
- **Email(s)**: avekony@westmont.edu
- **License**: MIT License
- **Presentation Link**: https://docs.google.com/presentation/d/1YbwxjzsjHQE020GQNWr3qqYGNpkruC5CiQu-376m2G8/edit?usp=sharing
- **Doc with results**: https://docs.google.com/document/d/1xAdbby9j1XHM76_XtG8MBluGfWl0SXKd_VSe3KIQQKU/edit?usp=sharing

# Guide
## MarkovModel

MarkovModel is a Python class that implements a query likelihood language model based on Markov chains. This class allows you to create language models for generating text and estimating the probability of a query given a document. The model supports both character-based and word-based representations with customizable order.

## Project Structure

- `src/models.py`: Contains the implementation of the `MarkovModel` class. Also includes main() which defines example usage of the MarkovModel class.
- `data/`: An empty directory where you can store your training data.
- `test/`: An empty directory intended for future testing.

## Installation

No installation required. Simply include the `MarkovModel` class in your project, and you're ready to go.

## Usage

1. Import the `MarkovModel` class:
   ```python
   from src.models import MarkovModel
2. Create an instance of the `MarkovModel` class by providing the mode ('char' or 'word') and the training text:
   ```python
   text_data = [...]  # List of training documents
   markov_model = MarkovModel(mode='word', text=text_data, n=3)
3. Train the model with the training data:
	```python
	markov_model.train(text=text_data)
4. Generate text using the trained model:
	```python
	generated_text = markov_model.generate(start='The quick brown fox', max_len=200)
	print(generated_text)
5. Estimate the probability of a query given a document:
	```python
	query = 'natural language processing'
	result = markov_model._most_probable_doc(query=query, l=0.7, corpus_percentage=1.0)
	print(result)

## Data and Testing
The `data/` directory is intended for storing your training data. Feel free to populate this directory with text documents to train your model!

The `test/` directory is currently empty, and more thorough testing should be implemented in the future to ensure the reliability of the MarkovModel class.

Feel free to contribute by adding your own test cases or improving the model based on your specific use case!

### Acknowledgements and Sources
While working on this assignment, I used the following resources:

- For **implementation of the Markov model** portion (transition matrix, predict, generate) of the project, I heavily referenced this blog post by Educative. https://www.educative.io/blog/deep-learning-text-generation-markov-chains 
- For the dataset, I used this **Arxiv Paper Abstract data** posted on Kaggle: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
- I used a version of the dataset hosted by this Github user and utilized some code in this jupyter notebook to load in from his link: https://github.com/soumik12345/multi-label-text-classification/blob/master/multi_label_trainer_tfidf.ipynb
- I used chapter 12 of *Introduction to Information Retrieval*, specifically the following section as a mathematical **reference for the implementation of the query likelihood estimation** portion of the project: https://nlp.stanford.edu/IR-book/html/htmledition/estimating-the-query-generation-probability-1.html
- Here is the home page of the textbook: https://nlp.stanford.edu/IR-book/information-retrieval-book.html
- No code was generated with any language model for this project. However, I did use ChatGPT to write the **docstrings for the class and its methods**, as well as the usage portion of this README. Then, I went through its output and corrected any errors I found. The entire conversation can be found here: https://chat.openai.com/share/397c1a4a-eeed-459d-9a91-ea81eefe5610
