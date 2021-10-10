#here we'll write the preprocessed code
import pandas as pd
import numpy  as  np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
# from sklearn.datasets import fetch_20newsgroups
df= pd.read_csv("data/IMDB Dataset.csv")
print(df[:5])

