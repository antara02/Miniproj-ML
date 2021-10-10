#here we'll write the preprocessed code
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import string
nltk.download('punkt')
df=pd.read_csv(r'data/IMDB Dataset.csv')
dataset=df[:100]

def clean_text(dataset):
    clean_dataset=list()
    lines=dataset['review'].values.tolist()
    for text in lines:
        text=text.lower()
        pattern=re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text=pattern.sub('',text)
        text=re.sub(r"[,','.\"!@#$%^&*(){}?/;`~:<>+=-_]","",text)
        tokens=word_tokenize(text)
        table=str.maketrans('','',string.punctuation)
        stripped=[w.translate(table)for w in tokens]
        words=[word for word in stripped if word.isalpha()]
        words=' '.join(words)
        clean_dataset.append(words)
    return clean_dataset
clean_dataset=clean_text(dataset)
print(clean_dataset[:5]) 
