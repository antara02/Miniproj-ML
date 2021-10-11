from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re
from tensorflow.keras.preprocessing.text import one_hot
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

df=pd.read_csv(r'/content/IMDB Dataset.csv')
dataset=df[:100]
toke=[]
def clean_text(dataset):
    clean_dataset=list()
    lines=dataset['review'].values.tolist()
    for text in lines:
        text=text.lower()
        pattern=re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text=pattern.sub('',text)
        text=re.sub(r"[,','.\"!@#$%^&*(){}?/;`~:<>+=-_]","",text)
        tokens=word_tokenize(text)
        # print(tokens)
        table=str.maketrans('','',string.punctuation)
        stripped=[w.translate(table)for w in tokens]
        words=[word for word in stripped if word.isalpha()]
        words=' '.join(words)
        clean_dataset.append(words)

    return clean_dataset
clean_dataset=clean_text(dataset)
wordnet= WordNetLemmatizer()
lemmetized_words=[]
for i in range(len(clean_dataset)):
    review = re.sub('[^a-zA-Z]', ' ', clean_dataset[i])
    review= review.split()
    review= [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    lemmetized_words.append(review)
print(lemmetized_words[:5])

vocab_size= 1000

onehot_repr= [one_hot(words, vocab_size)for words in lemmetized_words]
print(onehot_repr)

le= LabelEncoder()
xy= dataset['sentiment'].values.tolist()
le.fit(['positive', 'negative'])
new_xy=le.transform(xy)

df= pd.DataFrame({'review':onehot_repr, 'sentiment':new_xy})
df.to_csv('new_data.csv')
