import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
#nltk.download('stopwords')
#nltk.download('punkt')

from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model


stop_words = set(stopwords.words('english')) 
def sent2word(x):
    x=re.sub("[^A-Za-z]"," ",x)
    x.lower()
    filtered_sentence = [] 
    words=x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sent2word(i))
    return final_words


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0
    for i in words:
        if i in model:
            noOfWords += 1
            vec = np.add(vec, model[i])
    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)
    return vec


def getVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c+=1
    return essay_vecs


# Load the Word2Vec model from the binary file
word2vec_model = KeyedVectors.load_word2vec_format('Segregated/word2vecmodel.bin', binary=True)

new_essays = ["""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sed turpis quis mauris ullamcorper fringilla. Curabitur at tortor ac leo pulvinar malesuada in vitae felis. Aliquam quis lacus a justo iaculis vulputate sit amet nec lacus. Quisque non rhoncus nisi. Donec lobortis metus id libero vulputate suscipit. Praesent a sollicitudin odio. Ut dapibus euismod nisi fermentum lobortis. Duis lobortis pharetra dignissim. Duis vel dolor at ex molestie condimentum. Morbi aliquet eros a accumsan viverra. Etiam ac erat molestie, lobortis augue ut, auctor lorem. Aenean vitae varius augue. Maecenas sit amet ultricies nibh, et faucibus nibh. Nullam tristique quam urna, vehicula lobortis nibh interdum vel. Nunc sed nunc mauris. Maecenas diam ex, ornare sit amet vehicula in, sagittis ut tortor. Donec luctus dignissim nibh, at imperdiet est tincidunt ut. Sed venenatis faucibus dolor et tincidunt. Nullam sit amet sapien id libero scelerisque accumsan non id enim. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus."""]

preprocessed_new_essays = [sent2word(essay) for essay in new_essays]

num_features = 300
# Generating vectors for the new essays using the loaded Word2Vec model
new_essay_vectors = getVecs(preprocessed_new_essays, word2vec_model, num_features)

# Reshaping the vectors
new_essay_vectors = np.array(new_essay_vectors)
new_essay_vectors = np.reshape(new_essay_vectors, (new_essay_vectors.shape[0], 1, new_essay_vectors.shape[1]))

lstm_model = load_model('Automatic-Essay-Scoring/Segregated/owndata_lstm.keras')

# Predicting scores for the new essays using the LSTM model
predictions = lstm_model.predict(new_essay_vectors)
# predictions = np.around(predictions)
print("Predictions:", predictions)

pred = predictions