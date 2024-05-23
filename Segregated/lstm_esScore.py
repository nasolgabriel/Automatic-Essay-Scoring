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

new_essays = ["""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ut arcu eget magna fermentum feugiat nec et enim. Aliquam erat volutpat. Nulla facilisi. Vivamus ornare massa et eros consequat, quis fermentum neque tincidunt. Duis ac nisi vitae est lacinia varius nec in purus. Morbi non tellus auctor, molestie ex sed, accumsan mauris. Nulla facilisi. Proin feugiat, lorem vel tempor consectetur, massa eros sodales quam, in rutrum ipsum velit non libero. Donec in ex nec velit tincidunt facilisis. Nunc ut neque a tortor bibendum facilisis. Nulla facilisi. Integer ac neque id ipsum tincidunt elementum. Nam et mi a magna posuere ullamcorper. Sed non nisi nec enim gravida faucibus eu id lacus. Praesent vitae turpis id felis rutrum varius. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; In bibendum quam ut mi viverra, nec cursus libero feugiat. Mauris efficitur tortor at tortor scelerisque, quis tincidunt sem pulvinar. Vivamus euismod lacinia magna, at sollicitudin eros vestibulum id. Curabitur sed interdum ex. Phasellus tincidunt augue nec augue gravida, at malesuada leo consectetur. Sed vitae neque vitae leo tristique vehicula vitae at dolor. Sed sed arcu nec justo convallis feugiat vitae a velit. Ut sit amet commodo ex. Curabitur hendrerit enim vel ex fringilla, sed volutpat nulla rhoncus. Sed interdum tellus non velit rutrum malesuada. Sed auctor, nunc ut fermentum lobortis, felis sapien dapibus nisi, nec rutrum nunc neque quis ipsum. Maecenas dapibus condimentum turpis, sed ultricies libero."""]

preprocessed_new_essays = [sent2word(essay) for essay in new_essays]

num_features = 300
# Generating vectors for the new essays using the loaded Word2Vec model
new_essay_vectors = getVecs(preprocessed_new_essays, word2vec_model, num_features)

# Reshaping the vectors
new_essay_vectors = np.array(new_essay_vectors)
new_essay_vectors = np.reshape(new_essay_vectors, (new_essay_vectors.shape[0], 1, new_essay_vectors.shape[1]))

lstm_model = load_model('Segregated/final_lstm.keras')

# Predicting scores for the new essays using the LSTM model
predictions = lstm_model.predict(new_essay_vectors)
# predictions = np.around(predictions)
print("Predictions:", predictions)

pred = predictions