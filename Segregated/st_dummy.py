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



import streamlit as st



stop_words = set(stopwords.words('english')) 

# Load the Word2Vec model from the binary file
word2vec_model = KeyedVectors.load_word2vec_format('Segregated/word2vecmodel.bin', binary=True)

lstm_model = load_model('Segregated/final_lstm.keras')

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


# Streamlit app layout
st.title("ESSAY SCORER")

# Text input option
raw_text = st.text_area("Enter essay here", """""")


# Process raw text if provided
if st.button("Process Essay"):
    new_essays = raw_text
    preprocessed_new_essays = [sent2word(essay) for essay in new_essays]

    num_features = 300
    # Generating vectors for the new essays using the loaded Word2Vec model
    new_essay_vectors = getVecs(preprocessed_new_essays, word2vec_model, num_features)

    # Reshaping the vectors
    new_essay_vectors = np.array(new_essay_vectors)
    new_essay_vectors = np.reshape(new_essay_vectors, (new_essay_vectors.shape[0], 1, new_essay_vectors.shape[1]))

    # Predicting scores for the new essays using the LSTM model
    predictions = lstm_model.predict(new_essay_vectors)
    # predictions = np.around(predictions)
    print("Predictions:", predictions)

    score = predictions
    st.write(score)