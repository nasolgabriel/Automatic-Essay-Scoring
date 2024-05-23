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
word2vec_model = KeyedVectors.load_word2vec_format('Segregate/word2vecmodel.bin', binary=True)

new_essays = ["In the grand tapestry of human innovation, few threads shine as brightly as the computer. A marvel of modern ingenuity, the computer stands as a testament to humanity's insatiable curiosity and unyielding drive for progress. It is more than just a machine; it is a gateway to boundless possibilities, a portal to realms of knowledge and imagination.At its core, the computer is a symphony of silicon and circuitry, a labyrinth of ones and zeros woven into the fabric of our digital age. Yet, its true essence transcends mere hardware and software. It is a catalyst for creativity, a canvas upon which dreams are painted in pixels. From the intricate lines of digital art to the melodic harmonies of electronic music, the computer empowers artists to push the boundaries of expression.But the computer is not just a tool for the creative mind; it is also a beacon of connectivity in an increasingly interconnected world. With a few keystrokes, we can traverse vast virtual landscapes, connecting with friends and strangers alike across oceans and continents. Social media platforms, online forums, and virtual communities serve as digital watering holes where ideas flow freely, and bonds are forged in the digital ether.Moreover, the computer is a wellspring of knowledge, a repository of humanity's collective wisdom. With a few clicks, we can access a wealth of information spanning the breadth of human understanding. From the mysteries of the cosmos to the intricacies of quantum mechanics, the computer opens doors to realms once reserved for the privileged few.Yet, for all its wonders, the computer is not without its pitfalls. In the labyrinth of cyberspace, dangers lurk in the shadows, from cybercrime to information warfare. As we navigate this digital frontier, we must remain vigilant, guarding against the dark forces that seek to exploit our vulnerabilities.In the end, the computer is more than just a machine; it is a reflection of humanity itself—flawed yet full of promise. It is a tool for both creation and destruction, a mirror that reflects the best and worst of who we are. As we stand on the cusp of a new digital era, let us wield this power wisely, harnessing the potential of the computer to shape a brighter tomorrow for generations to come."]

preprocessed_new_essays = [sent2word(essay) for essay in new_essays]

num_features = 300
# Generating vectors for the new essays using the loaded Word2Vec model
new_essay_vectors = getVecs(preprocessed_new_essays, word2vec_model, num_features)

# Reshaping the vectors
new_essay_vectors = np.array(new_essay_vectors)
new_essay_vectors = np.reshape(new_essay_vectors, (new_essay_vectors.shape[0], 1, new_essay_vectors.shape[1]))

lstm_model = load_model('Segregate/final_lstm.h5')

# Predicting scores for the new essays using the LSTM model
predictions = lstm_model.predict(new_essay_vectors)
# predictions = np.around(predictions)
print("Predictions:", predictions)