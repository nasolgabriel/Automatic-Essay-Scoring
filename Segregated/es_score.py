import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import nltk
import re
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords 
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# nltk.download('stopwords')
# nltk.download('punkt')

# Load the necessary models and resources

word2vec_model = KeyedVectors.load_word2vec_format('d:\\GitHub_repositories\\pythonProject\\LSTM\\Automatic-Essay-Scoring\\Segregated\\word2vecmodel_esset.bin', binary=True)

# using own dataset
lstm_model = load_model('d:\\GitHub_repositories\\pythonProject\\LSTM\\Automatic-Essay-Scoring\\Segregated\\owndata_lstm.keras')

# # using online dataset
# lstm_model = load_model('d:\\GitHub_repositories\\pythonProject\\LSTM\\Automatic-Essay-Scoring\\Segregated\\final_lstm.keras')

stop_words = set(stopwords.words('english'))

# Define the functions for preprocessing and vectorizing the essay
def sent2word(x):
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = [w for w in x.split() if w not in stop_words]
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for sentence in raw:
        if len(sentence) > 0:
            final_words.extend(sent2word(sentence))
    return final_words

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0
    for word in words:
        if word in model:
            noOfWords += 1
            vec = np.add(vec, model[word])
    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)
    return vec

# Streamlit app
st.title("Essay Grading App")
st.write("Input your essay below and click the button to grade it.")

# Text input for essay
user_input = st.text_area("Enter your essay here:", """ """, key="user_input", height=250)

# Button to start grading the essay
if st.button("Grade Essay"):
    if user_input:
        preprocessed_essay = essay2word(user_input)
        num_features = 300
        essay_vector = makeVec(preprocessed_essay, word2vec_model, num_features)
        essay_vector = np.array(essay_vector)
        essay_vector = np.reshape(essay_vector, (1, 1, essay_vector.shape[0]))
        
        # Predicting the score using the LSTM model
        prediction = lstm_model.predict(essay_vector)
        prediction = np.around(prediction).flatten()

        # Display the score
        st.write("Predicted Score:", prediction[0])
        print(prediction)

    else:
        st.write("Please enter an essay to grade.")
