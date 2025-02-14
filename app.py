# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:30:41 2025

@author: subik
"""

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained Keras model using load_model (for .h5 or SavedModel format)
loaded_model = load_model("C:/Users/subik/Downloads/sentiment_model.h5")

# Load tokenizer using pickle
with open("C:/Users/subik/Downloads/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function for sentiment prediction
def sentiment_prediction(input_review):
    sequence = tokenizer.texts_to_sequences([input_review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = loaded_model.predict(padded_sequence)
    if prediction[0][0] > 0.5:
        return "Positive"
    else:
        return "☹️ Negative"

# Streamlit UI
def main():
    # Giving a title
    st.title('IMDB Sentiment Analysis')
    
    # Getting user input
    review = st.text_area("Enter a movie review:")
    
    # Predicting sentiment when button is clicked
    diagnosis = ''
    if st.button('Predict Sentiment'):
        if review.strip():
            diagnosis = sentiment_prediction(review)
        else:
            st.warning("Please enter a review!")
    
    # Display result
    st.success(f"Sentiment: {diagnosis}")

if __name__ == '__main__':
    main()
