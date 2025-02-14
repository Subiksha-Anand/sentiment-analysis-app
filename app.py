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
loaded_model = load_model("sentiment_model (1).h5")
try:
  
    loaded_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(" Model successfully loaded!")
except Exception as e:
    print(f" Error loading model: {e}")


# Load tokenizer using pickle
with open("tokenizer (1).pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
import os

print("Model exists:", os.path.exists("sentiment_model (1).h5"))
print("Tokenizer exists:", os.path.exists("tokenizer (1).pkl"))


# Function for sentiment prediction
def sentiment_prediction(input_review):
    sequence = tokenizer.texts_to_sequences([input_review])
    if not sequence or not sequence[0]:  # Ensuring sequence is not empty
      st.error(" The input text does not contain recognizable words!")
    else:
      padded_sequence = pad_sequences(sequence, maxlen=200)
      prediction = loaded_model.predict(padded_sequence)
      sentiment = "Positive " if prediction[0][0] > 0.5 else "Negative "
      st.success(f"Sentiment: {sentiment}")
 
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
