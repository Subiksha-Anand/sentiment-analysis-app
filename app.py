# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:30:41 2025

@author: subik
"""

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model (Ensure files are in the same directory)
loaded_model = load_model("sentiment_model.h5", compile=False)
loaded_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function for sentiment prediction
def sentiment_prediction(input_review):
    sequence = tokenizer.texts_to_sequences([input_review])

    if not sequence or not sequence[0]:  # Handle empty sequences
        return "⚠️ The input text does not contain recognizable words!"

    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = loaded_model.predict(padded_sequence)

    return "😊 Positive" if prediction[0][0] > 0.5 else "☹️ Negative"

# Streamlit UI
def main():
    st.title('IMDB Sentiment Analysis')

    review = st.text_area("Enter a movie review:")

    if st.button('Predict Sentiment'):
        if review.strip():
            result = sentiment_prediction(review)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter a review!")

if __name__ == '__main__':
    main()

