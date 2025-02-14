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
from tensorflow.keras.models import load_model

# Try loading the model with error handling
try:
    loaded_model = load_model("sentiment_model.h5")
    
    if loaded_model is None:
        raise ValueError("Model is None. It may be corrupted.")
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# Load tokenizer using pickle
# Try loading the tokenizer with error handling
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    if tokenizer is None:
        raise ValueError("Tokenizer is None. The file may be corrupted.")
    
    print("✅ Tokenizer loaded successfully!")
    print(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")


# Function for sentiment prediction
def sentiment_prediction(input_review):
    # Convert review to sequence
    sequence = tokenizer.texts_to_sequences([input_review])

    # Debugging: Print sequence output
    print(f"Input Review: {input_review}")
    print(f"Tokenized Sequence: {sequence}")

    # If sequence is empty, return a warning
    if not sequence or len(sequence[0]) == 0:
        return "⚠️ No recognizable words found in input! Try using different words."

    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=200)

    # Debugging: Print padded sequence
    print(f"Padded Sequence: {padded_sequence}")

    # Predict sentiment
    prediction = loaded_model.predict(padded_sequence)

    # Debugging: Print prediction output
    print(f"Model Prediction: {prediction}")

    return "😊 Positive" if prediction[0][0] > 0.5 else "☹️ Negative"


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
