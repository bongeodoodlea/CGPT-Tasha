#streamlit run streamlit_app.py
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import tensorflow as tf
import random

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    intents = json.loads(open('intents.json', encoding="utf8").read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    model = tf.keras.models.load_model('chatbot_model.h5')
    return intents, words, classes, model

# Tokenize and lemmatize input sentence
def tokenize_and_lemmatize(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

# Bag-of-words function
def bow(sentence, words):
    sentence_words = tokenize_and_lemmatize(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence, model, words, classes):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Check if results is empty
    if not results:
        return [{"intent": "Unknown", "probability": 0}]
    
    return_list = []
    for r in results:
        # Check if classes[r[0]] exists
        if r[0] < len(classes):
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        else:
            print(f"Error: Class index {r[0]} out of range")
    return return_list

# Get response
def get_response(ints, intents_json):
    # Check if ints is empty
    if not ints:
        return random.choice(["I didn't understand that.", "Can you please rephrase?", "I'm not sure with the answer."])
    tag = ints[0].get('intent', 'Unknown')
    list_of_intents = intents_json["intents"]
    result = "Unknown response"  # Define result here
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Check if responses exists and is not empty
            if 'responses' in i and i["responses"]:
                result = random.choice(i["responses"])
                break
            else:
                print(f"Error: No responses found for intent {tag}")
                result = random.choice(["I didn't understand that.", "Can you please rephrase?", "I'm not sure I understand."])
    return result

# Streamlit app
st.header("Bongeodoodles(Tasha) the Chatbot")
intents, words, classes, model = load_model_and_data()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
def chat():
    user_input =st.chat_input("Say something", key = "input")
    if user_input:
        with st.chat_message(name="user", avatar="🥸"):
            st.write(user_input)
        # Store user message in chat history
        st.session_state.chat_history.append({"User": user_input})

        ints = predict_class(user_input, model, words, classes)
        response = get_response(ints, intents)
        with st.chat_message(name="Ava", avatar="👽"): #784
            st.write(response)
        # Store bot response in chat history
        st.session_state.chat_history.append({"Tasha": response})
        
    # Display chat history
    for message in st.session_state.chat_history:
        for name, text in message.items():
            if name == "User":
                with st.chat_message(name="User", avatar="🥸"):
                    st.write(text)
            elif name == "Tasha":
                with st.chat_message(name="Tasha", avatar="👽"):
                    st.write(text)

chat()
