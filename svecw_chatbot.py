import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Set page config
st.set_page_config(page_title="Svecw College Chatbot", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load CSV data
csv_url = "svcew_details.csv"
try:
    df = pd.read_csv(csv_url)
    df = df.fillna("")
    df['Question'] = df['Question'].str.lower()
    df['Answer'] = df['Answer'].str.lower()
except Exception as e:
    st.error(f"Failed to load the CSV file. Please try again later. Error: {e}")
    st.stop()

# Vectorizer for similarity matching
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Load API key securely
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Google API key is missing. Please set it as an environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    return df.iloc[best_match_index]['Answer'] if best_match_score > 0.3 else None

# UI Title and Description
st.title("Svecw College Chatbot")
st.write("Welcome to the College Chatbot! Ask me anything about the college.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)
    if closest_answer:
        response_text = closest_answer
    else:
        try:
            response = model.generate_content(prompt)
            response_text = response.text if response else "Sorry, I couldn't generate a response."
        except Exception as e:
            response_text = f"Sorry, I couldn't generate a response. Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)
