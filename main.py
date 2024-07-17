import streamlit as st
import json
import difflib
from transformers import pipeline

# Load the corpus from JSON file
with open('Sample Question Answers.json', 'r') as file:
    corpus = json.load(file)

# Initialize Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Streamlit UI
st.title("Wine Business Chatbot")
st.write("Ask me anything about our wines!")

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = []


# Function to find the best matching question in the corpus
def find_best_match(question, corpus):
    questions = [item['question'] for item in corpus]
    best_match = difflib.get_close_matches(question, questions, n=1, cutoff=0.5)
    if best_match:
        for item in corpus:
            if item['question'] == best_match[0]:
                return item['answer']
    return None


# Function to handle user input and generate a response
def get_response(user_input, history):
    context = " ".join([f"{entry['role']}: {entry['content']}" for entry in history])
    context += f"\nuser: {user_input}\nbot: "

    response = find_best_match(user_input, corpus)

    if response:
        return response
    else:
        # Use Hugging Face QA model for out-of-corpus questions
        corpus_text = " ".join([item['answer'] for item in corpus])
        qa_input = {
            'question': user_input,
            'context': corpus_text
        }
        result = qa_pipeline(qa_input)
        answer = result['answer']
        # If the score is too low, fall back to a default response
        if result['score'] < 0.5:
            return "Please contact the business directly."
        return answer


# User input
user_input = st.text_input("You: ", "")

if user_input:
    # Append user input to conversation history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Generate and append bot response to conversation history
    bot_response = get_response(user_input, st.session_state.history)
    st.session_state.history.append({"role": "bot", "content": bot_response})

    # Display the conversation
    # Display the conversation
    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")
            # Display rating UI for bot's response
            if message.get("rating") is None:
                message["rating"] = st.slider("Rate this answer:", 0, 5,
                                              key=f"{message['content']}_{len(st.session_state.history)}")
                st.session_state.history[-1]["rating"] = message["rating"]

# Clear history button
if st.button("Clear Conversation"):
    st.session_state.history = []
