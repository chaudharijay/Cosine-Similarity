# Step 1: Building the similarity checker: code breakdown

from sentence_transformers import SentenceTransformer
import torch

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Define two example sentences
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A quick brown dog jumps over the lazy fox"

# Encode sentences and compute similarity score
embeddings1 = model.encode([sentence1], convert_to_tensor=True)
embeddings2 = model.encode([sentence2], convert_to_tensor=True)
cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

# Print similarity score
print(f"Similarity score: {cosine_similarities.item()}")



# Step 2: Building the Streamlit app


import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import time
from PIL import Image
import base64
import os

# Load the pre-trained sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_similarity_score(source_sentence, compare_sentences):
  """
  Computes cosine similarity scores between a source sentence and a list of compare sentences.

  Args:
      source_sentence (str): The source sentence.
      compare_sentences (list): A list of sentences to compare against the source.

  Returns:
      list: A list of cosine similarity scores.
  """
  source_embedding = model.encode(source_sentence, convert_to_tensor=True)
  compare_embeddings = model.encode(compare_sentences, convert_to_tensor=True)
  cosine_similarities = torch.nn.functional.cosine_similarity(source_embedding, compare_embeddings)
  return cosine_similarities.tolist()


# Set page configuration
st.set_page_config(
    page_title="Sentence Similarity Analyzer",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* Dark background */
        color: #c9d1d9; /* Light text color */
    }
    .st-h1 {
        color: #ffffff; /* White title */
    }
    .st-expanderHeader {
        background-color: #374151; /* Darker expander header */
        color: #c9d1d9; /* Light text color */
    }
    .st-expanderBody {
        background-color: #282c34; /* Dark background inside expander */
    }
    .st-button {
        background-color: #4285f4; /* Blue button */
        color: #ffffff; 
    }
    .st-error {
        background-color: #ff5722; /* Red error background */
        color: #ffffff;
    }
    .st-success {
        background-color: #4caf50; /* Green success background */
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Add a container for the main content
with st.container():
  st.title("Sentence Similarity Analyzer")
  st.markdown("""
      This app analyzes the similarity between a source sentence and a set of comparison sentences 
      using a pre-trained sentence embedding model.
  """)

  # Add a container for the input fields
  with st.container():
    col1, col2 = st.columns([1, 1])

    with col1:
      with st.expander("Source Sentence"):
        with st.spinner("Analyzing source sentence..."): 
          source_sentence = st.text_area(label="Enter the source sentence:", height=100) 

    with col2:
      with st.expander("Comparison Sentences"):
        compare_sentences = []
        compare_sentence1 = st.text_area(label="Enter sentence 1 to compare (optional):", key="compare_sentence_1", height=100)
        if compare_sentence1:
          compare_sentences.append(compare_sentence1)

        # Display additional comparison sentences below the first one
        compare_sentence2 = st.text_area(label="Enter sentence 2 to compare (optional):", key="compare_sentence_2", height=100)
        if compare_sentence2:
          compare_sentences.append(compare_sentence2)
        compare_sentence3 = st.text_area(label="Enter sentence 3 to compare (optional):", key="compare_sentence_3", height=100)
        if compare_sentence3:
          compare_sentences.append(compare_sentence3)


  # Add a container for the results
  with st.container():
    st.markdown("<h2 style='text-align:center;'>Analyze</h2>", unsafe_allow_html=True) 
    if st.button("Analyze", use_container_width=True):
      if not source_sentence:
        st.error("Please enter a source sentence.")
      else:
        with st.spinner("Calculating similarity scores..."):
          # Simulate a loading time (replace with actual computation)
          time.sleep(1)
          scores = compute_similarity_score(source_sentence, compare_sentences)

        st.success("Analysis Complete!")

        # Display results
        col1, col2 = st.columns([1, 2])
        with col1:
          st.subheader("Similarity Scores")
          for i, score in enumerate(scores):
            st.write(f"{i+1}. {score:.2f}")

        with col2:  
          st.subheader("Sentences")
          st.write("**Source Sentence:**")
          st.write(source_sentence)
          for i, compare_sentence in enumerate(compare_sentences):
            st.write(f"**Compare Sentence {i+1}:**")
            st.write(compare_sentence)

  # Add a footer
  with st.container():
    st.markdown("---")
    st.write("Developed by Yash Chaudhari ")
    st.write("Powered by Streamlit and Sentence Transformers")