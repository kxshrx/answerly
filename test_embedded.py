from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use embed_query for a single string query
embedding = embedding_model.embed_query("This is a test sentence to check the embedding model.")

st.write(embedding)
