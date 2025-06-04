import os
import streamlit as st
import pickle
from pathlib import Path

st.title("Debug App")

# Check environment variables
st.write("## Environment Variables")
if os.environ.get("OPENAI_API_KEY"):
    st.write("✅ OPENAI_API_KEY is set")
else:
    st.error("❌ OPENAI_API_KEY is not set")

# Try to find data directory
st.write("## Data Directory")
possible_paths = [
    "data/processed_data",
    "app/data/processed_data",
    "/data/processed_data", 
    "/app/data/processed_data"
]

for path in possible_paths:
    if os.path.exists(path):
        st.write(f"✅ Found data directory at: {path}")
        
        # Check for pickle files
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(chunks_path):
            st.write(f"✅ Found chunks.pkl: {os.path.getsize(chunks_path) / (1024*1024):.2f} MB")
            
            # Try to load 
            try:
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                st.write(f"✅ Successfully loaded chunks: {len(chunks)} items")
            except Exception as e:
                st.error(f"❌ Error loading chunks.pkl: {str(e)}")
        else:
            st.error(f"❌ chunks.pkl not found in {path}")
            
        embedded_docs_path = os.path.join(path, "embedded_docs.pkl")
        if os.path.exists(embedded_docs_path):
            st.write(f"✅ Found embedded_docs.pkl: {os.path.getsize(embedded_docs_path) / (1024*1024):.2f} MB")
            
            # Try to load 
            try:
                with open(embedded_docs_path, "rb") as f:
                    embedded_docs = pickle.load(f)
                st.write(f"✅ Successfully loaded embedded_docs: {len(embedded_docs)} items")
            except Exception as e:
                st.error(f"❌ Error loading embedded_docs.pkl: {str(e)}")
        else:
            st.error(f"❌ embedded_docs.pkl not found in {path}")
            
        break
else:
    st.error("❌ Could not find data directory")

st.write("## System Info")
import sys
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Directory contents: {os.listdir('.')}")

st.write("Debug complete.") 