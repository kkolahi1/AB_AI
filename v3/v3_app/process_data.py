import os
import json
import shutil
import pickle
import time
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clean_directory(directory_path):
    """Clean a directory by removing all files and subdirectories"""
    path = Path(directory_path)
    if path.exists():
        print(f"Cleaning directory: {directory_path}")
        shutil.rmtree(path)
    
    # Wait a moment to ensure OS releases the directory handles
    time.sleep(1)
    
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created clean directory: {directory_path}")

def load_preprocessed_chunks():
    """Load the preprocessed chunks from JSON file"""
    print("Loading preprocessed chunks...")
    
    # Path to the saved JSON file
    chunks_file = "all_chunks_95percentile.json"
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=chunk['page_content'], 
            metadata=chunk['metadata']
        ) 
        for chunk in chunks_data
    ]
    
    print(f"Loaded {len(documents)} preprocessed chunks.")
    return documents

def create_vectorstore_and_retrievers(documents):
    """Create vectorstore and retrievers using the latest chunking strategy."""
    
    try:
        # Initialize embedding model
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="kamkol/ab_testing_finetuned_arctic_ft-36dfff22-0696-40d2-b3bf-268fe2ff2aec"
        )
        
        # Create Qdrant vectorstore
        print("Creating Qdrant vectorstore...")
        qdrant_vectorstore = Qdrant.from_documents(
            documents,
            embedding_model,
            location=":memory:",
            collection_name="kohavi_ab_testing_pdf_collection",
        )
        
        # Create BM25 retriever
        print("Creating BM25 retriever...")
        texts = [doc.page_content for doc in documents]
        tokenized_corpus = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_retriever = BM25Retriever.from_texts(texts, metadatas=[doc.metadata for doc in documents])
        bm25_retriever.k = 10  # Set top-k results
        
        print(f"Successfully created vectorstore with {len(documents)} documents")
        print(f"BM25 retriever created with {len(texts)} texts")
        
        return qdrant_vectorstore, bm25_retriever, embedding_model
        
    except Exception as e:
        print(f"Error creating vectorstore and retrievers: {e}")
        raise

def save_processed_data(qdrant_vectorstore, bm25_retriever, embedding_model, documents):
    """Save all processed data files needed for the app"""
    print("Saving processed data...")
    
    # Create processed data directory
    processed_data_dir = Path("data/processed_data")
    clean_directory(processed_data_dir)
    
    # Save documents as chunks
    print("Saving document chunks...")
    with open(processed_data_dir / "chunks.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    # Save BM25 retriever
    print("Saving BM25 retriever...")
    with open(processed_data_dir / "bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    # Save embedding model info (we'll reinitialize it in the app)
    print("Saving embedding model info...")
    embedding_info = {
        "model_name": "kamkol/ab_testing_finetuned_arctic_ft-36dfff22-0696-40d2-b3bf-268fe2ff2aec"
    }
    with open(processed_data_dir / "embedding_info.json", "w") as f:
        json.dump(embedding_info, f)
    
    # Save vector data for Qdrant - we need to extract vectors and metadata
    print("Saving Qdrant vector data...")
    
    # Get all vectors and their metadata from Qdrant
    vectors_data = []
    for doc in documents:
        # We'll need to re-embed in the app since we can't easily serialize Qdrant's in-memory store
        vectors_data.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open(processed_data_dir / "vector_data.json", "w", encoding="utf-8") as f:
        json.dump(vectors_data, f, ensure_ascii=False, indent=2)
    
    print("All processed data saved successfully!")

def create_processed_data():
    """Create all processed data files needed for the RAG system"""
    
    # Ensure the processed_data directory exists
    processed_data_dir = Path("AB_AI_RAG_Agent/data/processed_data")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the improved chunks from the Jupyter notebook
    chunks_source_path = Path("all_chunks_95percentile.json")
    
    if not chunks_source_path.exists():
        raise FileNotFoundError(f"Source chunks file not found: {chunks_source_path}")
    
    print("Loading improved chunks from Jupyter notebook...")
    with open(chunks_source_path, 'r') as f:
        chunk_data = json.load(f)
    
    # Convert to Document objects
    documents = []
    for chunk in chunk_data:
        doc = Document(
            page_content=chunk['page_content'],
            metadata=chunk['metadata']
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} chunks")
    
    # Save documents as pickle
    chunks_path = processed_data_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved chunks to {chunks_path}")
    
    # Create BM25 retriever
    print("Creating BM25 retriever...")
    texts = [doc.page_content for doc in documents]
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Create BM25 retriever object
    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=[doc.metadata for doc in documents])
    
    # Save BM25 retriever
    bm25_path = processed_data_dir / "bm25_retriever.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"Saved BM25 retriever to {bm25_path}")
    
    # Initialize embedding model
    print("Initializing embedding model...")
    model_name = "kamkol/ab_testing_finetuned_arctic_ft-36dfff22-0696-40d2-b3bf-268fe2ff2aec"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # Save embedding model info
    embedding_info = {"model_name": model_name}
    embedding_info_path = processed_data_dir / "embedding_info.json"
    with open(embedding_info_path, "w") as f:
        json.dump(embedding_info, f)
    print(f"Saved embedding info to {embedding_info_path}")
    
    # Pre-compute embeddings for all documents
    print("Pre-computing embeddings (this may take a while)...")
    embedded_docs = []
    
    # Process in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Extract text
        texts = [doc.page_content for doc in batch]
        
        # Get embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Store with metadata
        for j, doc in enumerate(batch):
            embedded_docs.append({
                "id": i + j,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embeddings[j]
            })
        
        # Print progress
        print(f"Embedded {min(i+batch_size, len(documents))}/{len(documents)} chunks")
    
    # Save the embedded docs for fast loading
    embedded_docs_path = processed_data_dir / "embedded_docs.pkl"
    with open(embedded_docs_path, "wb") as f:
        pickle.dump(embedded_docs, f)
    print(f"Saved embedded docs to {embedded_docs_path}")
    
    print(f"Processing complete! All files saved to {processed_data_dir}")
    print(f"Files created:")
    print(f"  - chunks.pkl ({len(documents)} documents)")
    print(f"  - bm25_retriever.pkl")
    print(f"  - embedding_info.json")
    print(f"  - embedded_docs.pkl ({len(embedded_docs)} embedded documents)")

if __name__ == "__main__":
    create_processed_data() 