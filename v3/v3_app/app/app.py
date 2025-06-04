import os
import pickle
import streamlit as st
import json
from pathlib import Path
from typing import Annotated, List, TypedDict, Dict, Any, Literal, Optional, NotRequired
import operator
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field
import asyncio
import requests
from tavily import TavilyClient, AsyncTavilyClient
from langchain_community.retrievers import ArxivRetriever
from enum import Enum
from dataclasses import dataclass, fields
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.constants import Send
from langgraph.types import interrupt, Command
from IPython.display import Markdown, display
import uuid



# Debug function to print directory information at startup
def debug_startup_info():
    """Print debug information at startup to help identify file locations"""
    print("=" * 50)
    print("DEBUG STARTUP INFO")
    print("=" * 50)
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    # Check for the data directory
    print("\nChecking for data directory:")
    if os.path.exists("data"):
        print("Found 'data' directory in current directory")
        print(f"Contents: {os.listdir('data')}")
        if os.path.exists("data/processed_data"):
            print(f"Contents of data/processed_data: {os.listdir('data/processed_data')}")
    # Check common paths that might exist in Hugging Face Spaces
    common_paths = [
        "/data",
        "/repository",
        "/app",
        "/app/data",
        "/repository/data",
        "/app/repository",
        "AB_AI_RAG_Agent/data"
    ]
    print("\nChecking common paths:")
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found path: {path}")
            print(f"Contents: {os.listdir(path)}")
            # Check for processed_data subdirectory
            processed_path = os.path.join(path, "processed_data")
            if os.path.exists(processed_path):
                print(f"Found processed_data at: {processed_path}")
                print(f"Contents: {os.listdir(processed_path)}")
    print("=" * 50)


# Run debug info at startup
debug_startup_info()

# Enable debugging for file paths
import os
DEBUG_FILE_PATHS = True

def debug_paths():
    if DEBUG_FILE_PATHS:
        print("Current working directory:", os.getcwd())
        print("Files in /data:", os.listdir("/data") if os.path.exists("/data") else "Not found")
        print("Files in /data/processed_data:", os.listdir("/data/processed_data") if os.path.exists("/data/processed_data") else "Not found")
        for path in ["/repository", "/app", "/app/data"]:
            if os.path.exists(path):
                print(f"Files in {path}:", os.listdir(path))

# Load environment variables
load_dotenv()


# Check for required API keys
required_keys = ["COHERE_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]
if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}. Please set them as environment variables.")
    st.stop()


# Custom vector store implementation
class CustomVectorStore(VectorStore):
    def __init__(self, embedded_docs, embedding_model):
        self.embedded_docs = embedded_docs
        self.embedding_model = embedding_model

    def similarity_search_with_score(self, query, k=5):
        # Get the query embedding
        query_embedding = self.embedding_model.embed_query(query)
        # Calculate similarity scores
        results = []
        for doc in self.embedded_docs:
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, doc["embedding"])
            results.append((doc, similarity))
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        # Convert to Document objects and return top k
        documents_with_scores = []
        for doc, score in results[:k]:
            document = Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            )
            documents_with_scores.append((document, score))
        return documents_with_scores

    def similarity_search(self, query, k=5):
        docs_with_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_with_scores]


    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """Implement required abstract method from VectorStore base class."""
        # Create embeddings for the texts
        embeddings = embedding.embed_documents(texts)
        # Create embedded docs format
        embedded_docs = []
        for i, (text, embedding_vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            embedded_docs.append({
                "text": text,
                "embedding": embedding_vector,
                "metadata": metadata
            })
        # Return an instance with the embedded docs
        return cls(embedded_docs, embedding)


def find_processed_data():
    """Find the processed_data directory path"""
    possible_paths = [
        "data/processed_data",
        "../data/processed_data",
        "/data/processed_data",
        "/app/data/processed_data",
        "./data/processed_data",
        "/repository/data/processed_data",
        "AB_AI_RAG_Agent/data/processed_data"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            required_files = ["chunks.pkl", "bm25_retriever.pkl", "embedding_info.json", "embedded_docs.pkl"]
            if all(os.path.exists(os.path.join(path, f)) for f in required_files):
                print(f"Found processed_data at: {path}")
                return path
    raise FileNotFoundError("Could not find processed_data directory with required files")



@st.cache_resource
def initialize_vectorstore():
    """Initialize the hybrid retriever system with Cohere reranking"""
    try:
        # Find processed data directory
        processed_data_path = find_processed_data()
        
        # Load documents
        with open(os.path.join(processed_data_path, "chunks.pkl"), "rb") as f:
            documents = pickle.load(f)
        
        # Load BM25 retriever
        with open(os.path.join(processed_data_path, "bm25_retriever.pkl"), "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 5 
        
        # Load embedding model info
        with open(os.path.join(processed_data_path, "embedding_info.json"), "r") as f:
            embedding_info = json.load(f)
        
        # Load pre-computed embedded docs
        with open(os.path.join(processed_data_path, "embedded_docs.pkl"), "rb") as f:
            embedded_docs = pickle.load(f)
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_info["model_name"]
        )
        
        # Create custom vectorstore using pre-computed embeddings
        vectorstore = CustomVectorStore(embedded_docs, embedding_model)
        qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create hybrid retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[qdrant_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        
        # Create Cohere reranker
        cohere_rerank = CohereRerank(
            model="rerank-english-v3.0",
            top_n=5,
        )
        
        reranker = ContextualCompressionRetriever(
            base_compressor=cohere_rerank,
            base_retriever=hybrid_retriever
        )
        
        print("Successfully initialized retriever system")
        return reranker, documents
    except Exception as e:
        st.error(f"Error initializing retrievers: {str(e)}")
        st.stop()


# Streamlit interface
st.markdown(
    "<h1>📊 A/B<sub><span style='color:green;'>AI</span></sub></h1>",
    unsafe_allow_html=True
)
st.markdown("""
A/B<sub><span style='color:green;'>AI</span></sub> is a specialized agent with 2 modes: Q&A Mode and Report-Generating Mode. The Q&A Mode answers your A/B Testing questions and the Report-Generating Mode generates comprehensive reports on your provided A/B testing topics. Both modes use a thorough collection of Ron Kohavi's work, including his book, papers, and LinkedIn posts. If the Q&A Mode can't answer your questions using this collection, it will then search arXiv. For each section of the Report-Generating Mode's report, if it can't answer your questions using this collection, it will then search arXiv. If that's not enough, it will finally search the web using Tavily. It provides ALL sources, section by section. Both modes have been trained to only respond based on the sources they retrieve. You can toggle between both modes using the sidebar on the left. Let's begin!
""", unsafe_allow_html=True)


# Initialize the system
try:
    # Show loading indicator
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        import time
        for dots in [".", "..", "..."]:
            loading_placeholder.text(f"Loading{dots}")
            time.sleep(0.2)
    
    # Initialize components (but hide the details)
    vectorstore, chunks = initialize_vectorstore()

       
    # Clear loading indicator
    loading_placeholder.empty()
except Exception as e:
    st.error(f"Error initializing the system: {str(e)}")
    st.stop()

# Add mode toggle in sidebar
with st.sidebar:
    st.markdown("### A/B<sub><span style='color:green;'>AI</span></sub> Mode", unsafe_allow_html=True)
    mode_version = st.radio(
        "Choose Mode:",
        ["Q&A Mode", "Report-Generating Mode"],
        index=0  # Default to Q&A Mode
    )


# Define mode functions
def run_qa_mode():
    # Import Q&A system from app_1
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from app_1 import initialize_qa_system
    
    # Initialize QA system
    qa_system = initialize_qa_system(vectorstore)
    
    # Initialize session state for chat history
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []


    # Clear and create chat area
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.qa_messages):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])


    # Chat input - only for Q&A mode
    query = st.chat_input("Ask me anything about A/B Testing...", key="qa_mode_input")

    if query:
        # Display user message
        st.chat_message("user").write(query)
        st.session_state.qa_messages.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("Thinking..."):
            # Create a placeholder for streaming output
            with st.chat_message("assistant"):
                streaming_container = st.empty()
            
                # Create input state for the graph with streaming container
                input_state = {
                    "messages": [HumanMessage(content=query)],
                    "sources": [],
                    "follow_up_questions": [],
                    "streaming_container": streaming_container
                }
            
                # Execute graph
                result = qa_system.invoke(input_state)
            
                # Extract result
                answer = result["messages"][-1].content
                sources = result["sources"]
                follow_up_questions = result.get("follow_up_questions", [])

                # Process sources to remove duplicates and format properly
                unique_sources = set()
                sources_text = ""
                
                for source in sources:
                    if "type" in source and source["type"] == "arxiv_paper":
                        entry_id = source.get('Entry ID', '')
                        if entry_id:
                            arxiv_id = entry_id.split('/abs/')[-1].split('v')[0]
                            sources_text += f"- ArXiv paper: [{source['title']}](https://arxiv.org/abs/{arxiv_id})\n"
                        else:
                            sources_text += f"- ArXiv paper: {source['title']}\n"
                    else:
                        title = source['title'].replace('.pdf', '')
                        source_id = f"{title}|{source['section']}"
                        if source_id not in unique_sources:
                            unique_sources.add(source_id)
                            sources_text += f"- Ron Kohavi: {title}, Section: {source['section']}\n"

                # Final display with the complete answer and sources
                answers_and_sources = answer
                
                if "I don't know" not in answer:
                    if sources_text:
                        answers_and_sources += "\n\n" + "**Sources:**" + "\n\n" + sources_text
                    
                    if follow_up_questions:
                        follow_up_text = "\n\n**Follow-up Questions:**\n\n"
                        for i, question in enumerate(follow_up_questions):
                            follow_up_text += f"{i+1}. {question}\n"
                        answers_and_sources += follow_up_text
                    
                streaming_container.markdown(answers_and_sources)
        
        # Save to chat history
        st.session_state.qa_messages.append({
            "role": "assistant", 
            "content": answers_and_sources,
            "sources": sources,
            "follow_up_questions": follow_up_questions
        })

def run_report_mode():
    # Import report system from app_2
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from app_2 import initialize_report_system
    import asyncio
    
    # Initialize report system
    report_system = initialize_report_system(vectorstore)
    
    # Initialize session state for chat history
    if "report_messages" not in st.session_state:
        st.session_state.report_messages = []


    # Clear and create chat area
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.report_messages):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])


    # Chat input - only for Report mode
    query = st.chat_input("Please give me a topic on anything regarding A/B Testing...", key="report_mode_input")

    if query:
        # Display user message
        st.chat_message("user").write(query)
        st.session_state.report_messages.append({"role": "user", "content": query})

        # Create assistant container immediately
        with st.chat_message("assistant"):
            report_placeholder = st.empty()
        
        # Start new report generation
        def start_new_report(topic, report_placeholder):
            """Start a new report generation process"""
            with st.spinner("Generating comprehensive report...This may take about 3-7 minutes."):
                
                # Create input state
                input_state = {"topic": topic}
                
                # Run graph to completion
                try:
                    config = {}

                    # Use asyncio.run to handle async function
                    async def run_graph_to_completion(input_state, config):
                        """Run the graph to completion"""
                        result = await report_system.ainvoke(input_state, config)
                        return result
                    
                    result = asyncio.run(run_graph_to_completion(input_state, config))
                    
                    if result.get("ab_testing_check") == False:
                        # Not AB testing related
                        response = result.get("final_report", "This topic is not related to A/B testing.")
                        report_placeholder.markdown(response)
                        return response
                    else:
                        # AB testing related - show final report
                        final_report = result.get("final_report", "")
                        if final_report:
                            final_content = f"## 📄 Final Report\n\n{final_report}"
                            report_placeholder.markdown(final_content)
                            return final_content
                        else:
                            error_msg = "No report was generated."
                            report_placeholder.error(error_msg)
                            return None
                            
                except Exception as e:
                    error_msg = f"Error generating report: {str(e)}"
                    report_placeholder.error(error_msg)
                    return None
        
        # Start new report generation with placeholder
        final_content = start_new_report(query, report_placeholder)
        
        # Add to session state only after completion
        if final_content:
            st.session_state.report_messages.append({
                "role": "assistant", 
                "content": final_content
            })

# Track mode changes without using st.rerun()
if "current_mode" not in st.session_state:
    st.session_state.current_mode = mode_version

# Only update mode if it actually changed
if st.session_state.current_mode != mode_version:
    st.session_state.current_mode = mode_version

    # Clear conflicting widget/input states
    for key in ["qa_mode_input", "report_mode_input", "qa_messages", "report_messages"]:
        if key in st.session_state:
            del st.session_state[key]

# Call the appropriate mode function
if mode_version == "Q&A Mode":
    run_qa_mode()
else:  # Report-Generating Mode
    run_report_mode()






