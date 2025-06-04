import os
import pickle
import streamlit as st
import json
from pathlib import Path
from typing import Annotated, List, TypedDict, Dict, Any
import operator
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


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
required_keys = ["OPENAI_API_KEY", "COHERE_API_KEY"]
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


# Define prompts
RAG_PROMPT = """
CONTEXT:
{context}
QUERY:
{question}
You are a helpful assistant. Use the available context to answer the question. Do not use your own knowledge! If you cannot answer the question based on the context, you must say "I don't know".
"""

REPHRASE_QUERY_PROMPT = """
QUERY:
{question}
You are a helpful assistant. Rephrase the provided query to be more specific and to the point in order to improve retrieval in our RAG pipeline about AB Testing.
"""

FOLLOW_UP_PROMPT = """
You are an expert question architect. Based ONLY on the final answer below, generate 3 concise, relevant follow-up questions that:
- Probe deeper into specific details mentioned
- Explore related concepts or implications
- Ask for practical applications or examples
- Do not repeat the final answer
Format output as JSON with a "questions" key containing the list. Never include markdown.
Final Answer:
{response}
JSON:
"""

# Define the GraphState for the LangGraph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sources: Annotated[List[Dict[str, Any]], operator.add]  # Track all sources
    follow_up_questions: List[str]  # Only want the most recent follow up questions

# Initialize the AB Testing QA system
@st.cache_resource
def initialize_qa_system(_reranker):
    """Initialize the AB Testing QA system"""
    # Create a retriever
    retriever = _reranker
    
    # Initialize prompt templates
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    rephrase_query_prompt = ChatPromptTemplate.from_template(REPHRASE_QUERY_PROMPT)
    follow_up_prompt = ChatPromptTemplate.from_template(FOLLOW_UP_PROMPT)
    
    # Initialize models (with streaming enabled)
    openai_chat_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0, streaming=True)
    # Use gpt-4.1-mini for improving latency
    follow_up_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    
    # Define the RAG chain node
    def rag_chain_node(state: GraphState) -> GraphState:
        query = state["messages"][-1].content
        
        # 1. Retrieve documents. It's a best practice to return contexts in ascending order
        docs_descending = retriever.get_relevant_documents(query)
        docs = docs_descending[::-1]
        
        # 2. Extract sources from the documents
        sources = []
        for doc in docs:
            source_path = doc.metadata.get("source", "")
            filename = source_path.split("/")[-1] if "/" in source_path else source_path
            
            sources.append({
                "title": filename,
                "section": doc.metadata.get("section_title", "unknown"),
            })
        
        # 3. Use a simplified RAG chain without retrieval
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format the prompt with context and query
        formatted_prompt = rag_prompt.format(context=context, question=query)
        
        # Get a placeholder for streaming
        if "streaming_container" in state:
            streaming_container = state["streaming_container"]
            full_response = []
            
            # Stream the response
            for chunk in openai_chat_model.stream(formatted_prompt):
                content = chunk.content
                full_response.append(content)
                streaming_container.markdown("".join(full_response))
            
            response_text = "".join(full_response)
        else:
            # If no streaming container provided, fall back to non-streaming
            response = openai_chat_model.invoke(formatted_prompt)
            response_text = StrOutputParser().invoke(response)
        
        return {
            "messages": [AIMessage(content=response_text)],
            "sources": sources
        }
    
    # Define the tools
    @tool
    def retrieve_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
        ):
        """Use Retrieval Augmented Generation to retrieve information about AB Testing."""
        # 1. Retrieve documents. It's a best practice to return contexts in ascending order
        docs_descending = retriever.get_relevant_documents(query)
        docs = docs_descending[::-1]
        
        # 2. Extract and store sources
        sources = []
        for doc in docs:
            source_path = doc.metadata.get("source", "")
            filename = source_path.split("/")[-1] if "/" in source_path else source_path
            
            sources.append({
                "title": filename,
                "section": doc.metadata.get("section_title", "unknown"),
            })
        
        # Store sources for later access
        retrieve_information.last_sources = sources
        
        # 3. Return just the formatted document contents
        formatted_content = "\n\n".join([f"Retrieved Information: {i+1}\n{doc.page_content}" 
                                        for i, doc in enumerate(docs)])
        return formatted_content
    
    @tool
    def retrieve_information_with_rephrased_query(
        query: Annotated[str, "query to be rephrased before asking the retrieve information tool"]
        ):
        """This tool will intelligently rephrase your AB testing query and then will use Retrieval Augmented Generation to retrieve information about the rephrased query."""
        
        # 1. Rephrase the query first
        rephrased_query = rephrase_query_prompt.format(question=query)
        rephrased_query = openai_chat_model.invoke(rephrased_query)
        rephrased_query = StrOutputParser().invoke(rephrased_query)
        
        # 2. Retrieve documents using the rephrased query. It's a best practice to return contexts in ascending order
        docs_descending = retriever.get_relevant_documents(rephrased_query)
        docs = docs_descending[::-1]
        
        # 3. Extract and store sources
        sources = []
        for doc in docs:
            source_path = doc.metadata.get("source", "")
            filename = source_path.split("/")[-1] if "/" in source_path else source_path
            
            sources.append({
                "title": filename,
                "section": doc.metadata.get("section_title", "unknown"),
            })
        
        # Store sources for later access
        retrieve_information_with_rephrased_query.last_sources = sources
        
        # 4. Return formatted content with rephrased query
        formatted_content = f"Rephrased query: {rephrased_query}\n\n" + "\n\n".join(
            [f"Retrieved Information: {i+1}\n{doc.page_content}" for i, doc in enumerate(docs)]
        )
        return formatted_content
    
    # Define follow up questions node
    def follow_up_questions_node(state: GraphState) -> GraphState:
        # Get last AI response from messages
        last_response = state["messages"][-1].content
        
        # Format prompt using template
        formatted_prompt = follow_up_prompt.format(response=last_response)
        
        response = follow_up_llm.invoke(formatted_prompt)
        response_text = StrOutputParser().invoke(response)
        
        # Parse JSON output
        try:
            questions_data = json.loads(response_text)
            follow_up_questions = questions_data.get("questions", [])[:3]
        except Exception as e:
            print(f"Error parsing follow-up questions: {e}")
            follow_up_questions = []

        return {
            "follow_up_questions": follow_up_questions
        }
    
    # Create tool belt
    tool_belt = [
        retrieve_information,
        retrieve_information_with_rephrased_query,
        ArxivQueryRun(),
    ]
    
    # Create tool node
    tool_node = ToolNode(tool_belt)
    
    # Setup agent model (with streaming)
    model = ChatOpenAI(model="gpt-4.1", temperature=0, streaming=True)
    model = model.bind_tools(tool_belt)
    
    # Define model calling function
    def call_model(state):
        messages = state["messages"]
        
        # Check if we have a streaming container
        streaming_container = state.get("streaming_container", None)
        
        # For streaming response
        if streaming_container:
            full_response = []
            
            # Stream the response
            for chunk in model.stream(messages):
                if hasattr(chunk, "content") and chunk.content is not None:
                    content = chunk.content
                    full_response.append(content)
                    streaming_container.markdown("".join(full_response))
            
            # Get the final response
            if full_response:
                response = AIMessage(content="".join(full_response))
            else:
                # Fall back to non-streaming if needed
                response = model.invoke(messages)
        else:
            # Non-streaming fallback
            response = model.invoke(messages)
        
        # Extract sources if available by examining the last message
        sources = []
        if len(messages) > 0:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                
                # Check for specific patterns in the content
                if isinstance(content, str):
                    if "Rephrased query:" in content and hasattr(retrieve_information_with_rephrased_query, "last_sources"):
                        sources = retrieve_information_with_rephrased_query.last_sources
                    elif "Retrieved Information:" in content and hasattr(retrieve_information, "last_sources"):
                        sources = retrieve_information.last_sources
                    elif "Title:" in content and "Authors:" in content:  # ArxivQueryRun pattern
                        # Extract paper titles and IDs from ArXiv results
                        import re
                        titles = re.findall(r"Title: (.*?)$", content, re.MULTILINE)
                        # Try to extract the arxiv IDs - match both old and new format IDs
                        arxiv_ids = re.findall(r"URL: https://arxiv\.org/abs/([0-9v\.]+)", content) 
                        
                        sources = []
                        for i, title in enumerate(titles):
                            source = {"title": title, "type": "arxiv_paper"}
                            # Add arxiv_id if available
                            if i < len(arxiv_ids):
                                source["arxiv_id"] = arxiv_ids[i]
                            sources.append(source)
        
        # Return both the response and sources
        return {
            "messages": [response],
            "sources": sources
        }
    
    # Define continuation condition
    def should_continue(state):
        last_message = state["messages"][-1]
        
        if last_message.tool_calls:
            return "action"
        
        return "follow_up_questions_from_llm"
    
    # Define helpfulness check
    def NonAB_Testing_or_helpful_RAG_or_continue(state):
        initial_query = state["messages"][0]
        final_response = state["messages"][-1]
        
        prompt_template = """\
        
        Given an initial query, determine if the initial query is related to AB Testing (even vaguely e.g. statistics, A/B testing, etc.) or not. If not related to AB Testing, return 'Y'. If related to AB Testing, then given the initial query and a final response, determine if the final response is extremely helpful or not. If extremely helpful, return 'Y'. If not extremely helpful, return 'N'.
        Initial Query:
        {initial_query}
        Final Response:
        {final_response}"""
        
        prompt_template = PromptTemplate.from_template(prompt_template)
        
        helpfulness_check_model = ChatOpenAI(model="gpt-4.1", temperature=0)
        
        helpfulness_chain = prompt_template | helpfulness_check_model | StrOutputParser()
        
        helpfulness_response = helpfulness_chain.invoke({
            "initial_query": initial_query.content, 
            "final_response": final_response.content
        })
        
        if "Y" in helpfulness_response:
            return "follow_up_questions_from_llm"
        else:
            return "agent"
    
    # Create graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("Initial_RAG", rag_chain_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("follow_up_questions_from_llm", follow_up_questions_node)
    
    # Set entry point
    graph.set_entry_point("Initial_RAG")
    
    # Add edges
    graph.add_conditional_edges(
        "Initial_RAG",
        NonAB_Testing_or_helpful_RAG_or_continue,
        {
            "agent": "agent",
            "follow_up_questions_from_llm": "follow_up_questions_from_llm"
        }
    )
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "follow_up_questions_from_llm": "follow_up_questions_from_llm"
        }
    )
    
    graph.add_edge("action", "agent")
    graph.add_edge("follow_up_questions_from_llm", END)
    
    # Compile graph
    return graph.compile()

# Streamlit interface
st.markdown(
    "<h1>📊 A/B<sub><span style='color:green;'>AI</span></sub></h1>",
    unsafe_allow_html=True
)
st.markdown("""
A/B<sub><span style='color:green;'>AI</span></sub> is a specialized agent that answers your A/B Testing questions using a thorough collection of Ron Kohavi's work, including his book, papers, and LinkedIn posts. If A/B<sub><span style='color:green;'>AI</span></sub> can't answer your questions using this collection, it will then search Arxiv. It has been trained to only answer based on the sources it retrieves. Let's begin!
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
    qa_system = initialize_qa_system(vectorstore)
    
    # Clear loading indicator
    loading_placeholder.empty()
except Exception as e:
    st.error(f"Error initializing the system: {str(e)}")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            

# Chat input
query = st.chat_input("Ask me anything about A/B Testing...")

if query:
    # Display user message
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
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
                "streaming_container": streaming_container  # Pass the container for streaming
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
                    # Extract arXiv ID from Entry ID metadata
                    entry_id = source.get('Entry ID', '')  # This is the key field containing the ID
                    if entry_id:
                        # Extract arXiv ID from format like "http://arxiv.org/abs/2404.19647v1"
                        arxiv_id = entry_id.split('/abs/')[-1].split('v')[0]  # Removes version suffix
                        sources_text += f"- ArXiv paper: [{source['title']}](https://arxiv.org/abs/{arxiv_id})\n"
                    else:
                        sources_text += f"- ArXiv paper: {source['title']}\n"

                else:
                    # Handle retrieval sources (Ron Kohavi's work)
                    # Remove .pdf extension if present
                    title = source['title'].replace('.pdf', '')
                        
                    # Create a unique identifier for this source
                    source_id = f"{title}|{source['section']}"
                        
                    # Only add if not already added
                    if source_id not in unique_sources:
                        unique_sources.add(source_id)
                        sources_text += f"- Ron Kohavi: {title}, Section: {source['section']}\n"
                

            # Final display with the complete answer and sources
            answers_and_sources = answer
            
            # Only add sources and follow-up questions if answer is not "I don't know"
            if "I don't know" not in answer:
                if sources_text:
                    answers_and_sources += "\n\n" + "**Sources:**" + "\n\n" + sources_text
                
                # Add follow-up questions if available
                if follow_up_questions:
                    follow_up_text = "\n\n**Follow-up Questions:**\n\n"
                    for i, question in enumerate(follow_up_questions):
                        follow_up_text += f"{i+1}. {question}\n"
                    answers_and_sources += follow_up_text
                
            streaming_container.markdown(answers_and_sources)
        
            
    
    # Save to chat history (still save sources for internal use, even if not displayed)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answers_and_sources,
        "sources": sources,
        "follow_up_questions": follow_up_questions
    })

